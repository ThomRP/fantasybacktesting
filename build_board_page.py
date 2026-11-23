"""Generate a minimalist, old-school dark-mode self-contained HTML dashboard (`board_2026_27.html`)
covering ALL out-of-sample seasons:
  - 6 CV Folds (2018-19 .. 2023-24)
  - 2 Untouched Holdout Seasons (2024-25, 2025-26)
  - Live Out-of-Sample Season (2026-27 through latest games)

Features:
  - Season switcher (CV / Holdout / Live) with strict zero-leakage expanding-window training
  - Real-time game-by-game season replay button (4 games/sec = 0.25s/game) + draggable slider
  - Interactive old-school SVG edge trajectory chart (click-to-scrub)
  - Live-updating Total Edge & FPTS boxes, Top-150 Stage 2 / Stage 1 / Deep Sleeper tables

Usage:
  .\\.venv\\Scripts\\python.exe build_board_page.py
"""

import json
import math
from collections import defaultdict
from html import escape
from pathlib import Path
from statistics import mean, median

import numpy as np

import draft_engine as de
from compare_early_season import read_in_progress_season


ANCHOR_RANKS = (
    list(range(1, 21))
    + list(range(25, 155, 5))
    + [160, 175, 200, 225, 250]
)


def _sample_anchors(adj_lb):
    n = len(adj_lb)
    if n == 0:
        return [0.0] * len(ANCHOR_RANKS)
    return [round(float(adj_lb[min(rk - 1, n - 1)]), 2) for rk in ANCHOR_RANKS]


def load_season_timeline(folder: Path, scoring: dict, is_live: bool = False):
    """Build exact per-team-game checkpoints (1..max_g) and per-player sparse step box scores."""
    game_dates = {}
    games_per_date = defaultdict(int)
    for g in de.read_csv(folder / "games.csv"):
        if not de.regular_season_game(g):
            continue
        gid = g.get("game_id")
        gd = (g.get("game_date") or "")[:10]
        if not gid or not gd:
            continue
        # For live in-progress season, require completed game scores
        if is_live:
            hs = de._safe_float(g.get("home_score"))
            as_ = de._safe_float(g.get("away_score"))
            if hs is None or as_ is None or (hs == 0 and as_ == 0):
                continue
        game_dates[gid] = gd
        games_per_date[gd] += 1

    dates_sorted = sorted(games_per_date)
    date_to_idx = {d: i for i, d in enumerate(dates_sorted)}

    # Collect per-player box scores keyed by date_idx
    # Use list of (date_idx, pts) per player (a player plays at most 1 game per date)
    player_daily = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))  # aid -> di -> [gp, pts]
    seen_game_dates = set()
    for gid, aid, _name, stats, _pos, _starter, _tid in de.player_box_scores(folder, game_dates):
        gd = game_dates[gid]
        di = date_to_idx[gd]
        pts = de.score_box_stats(stats, scoring)
        slot = player_daily[aid][di]
        slot[0] += 1
        slot[1] += pts
        seen_game_dates.add(gd)

    # Filter dates_sorted to dates that actually have box scores (important for live season)
    if is_live and seen_game_dates:
        dates_sorted = [d for d in dates_sorted if d in seen_game_dates]
        date_to_idx_new = {d: i for i, d in enumerate(dates_sorted)}
        remapped = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
        for aid, dmap in player_daily.items():
            for old_di, val in dmap.items():
                # Recover date from games_per_date? Easier: build old_idx_to_date first
                pass
        # Rebuild cleanly using date strings
        dates_all = sorted(games_per_date)
        remapped = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
        for aid, dmap in player_daily.items():
            for old_di, val in dmap.items():
                gd = dates_all[old_di]
                if gd in date_to_idx_new:
                    remapped[aid][date_to_idx_new[gd]] = val
        player_daily = remapped
        date_to_idx = date_to_idx_new

    cum_games = []
    running = 0
    for d in dates_sorted:
        running += games_per_date[d]
        cum_games.append(running)

    avg_team_g_by_date = [(2.0 * cg) / 30.0 for cg in cum_games]
    final_avg_g = avg_team_g_by_date[-1] if avg_team_g_by_date else 1.0
    max_step = max(1, int(round(final_avg_g)))

    # Map each step s in 0..max_step-1 (representing Game 1..max_step) to a date cutoff index
    step_date_idx = []
    for g_num in range(1, max_step + 1):
        if g_num == max_step:
            di = len(dates_sorted) - 1
        else:
            target = g_num - 0.15
            di = next((i for i, av in enumerate(avg_team_g_by_date) if av >= target), len(dates_sorted) - 1)
            if step_date_idx and di < step_date_idx[-1]:
                di = step_date_idx[-1]
        step_date_idx.append(di)

    # Map each date_idx -> step_idx (0..max_step-1)
    di_to_step = [max_step - 1] * len(dates_sorted)
    prev_di = -1
    for s_idx, cut_di in enumerate(step_date_idx):
        for di in range(prev_di + 1, cut_di + 1):
            di_to_step[di] = s_idx
        prev_di = cut_di

    # Collapse per-player daily logs into per-step logs: step_idx -> {aid: (gp, pts)}
    step_events = [defaultdict(lambda: [0, 0.0]) for _ in range(max_step)]
    player_step_sparse = {}  # aid -> [[step_idx, gp, round(pts, 1)], ...]
    for aid, dmap in player_daily.items():
        per_s = defaultdict(lambda: [0, 0.0])
        for di, (gp, pts) in dmap.items():
            s_idx = di_to_step[di]
            per_s[s_idx][0] += gp
            per_s[s_idx][1] += pts
            step_events[s_idx][aid][0] += gp
            step_events[s_idx][aid][1] += pts
        player_step_sparse[aid] = [
            [s_idx, val[0], round(val[1], 1)]
            for s_idx, val in sorted(per_s.items())
        ]

    # Walk step-by-step across ALL NBA players to compute exact waiver, ref_g, and full-NBA leaderboard anchors
    cum_gp = defaultdict(int)
    cum_pts = defaultdict(float)
    checkpoints = []
    step_adj_by_id = []  # keep per-step exact adj_by_id + adj_lb for chart precomputation

    for s_idx in range(max_step):
        for aid, (gp, pts) in step_events[s_idx].items():
            cum_gp[aid] += gp
            cum_pts[aid] += pts

        di = step_date_idx[s_idx]
        reg_g = cum_games[di]
        avg_g = avg_team_g_by_date[di]
        if is_live:
            ref_g = max(1.0, de.REF_GAMES * min(avg_g, de.SEASON_GAMES) / de.SEASON_GAMES)
            season_pct = round(100.0 * avg_g / de.SEASON_GAMES, 1)
        else:
            # Completed seasons (handles 82g, 72g 2020-21, 64g 2019-20 smoothly ending at REF_GAMES=65.0)
            frac = min(1.0, avg_g / max(final_avg_g, 1.0))
            ref_g = max(1.0, de.REF_GAMES * frac)
            season_pct = round(100.0 * frac, 1)

        min_qual = max(1, int(round(20.0 * (ref_g / de.REF_GAMES))))
        qual_raw = sorted(
            (cum_pts[aid] / g for aid, g in cum_gp.items() if g >= min_qual),
            reverse=True,
        )
        if not qual_raw:
            qual_raw = sorted((cum_pts[aid] / g for aid, g in cum_gp.items() if g > 0), reverse=True)
        repl_fppg = qual_raw[min(149, len(qual_raw) - 1)] if qual_raw else 20.0
        waiver = de.WAIVER_FACTOR * repl_fppg

        adj_by_id = {}
        for aid, g in cum_gp.items():
            raw = cum_pts[aid] / g
            eff_g = min(float(g), ref_g)
            adj_by_id[aid] = (eff_g * raw + (ref_g - eff_g) * waiver) / ref_g
        adj_lb = sorted(adj_by_id.values(), reverse=True)
        top150_cut = adj_lb[min(149, len(adj_lb) - 1)] if adj_lb else waiver

        checkpoints.append({
            "g": s_idx + 1,
            "date": dates_sorted[di],
            "reg": int(reg_g),
            "avg_g": round(avg_g, 1),
            "pct": season_pct,
            "ref_g": round(ref_g, 3),
            "waiver": round(waiver, 2),
            "t150_cut": round(top150_cut, 2),
            "anchors": _sample_anchors(adj_lb),
        })
        step_adj_by_id.append((adj_by_id, adj_lb, waiver, dict(cum_pts)))

    return {
        "max_step": max_step,
        "checkpoints": checkpoints,
        "player_step_sparse": player_step_sparse,
        "step_adj_by_id": step_adj_by_id,
    }


def build_stage1_pool_for_season(
    target_year, prior_s, prev2_s, births_prior, target_profs, played_before_prior,
    adp_entries, tr_pv, tr_pr,
):
    prior_year = target_year - 1
    prior_name = de.season_label(prior_year)
    target_season = de.season_label(target_year)
    adp_by_key = {e["key"]: e for e in adp_entries}
    adp_rank_by_key = {e["key"]: i + 1 for i, e in enumerate(adp_entries)}

    s1_cands, seen_ids, seen_keys = [], set(), set()
    for aid, p in sorted(prior_s["players"].items(), key=lambda kv: kv[1]["adj_rank"]):
        if p["adj_rank"] > 350:
            continue
        k = de.name_key(p["display_name"])
        s1_cands.append(de.make_vet_feature_row(aid, prior_s, prev2_s, births_prior, target_season, prior_name))
        seen_ids.add(aid)
        seen_keys.add(k)

    if prev2_s is not None:
        for aid, p2 in sorted(prev2_s["players"].items(), key=lambda kv: kv[1]["adj_rank"]):
            if aid in seen_ids or p2["adj_rank"] > 60:
                continue
            k = de.name_key(p2["display_name"])
            if k in seen_keys:
                continue
            s1_cands.append({
                "prior_season": prior_name, "draft_season": target_season,
                "athlete_id": aid, "player": p2["display_name"],
                "player_type": "returning_absence", "positions": p2["primary_pos"],
                "prior_waiver_fppg": prior_s["waiver_fppg"],
                "prior_adj_fppg": None, "prior_adj_rank": None, "prior_raw_rank": None,
                "prior_fppg": None, "prior_mpg": None, "prior_fppm": None, "prior_games": 0,
                "prior_tp_pct": None, "age30_x_low_g": None, "young_x_post_mpg": None,
                "prev2_adj_fppg": p2["adj_fppg"], "draft_pick": None, "market_collapse_excess": 0.0,
            })
            seen_ids.add(aid)
            seen_keys.add(k)

    for aid, prof in target_profs.items():
        if aid in seen_ids or aid in played_before_prior:
            continue
        k = de.name_key(prof["display_name"])
        if k in seen_keys:
            continue
        pick, dy = prof.get("overall_pick"), prof.get("draft_year")
        if pick is not None and pick <= 45 and (dy is None or dy in (target_year, prior_year)):
            s1_cands.append(de.make_rookie_feature_row(
                aid=aid, player_name=prof["display_name"], prof=prof,
                prior_year=prior_year, prior_name=prior_name, target_season=target_season,
                prior_s=prior_s,
            ))
            seen_ids.add(aid)
            seen_keys.add(k)

    pure_res = de.fit_predict_pure_stats(tr_pv, tr_pr, s1_cands, prior_s["waiver_fppg"])
    pure_preds = pure_res["pure_stats"]
    for r in s1_cands:
        k = de.name_key(r["player"])
        e = adp_by_key.get(k)
        if e is not None:
            de.attach_adp_fields(r, e, prior_s)
            r["market_rank"] = adp_rank_by_key[k]
        else:
            r["adp"] = None
            r["market_rank"] = 999
            r["mkt_expected_adj_fppg"] = None

    guard_preds, guard_alerts = de.apply_optional_adp_layer(s1_cands, pure_preds, adp_weight=0.0, apply_collapse_guard=True)
    s1_raw_diffs = [
        float(guard_preds[i] - r["mkt_expected_adj_fppg"])
        for i, r in enumerate(s1_cands)
        if r.get("mkt_expected_adj_fppg") is not None and r["market_rank"] <= 150
    ]
    s1_peer_center = median(s1_raw_diffs) if s1_raw_diffs else 0.0
    for i, r in enumerate(s1_cands):
        r["model_proj"] = round(float(guard_preds[i]), 2)
        r["mkt_proj"] = round(float(r["mkt_expected_adj_fppg"]), 2) if r.get("mkt_expected_adj_fppg") is not None else None
        raw_d = (
            float(guard_preds[i] - r["mkt_expected_adj_fppg"])
            if r.get("mkt_expected_adj_fppg") is not None
            else float(guard_preds[i] - prior_s["waiver_fppg"])
        )
        r["edge_vs_peers"] = round(raw_d - s1_peer_center, 2)
        r["raw_edge"] = round(raw_d, 2)
        r["alert"] = guard_alerts[i]

    s1_ordered = sorted(s1_cands, key=lambda r: -r["model_proj"])
    for rk, r in enumerate(s1_ordered, 1):
        r["model_rank"] = rk
        r["rank_delta"] = int(r["market_rank"] - rk) if r["market_rank"] < 999 else None
    return s1_ordered


def build_live_2026_27_s2_cands(data, prior_s, prev2_s, births_prior, target_profs, prior_profs, played_before, adp_entries, early_data):
    target_season = "2026-27"
    target_year = 2026
    prior_year = 2025
    prior_name = "2025-26"

    ids_prior = defaultdict(set)
    for aid, p in prior_s["players"].items():
        for n in p["names"]:
            ids_prior[de.name_key(n)].add(aid)
    ids_prev2 = defaultdict(set)
    if prev2_s is not None:
        for aid, p in prev2_s["players"].items():
            for n in p["names"]:
                ids_prev2[de.name_key(n)].add(aid)
    ids_early = defaultdict(set)
    for aid, p in early_data["players"].items():
        for n in p.names:
            ids_early[de.name_key(n)].add(aid)
    ids_tprof = defaultdict(set)
    for aid, prof in target_profs.items():
        for n in prof["names"]:
            ids_tprof[de.name_key(n)].add(aid)
    ids_pprof = defaultdict(set)
    for aid, prof in prior_profs.items():
        for n in prof["names"]:
            ids_pprof[de.name_key(n)].add(aid)

    s2_cands, seen_ids = [], set()
    for entry in adp_entries:
        if len(s2_cands) >= 225 and entry["adp"] > 225:
            continue
        key = entry["key"]
        aid, ptype = None, "returning_absence"
        p_ids = ids_prior[key]
        if len(p_ids) == 1 and next(iter(p_ids)) not in seen_ids:
            (aid,) = p_ids
            ptype = "returning_vet"
        else:
            cand_e = ids_early.get(key, set())
            if len(cand_e) == 1 and next(iter(cand_e)) not in seen_ids:
                (aid,) = cand_e
            else:
                pc = [c for c in ids_tprof.get(key, set()) if c not in seen_ids]
                if len(pc) > 1:
                    rec_c = [c for c in pc if target_profs[c]["draft_year"] in (target_year, prior_year)]
                    if len(rec_c) == 1:
                        pc = rec_c
                if len(pc) == 1:
                    aid = pc[0]
            if aid is not None and aid not in played_before:
                ptype = "rookie"
            elif aid is None:
                prev_c = [c for c in ids_pprof.get(key, set()) if c in played_before]
                if len(prev_c) == 1:
                    aid = prev_c[0]
                    ptype = "returning_absence"
                elif len(ids_prev2.get(key, set())) == 1:
                    (aid,) = ids_prev2[key]
                    ptype = "returning_absence"
                else:
                    ptype = "rookie"
        if aid is not None:
            seen_ids.add(aid)

        if ptype == "returning_vet":
            row = de.make_vet_feature_row(aid, prior_s, prev2_s, births_prior, target_season, prior_name)
            row["player"] = entry["player"]
        elif ptype == "rookie":
            prof = target_profs.get(aid, {}) if aid is not None else {}
            row = de.make_rookie_feature_row(
                aid=aid or f"unmatched_{key}", player_name=entry["player"], prof=prof,
                prior_year=prior_year, prior_name=prior_name, target_season=target_season,
                prior_s=prior_s, adp_team_abbr=entry["adp_team_abbr"], pos_override=entry["adp_positions"],
            )
        else:
            p2_rec = prev2_s["players"].get(aid) if (prev2_s is not None and aid is not None) else None
            row = {
                "prior_season": prior_name, "draft_season": target_season,
                "athlete_id": aid or f"unmatched_{key}", "player": entry["player"],
                "player_type": "returning_absence",
                "positions": entry["adp_positions"] or (p2_rec["primary_pos"] if p2_rec else ("SG", "SF")),
                "prior_waiver_fppg": prior_s["waiver_fppg"],
                "prior_adj_fppg": None, "prior_adj_rank": None, "prior_raw_rank": None,
                "prior_fppg": None, "prior_mpg": None, "prior_fppm": None, "prior_games": 0,
                "prior_tp_pct": None, "age30_x_low_g": None, "young_x_post_mpg": None,
                "prev2_adj_fppg": p2_rec["adj_fppg"] if p2_rec else None,
                "draft_pick": None, "market_collapse_excess": 0.0,
            }
        row["market_rank"] = len(s2_cands) + 1
        de.attach_adp_fields(row, entry, prior_s)
        s2_cands.append(row)
    return s2_cands


def build_all_seasons_payload(data: Path, scoring_name: str = "default"):
    _, scoring = de.resolve_scoring(scoring_name)
    seasons_dirs, season_data, births_cache, profiles_cache = de.load_all_completed_seasons(data, scoring)
    pure_vet_by_s, pure_rk_by_s, adp_pool_by_s, played_before_all, _ = de.build_all_training_datasets(
        data, seasons_dirs, season_data, births_cache, profiles_cache,
    )

    # Load historical 12-team sim summary by season (if available) to display in season header
    sim_by_season = {}
    sim_csv = data / f"draft_engine_sim_by_season_{scoring_name}.csv"
    if sim_csv.is_file():
        for row in de.read_csv(sim_csv):
            if row.get("strategy") in ("residual_unified", "residual_wait"):
                sim_by_season[(row["season"], row["strategy"])] = {
                    "diff": float(row["espn_diff"]),
                    "rank": float(row["avg_rank"]),
                    "win_pct": float(row["win_pct"]),
                    "top4_pct": float(row["top4_pct"]),
                }

    eval_seasons = sorted(adp_pool_by_s)
    development, holdout = eval_seasons[:-2], eval_seasons[-2:]
    cv_seasons = development[2:][-6:]
    all_targets = [(s, "cv") for s in cv_seasons] + [(s, "holdout") for s in holdout]
    if (data / "2026-27" / "adp.csv").is_file():
        all_targets.append(("2026-27", "live"))

    seasons_payload = {}
    season_order = []

    for target_s, split in all_targets:
        target_year = int(target_s[:4])
        prior_year = target_year - 1
        prior_s = season_data[prior_year]
        prev2_s = season_data.get(prior_year - 1)
        is_live = (split == "live")

        # Strict zero-leakage training cutoff (matches draft_engine.py --backtest exactly!)
        if split == "holdout":
            cutoff = development[-1]
            tr_pv = [r for s, grp in pure_vet_by_s.items() if s <= cutoff for r in grp]
            tr_pr = [r for s, grp in pure_rk_by_s.items() if s <= cutoff for r in grp]
            tr_adp = [r for s, grp in adp_pool_by_s.items() if s <= cutoff for r in grp]
        else:
            tr_pv = [r for s, grp in pure_vet_by_s.items() if s < target_s for r in grp]
            tr_pr = [r for s, grp in pure_rk_by_s.items() if s < target_s for r in grp]
            tr_adp = [r for s, grp in adp_pool_by_s.items() if s < target_s for r in grp]

        adp_entries = de.read_adp_entries(data / target_s / "adp.csv")
        if is_live:
            early_data = read_in_progress_season(data / target_s)
            try:
                target_profs = de.parse_draft_info_from_profiles(de.load_player_profiles(data, target_year + 1))
            except Exception:
                target_profs = {}
            s2_cands = build_live_2026_27_s2_cands(
                data, prior_s, prev2_s, births_cache.get(prior_year, {}),
                target_profs, profiles_cache.get(prior_year, {}), played_before_all, adp_entries, early_data,
            )
            pb_prior = played_before_all
        else:
            target_profs = profiles_cache.get(target_year, {})
            s2_cands = [r.copy() for r in adp_pool_by_s[target_s]]
            pb_prior = set()
            for y in season_data:
                if y < target_year:
                    pb_prior.update(season_data[y]["players"].keys())

        # Fit Stage 2 Residual (lam=0.50 with late-ADP ramp)
        res_s2 = de.fit_predict_residual_unified(tr_adp, s2_cands, prior_s["waiver_fppg"], lam=0.50)
        peer_center = median(res_s2["scaled_edges"][i] for i, r in enumerate(s2_cands) if r["market_rank"] <= 150)
        for i, r in enumerate(s2_cands):
            r["model_proj"] = round(float(res_s2["residual_proj"][i]), 2)
            r["mkt_proj"] = round(float(r["mkt_expected_adj_fppg"]), 2)
            r["raw_edge"] = round(float(res_s2["raw_residuals"][i]), 2)
            r["edge_vs_peers"] = round(float(res_s2["scaled_edges"][i] - peer_center), 2)
            r["alert"] = "CHECK_INJURY" if r.get("market_collapse_excess", 0.0) > 0 else ""

        s2_ordered = sorted(s2_cands, key=lambda r: (-r["model_proj"], r["market_rank"]))
        for rk, r in enumerate(s2_ordered, 1):
            r["model_rank"] = rk
            r["rank_delta"] = int(r["market_rank"] - rk)

        # Fit Stage 1 Pure Stats (0% ADP, whole NBA)
        s1_ordered = build_stage1_pool_for_season(
            target_year, prior_s, prev2_s, births_cache.get(prior_year, {}),
            target_profs, pb_prior, adp_entries, tr_pv, tr_pr,
        )

        # Build exact game-by-game timeline for this season
        tl = load_season_timeline(data / target_s, scoring, is_live=is_live)

        s2_top150 = s2_ordered[:150]
        s1_top150 = s1_ordered[:150]
        mkt_top150 = sorted([r for r in s2_cands if r["market_rank"] <= 150], key=lambda r: r["market_rank"])
        deep_sleepers = sorted([r for r in s2_cands if r["market_rank"] > 150], key=lambda r: (-r["edge_vs_peers"], r["market_rank"]))
        t15 = sorted(mkt_top150, key=lambda r: -r["edge_vs_peers"])[:15]
        f15 = sorted(mkt_top150, key=lambda r: r["edge_vs_peers"])[:15]

        # Precompute exact time-series chart arrays across every step 0..max_step-1
        ts_t15_fpts_diff = []
        ts_top150_fpts_diff = []
        ts_corr_r = []
        ts_t15_vs_diff = []

        s2_150_ids = [r["athlete_id"] for r in s2_top150]
        mkt_150_ids = [r["athlete_id"] for r in mkt_top150]
        t15_ids = [r["athlete_id"] for r in t15]
        f15_ids = [r["athlete_id"] for r in f15]
        mkt_150_adps = [float(r["adp"]) for r in mkt_top150]
        mkt_150_raw_edges = [float(r["raw_edge"]) for r in mkt_top150]
        t15_adps = [float(r["adp"]) for r in t15]
        f15_adps = [float(r["adp"]) for r in f15]

        for s_idx in range(tl["max_step"]):
            adj_by_id, adj_lb, waiver_s, cum_pts_s = tl["step_adj_by_id"][s_idx]
            s2_f = sum(cum_pts_s.get(aid, 0.0) for aid in s2_150_ids)
            mk_f = sum(cum_pts_s.get(aid, 0.0) for aid in mkt_150_ids)
            ts_top150_fpts_diff.append(round(s2_f - mk_f, 1))

            t15_f = sum(cum_pts_s.get(aid, 0.0) for aid in t15_ids)
            f15_f = sum(cum_pts_s.get(aid, 0.0) for aid in f15_ids)
            ts_t15_fpts_diff.append(round(t15_f - f15_f, 1))

            t15_vs = sum(adj_by_id.get(aid, waiver_s) - de.adp_slot_points(adj_lb, a) for aid, a in zip(t15_ids, t15_adps))
            f15_vs = sum(adj_by_id.get(aid, waiver_s) - de.adp_slot_points(adj_lb, a) for aid, a in zip(f15_ids, f15_adps))
            ts_t15_vs_diff.append(round(t15_vs - f15_vs, 2))

            vs_vec = [adj_by_id.get(aid, waiver_s) - de.adp_slot_points(adj_lb, a) for aid, a in zip(mkt_150_ids, mkt_150_adps)]
            if np.std(vs_vec) > 1e-6 and np.std(mkt_150_raw_edges) > 1e-6:
                rc = float(np.corrcoef(mkt_150_raw_edges, vs_vec)[0, 1])
            else:
                rc = 0.0
            ts_corr_r.append(round(rc, 3))

        # Build compact per-season player lookup (deduplicated across s2_cands + s1_top150!)
        # Store game log once per athlete_id so JSON stays tiny
        needed_aids = {r["athlete_id"] for r in s2_cands} | {r["athlete_id"] for r in s1_top150}
        logs = {
            aid: tl["player_step_sparse"].get(aid, [])
            for aid in needed_aids
        }

        type_short = {"returning_vet": "VET", "rookie": "ROOK", "returning_absence": "RET"}

        def ser_rows(rows):
            out = []
            for r in rows:
                out.append({
                    "id": r["athlete_id"],
                    "rk": r["model_rank"],
                    "p": r["player"],
                    "pos": "/".join(r["positions"][:2]) if isinstance(r.get("positions"), (tuple, list)) else str(r.get("positions") or ""),
                    "t": type_short.get(r["player_type"], "VET"),
                    "mrk": r["market_rank"] if r["market_rank"] < 999 else None,
                    "adp": round(float(r["adp"]), 1) if r.get("adp") is not None else None,
                    "drk": r["rank_delta"],
                    "mp": r["mkt_proj"],
                    "e": r["edge_vs_peers"],
                    "re": r["raw_edge"],
                    "pr": r["model_proj"],
                    "al": r.get("alert", ""),
                })
            return out

        sim_g = sim_by_season.get((target_s, "residual_unified"))
        sim_w = sim_by_season.get((target_s, "residual_wait"))

        seasons_payload[target_s] = {
            "season": target_s,
            "split": split,
            "max_step": tl["max_step"],
            "prior_waiver": round(prior_s["waiver_fppg"], 2),
            "peer_center": round(float(peer_center), 2),
            "sim_greedy": sim_g,
            "sim_wait": sim_w,
            "checkpoints": tl["checkpoints"],
            "ts": {
                "t15_fpts": ts_t15_fpts_diff,
                "t15_vs": ts_t15_vs_diff,
                "top150_fpts": ts_top150_fpts_diff,
                "corr_r": ts_corr_r,
            },
            "logs": logs,
            "s2_all": ser_rows(s2_ordered),     # full 225 Stage-2 pool (top 150 + 151-225 sleepers + dropped)
            "s1_150": ser_rows(s1_top150),      # Stage-1 Pure Stats Top 150
        }
        season_order.append({"season": target_s, "split": split, "max_step": tl["max_step"]})
        print(
            f"  Built {target_s} ({split:7s}, {tl['max_step']:2d} steps) | "
            f"End Top150 FPTS diff: {ts_top150_fpts_diff[-1]:+8.1f} | "
            f"End Top15 Tgt-Fade FPTS: {ts_t15_fpts_diff[-1]:+8.1f} | End r={ts_corr_r[-1]:+.3f}"
        )

    return {
        "anchor_ranks": ANCHOR_RANKS,
        "season_order": season_order,
        "default_season": season_order[-1]["season"],
        "seasons": seasons_payload,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fantasy Basketball Draft Engine — Multi-Season Top 150 &amp; Real-Time Edge Replay</title>
<style>
  :root {
    --bg: #101215;
    --paper: #171a1f;
    --paper-hi: #1d2128;
    --ink: #e7e5dd;
    --muted: #8b919c;
    --rule: #292e37;
    --rule-strong: #464d5b;
    --row-alt: #13161b;
    --row-hover: #212630;
    --pos: #4ade80;
    --pos-bg: rgba(74, 222, 128, 0.12);
    --pos-bd: rgba(74, 222, 128, 0.30);
    --neg: #f87171;
    --neg-bg: rgba(248, 113, 113, 0.12);
    --neg-bd: rgba(248, 113, 113, 0.30);
    --warn: #fbbf24;
    --warn-bg: rgba(251, 191, 36, 0.13);
    --warn-bd: rgba(251, 191, 36, 0.35);
    --accent: #60a5fa;
    --accent-bg: rgba(96, 165, 250, 0.13);
    --accent-bd: rgba(96, 165, 250, 0.32);
    --live: #34d399;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 24px 18px 60px;
    background: var(--bg);
    color: var(--ink);
    font-family: Georgia, "Times New Roman", serif;
    line-height: 1.45;
  }
  .wrap { max-width: 1280px; margin: 0 auto; }
  header {
    border-bottom: 2px solid var(--rule-strong);
    padding-bottom: 14px;
    margin-bottom: 14px;
  }
  .kicker {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: var(--muted);
    margin-bottom: 4px;
  }
  h1 {
    margin: 0 0 6px;
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: #f3f1ea;
  }
  .subtitle {
    margin: 0;
    font-size: 13.5px;
    color: var(--muted);
  }
  .subtitle strong { color: var(--ink); }

  /* Season Switcher Bar */
  .season-bar {
    background: var(--paper);
    border: 1px solid var(--rule);
    padding: 10px 12px;
    margin-bottom: 12px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 14px;
  }
  .season-group {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 5px;
  }
  .season-group-lbl {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--muted);
    margin-right: 3px;
  }
  .season-btn {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
    background: var(--bg);
    border: 1px solid var(--rule);
    color: var(--muted);
    padding: 5px 9px;
    cursor: pointer;
  }
  .season-btn:hover { background: var(--row-hover); color: var(--ink); }
  .season-btn.holdout { border-color: rgba(251, 191, 36, 0.35); color: #e5c678; }
  .season-btn.live { border-color: rgba(74, 222, 128, 0.4); color: var(--pos); }
  .season-btn.active {
    background: var(--ink);
    border-color: var(--ink);
    color: #101215;
    font-weight: 700;
  }
  .season-sim-note {
    margin-left: auto;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11.5px;
    color: var(--muted);
  }
  .season-sim-note strong { color: var(--pos); }

  /* Real-Time Replay Panel */
  .replay-panel {
    background: var(--paper-hi);
    border: 1px solid var(--rule-strong);
    padding: 12px 14px;
    margin-bottom: 16px;
  }
  .replay-top {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 10px;
  }
  .replay-controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
  }
  .play-btn {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
    font-weight: 700;
    background: var(--pos-bg);
    border: 1px solid var(--pos);
    color: var(--pos);
    padding: 6px 13px;
    cursor: pointer;
    min-width: 138px;
  }
  .play-btn:hover { background: rgba(74, 222, 128, 0.22); }
  .play-btn.playing {
    background: var(--warn-bg);
    border-color: var(--warn);
    color: var(--warn);
  }
  .mini-btn {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11px;
    background: var(--paper);
    border: 1px solid var(--rule);
    color: var(--ink);
    padding: 6px 9px;
    cursor: pointer;
  }
  .mini-btn:hover { background: var(--row-hover); }
  .mini-btn.active {
    background: var(--accent-bg);
    border-color: var(--accent);
    color: var(--accent);
    font-weight: 700;
  }
  .replay-clock {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12.5px;
    color: var(--ink);
  }
  .replay-clock span { color: var(--muted); }
  .slider-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
  }
  .slider-end {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11px;
    color: var(--muted);
    min-width: 44px;
  }
  .slider-end.right { text-align: right; }
  input[type="range"]#game-slider {
    flex: 1;
    accent-color: var(--pos);
    cursor: pointer;
    height: 6px;
  }
  .chart-header {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
  }
  .chart-title {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
  }
  .chart-modes { display: flex; gap: 5px; flex-wrap: wrap; }
  #edge-chart {
    width: 100%;
    height: 118px;
    display: block;
    background: #12151a;
    border: 1px solid var(--rule);
    cursor: crosshair;
  }

  .section-kicker {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin: 14px 0 7px;
  }
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 10px;
    margin: 0 0 12px;
  }
  .stat-box {
    background: var(--paper);
    border: 1px solid var(--rule);
    padding: 12px 14px;
  }
  .stat-box.featured {
    border-top: 2px solid var(--rule-strong);
    background: var(--paper-hi);
  }
  .stat-label {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    margin-bottom: 5px;
  }
  .stat-val {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 22px;
    font-weight: 700;
    line-height: 1.15;
    color: var(--ink);
  }
  .stat-sub {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 12px;
    color: var(--muted);
    margin-top: 5px;
    line-height: 1.38;
  }
  .stat-sub strong { color: var(--ink); font-weight: 600; }
  .bands {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11.5px;
    color: var(--muted);
    background: var(--paper);
    border: 1px solid var(--rule);
    padding: 8px 12px;
    margin-bottom: 14px;
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    align-items: center;
  }
  .bands strong { color: var(--ink); font-weight: 700; }
  details.explainer {
    background: var(--paper);
    border: 1px solid var(--rule);
    padding: 10px 14px;
    margin-bottom: 16px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 12.5px;
    color: #c8cbd2;
  }
  details.explainer summary {
    cursor: pointer;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11.5px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 700;
    color: var(--ink);
  }
  .guide-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
    gap: 14px 26px;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid var(--rule);
  }
  .guide-col h3 {
    margin: 0 0 6px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--accent);
  }
  .guide-list { margin: 0; padding: 0; list-style: none; }
  .guide-list li { margin-bottom: 5px; line-height: 1.42; }
  .guide-list code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11.5px;
    font-weight: 700;
    background: #252a33;
    color: #f3f1ea;
    padding: 1px 4px;
    margin-right: 4px;
  }
  .controls {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: flex-end;
    gap: 10px;
    border-bottom: 1px solid var(--rule-strong);
    padding-bottom: 10px;
    margin-bottom: 10px;
  }
  .tabs { display: flex; flex-wrap: wrap; gap: 5px; }
  .tab-btn {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
    background: var(--paper);
    border: 1px solid var(--rule);
    color: var(--muted);
    padding: 6px 11px;
    cursor: pointer;
  }
  .tab-btn:hover { background: var(--row-hover); color: var(--ink); }
  .tab-btn.active {
    background: var(--ink);
    border-color: var(--ink);
    color: #101215;
    font-weight: 700;
  }
  .filters { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
  .filters input[type="text"] {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
    padding: 5px 8px;
    border: 1px solid var(--rule-strong);
    background: var(--paper);
    color: var(--ink);
    width: 175px;
  }
  .filters input[type="text"]::placeholder { color: var(--muted); }
  .pill {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11px;
    background: var(--paper);
    border: 1px solid var(--rule);
    color: var(--ink);
    padding: 5px 8px;
    cursor: pointer;
  }
  .pill:hover { background: var(--row-hover); }
  .pill.active {
    background: var(--accent-bg);
    border-color: var(--accent);
    color: var(--accent);
    font-weight: 700;
  }
  .table-meta {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11.5px;
    color: var(--muted);
    margin-bottom: 6px;
  }
  .table-wrap {
    overflow-x: auto;
    background: var(--paper);
    border: 1px solid var(--rule);
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12.5px;
    font-variant-numeric: tabular-nums;
  }
  thead tr.super-head th {
    position: static;
    background: #14171c;
    border-bottom: 1px solid var(--rule);
    font-size: 10px;
    letter-spacing: 0.08em;
    color: var(--muted);
    text-align: center;
    cursor: default;
    padding: 5px 8px;
  }
  thead tr.super-head th:hover { background: #14171c; }
  thead th {
    position: sticky;
    top: 0;
    background: #1d2129;
    border-bottom: 2px solid var(--rule-strong);
    padding: 7px 8px;
    text-align: right;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--ink);
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
  }
  thead th.left { text-align: left; }
  thead th.group-sep, tbody td.group-sep { border-left: 2px solid var(--rule-strong); }
  thead th:hover { background: #262c36; }
  thead th .sort-arrow { color: var(--accent); font-size: 10px; margin-left: 3px; }
  tbody tr { border-bottom: 1px solid #22262e; }
  tbody tr:nth-child(even) { background: var(--row-alt); }
  tbody tr:hover { background: var(--row-hover); }
  tbody td { padding: 6px 8px; text-align: right; white-space: nowrap; }
  tbody td.left { text-align: left; }
  td.player-cell {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 13px;
    font-weight: 600;
    color: #f0eee6;
  }
  .tag {
    display: inline-block;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 10px;
    padding: 1px 4px;
    border: 1px solid var(--rule-strong);
    color: var(--muted);
    margin-left: 5px;
    vertical-align: 1px;
  }
  .tag-rook { border-color: var(--accent-bd); color: var(--accent); background: var(--accent-bg); }
  .tag-ret { border-color: rgba(217, 160, 91, 0.4); color: #e2b478; background: rgba(217, 160, 91, 0.12); }
  .tag-inj { border-color: var(--warn-bd); color: var(--warn); background: var(--warn-bg); font-weight: 700; }
  .pos { color: var(--pos); font-weight: 700; }
  .neg { color: var(--neg); font-weight: 700; }
  .muted { color: var(--muted); }
  .edge-pill { display: inline-block; padding: 1px 5px; min-width: 54px; text-align: right; }
  .edge-pos { background: var(--pos-bg); color: var(--pos); font-weight: 700; }
  .edge-neg { background: var(--neg-bg); color: var(--neg); font-weight: 700; }
  .edge-flat { color: var(--muted); }
  .call-badge {
    display: inline-block;
    font-size: 10.5px;
    padding: 1px 6px;
    border: 1px solid transparent;
    font-weight: 700;
    min-width: 56px;
    text-align: center;
  }
  .call-hit { color: var(--pos); background: var(--pos-bg); border-color: var(--pos-bd); }
  .call-miss { color: var(--neg); background: var(--neg-bg); border-color: var(--neg-bd); }
  .call-agree { color: var(--muted); background: #1e222a; border-color: var(--rule); font-weight: 600; }
  footer {
    margin-top: 20px;
    padding-top: 10px;
    border-top: 1px solid var(--rule);
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11px;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="kicker">Fantasy Basketball Draft Engine &bull; Strict Out-of-Sample Multi-Season Audit</div>
    <h1 id="page-title">2026&ndash;27 Projected Top 150 &amp; Real-Time Edge Replay</h1>
    <p class="subtitle" id="page-subtitle"></p>
  </header>

  <div class="season-bar" id="season-bar"></div>

  <section class="replay-panel">
    <div class="replay-top">
      <div class="replay-controls">
        <button class="play-btn" id="play-btn">&#9654; Play (4 g/s)</button>
        <button class="mini-btn" id="reset-btn">&#8634; Game 1</button>
        <button class="mini-btn" id="end-btn">End</button>
        <button class="mini-btn active" data-speed="250">4 g/s</button>
        <button class="mini-btn" data-speed="125">8 g/s</button>
      </div>
      <div class="replay-clock" id="replay-clock"></div>
    </div>
    <div class="slider-row">
      <span class="slider-end">G 1</span>
      <input type="range" id="game-slider" min="1" max="82" value="82">
      <span class="slider-end right" id="slider-max-lbl">G 82</span>
    </div>
    <div class="chart-header">
      <div class="chart-title" id="chart-title">Season Trajectory (Click chart to jump to any game)</div>
      <div class="chart-modes">
        <button class="mini-btn active" data-chart="t15_fpts">Top 15 Targets vs Fades (FPTS)</button>
        <button class="mini-btn" data-chart="top150_fpts">Top 150 Model vs ADP (FPTS)</button>
        <button class="mini-btn" data-chart="corr_r">Top 150 Edge Corr (r)</button>
      </div>
    </div>
    <svg id="edge-chart" viewBox="0 0 1000 118" preserveAspectRatio="none"></svg>
  </section>

  <div class="section-kicker" id="hero-kicker">Total Edge &amp; Fantasy Points Gained</div>
  <section class="stats-grid" id="hero-cards"></section>

  <div class="section-kicker">Ranking Quality &amp; Sleeper Diagnostics</div>
  <section class="stats-grid" id="detail-cards"></section>

  <div class="bands" id="band-strip"></div>

  <details class="explainer">
    <summary>Column Guide &amp; How to Read This Board (Click to Expand)</summary>
    <div class="guide-grid">
      <div class="guide-col">
        <h3>Pre-Season Projection Columns</h3>
        <ul class="guide-list">
          <li><code>Rk</code> Model's projected draft rank (#1&ndash;#150), sorted by <strong>Model Proj</strong>.</li>
          <li><code>ADP Rk</code> Consensus ADP rank position (1st, 2nd, ..., 150th player in ADP order).</li>
          <li><code>Avg ADP</code> Raw average pick number across fantasy platforms (e.g., <code>14.5</code>).</li>
          <li><code>&Delta;Rk</code> <strong>ADP Rk minus Model Rk.</strong> Positive (green) = model ranked player higher than ADP; negative = lower.</li>
          <li><code>ADP Proj</code> Expected <strong>Availability-Adjusted FPPG (<code>AdjFPPG</code>)</strong> implied by that draft slot from the prior season's curve (65-game scale).</li>
          <li><code>Edge</code> Model's predicted edge <strong>relative to the average Top-150 drafted peer</strong> (in <code>AdjFPPG</code>). Between <code>-0.25</code> and <code>+0.25</code> = agrees with ADP.</li>
          <li><code>Model Proj</code> Final projected <code>AdjFPPG</code> (strictly out-of-sample — trained only on earlier seasons).</li>
        </ul>
      </div>
      <div class="guide-col">
        <h3>Game-by-Game Actuals at Current Slider Point</h3>
        <ul class="guide-list">
          <li><code>GP</code> Games played through the selected team-game checkpoint (<code>0</code> in red = missed all games to this point).</li>
          <li><code>FPTS</code> Exact cumulative fantasy points scored through this game (ESPN default points scoring).</li>
          <li><code>FPPG</code> Actual raw fantasy points per game in games played.</li>
          <li><code>Act Adj</code> Actual <strong>Availability-Adjusted FPPG</strong> at this point in the season (missed games filled at the live checkpoint waiver rate).</li>
          <li><code>Act Rk</code> Actual rank across the entire NBA at this point in the season by <code>Act Adj</code>.</li>
          <li><code>Act vs ADP</code> Player's <code>Act Adj</code> minus what their <strong>ADP slot</strong> is producing at this exact checkpoint.</li>
          <li><code>Call</code> Grades meaningful disagreements (<code>|Edge| &ge; 0.25</code> or <code>|&Delta;Rk| &ge; 3</code>): <strong>TGT &check;</strong> / <strong>FADE &check;</strong> / <strong>TGT &cross;</strong> / <strong>FADE &cross;</strong> / <strong>AGREE</strong>.</li>
        </ul>
      </div>
    </div>
  </details>

  <div class="controls">
    <div class="tabs" id="tabs">
      <button class="tab-btn active" data-tab="stage2_top150">Stage 2: Top 150 (ADP + Edge)</button>
      <button class="tab-btn" data-tab="stage1_top150">Stage 1: Pure Stats Top 150 (0% ADP)</button>
      <button class="tab-btn" data-tab="deep_sleepers">ADP 151&ndash;225 Sleepers</button>
    </div>
    <div class="filters">
      <input type="text" id="search" placeholder="Filter player or pos...">
      <button class="pill active" data-filter="all">All</button>
      <button class="pill" data-filter="top15">Top 15</button>
      <button class="pill" data-filter="pos">Targets (&ge; +0.25)</button>
      <button class="pill" data-filter="neg">Fades (&le; &minus;0.25)</button>
      <button class="pill" data-filter="gt150">ADP &gt; 150</button>
      <button class="pill" data-filter="rook">Rookies</button>
    </div>
  </div>

  <div class="table-meta">
    <span id="row-count"></span>
    <span id="filtered-summary"></span>
  </div>

  <div class="table-wrap">
    <table>
      <thead>
        <tr class="super-head">
          <th colspan="9">Pre-Season Model Projection (Zero Leakage)</th>
          <th colspan="7" class="group-sep" id="actuals-super-head">Season Actuals Through Selected Game</th>
        </tr>
        <tr>
          <th data-col="rk">Rk<span class="sort-arrow"></span></th>
          <th class="left" data-col="p">Player<span class="sort-arrow"></span></th>
          <th class="left" data-col="pos">Pos<span class="sort-arrow"></span></th>
          <th data-col="mrk">ADP Rk<span class="sort-arrow"></span></th>
          <th data-col="adp">Avg ADP<span class="sort-arrow"></span></th>
          <th data-col="drk">&Delta;Rk<span class="sort-arrow"></span></th>
          <th data-col="mp">ADP Proj<span class="sort-arrow"></span></th>
          <th data-col="e">Edge<span class="sort-arrow"></span></th>
          <th data-col="pr">Model Proj<span class="sort-arrow"></span></th>
          <th class="group-sep" data-col="gp">GP<span class="sort-arrow"></span></th>
          <th data-col="fpts">FPTS<span class="sort-arrow"></span></th>
          <th data-col="act_fppg">FPPG<span class="sort-arrow"></span></th>
          <th data-col="act_adj">Act Adj<span class="sort-arrow"></span></th>
          <th data-col="act_rank">Act Rk<span class="sort-arrow"></span></th>
          <th data-col="act_vs_adp">Act vs ADP<span class="sort-arrow"></span></th>
          <th data-col="call_score">Call<span class="sort-arrow"></span></th>
        </tr>
      </thead>
      <tbody id="board-body"></tbody>
    </table>
  </div>

  <footer>
    <span>Generated by build_board_page.py from draft_engine.py (ESPN Default Scoring)</span>
    <span>8-Season Backtest (1,152 drafts/strategy): +1,725.8 FPTS greedy / +2,000.9 FPTS reach-vs-wait (8/8 winning seasons)</span>
  </footer>
</div>

<script>
const APP = __PAYLOAD_JSON__;
const ANCHORS_RK = APP.anchor_ranks;

const state = {
  season: APP.default_season,
  step: 0, // 0-indexed (0 = Game 1)
  tab: "stage2_top150",
  filter: "all",
  search: "",
  sortCol: "rk",
  sortDir: 1,
  chartMode: "t15_fpts",
  speedMs: 250,
  timer: null,
};

// Pre-build cumulative lookup tables per season on first access for instant 60fps scrubbing
const seasonCache = {};

function getSeasonCum(seasonKey) {
  if (seasonCache[seasonKey]) return seasonCache[seasonKey];
  const S = APP.seasons[seasonKey];
  const nSteps = S.max_step;
  const cumGp = {};
  const cumFpts = {};
  for (const [aid, sparse] of Object.entries(S.logs)) {
    const gpArr = new Int16Array(nSteps);
    const fptsArr = new Float32Array(nSteps);
    let g = 0, f = 0, ptr = 0;
    for (let s = 0; s < nSteps; s++) {
      while (ptr < sparse.length && sparse[ptr][0] === s) {
        g += sparse[ptr][1];
        f += sparse[ptr][2];
        ptr++;
      }
      gpArr[s] = g;
      fptsArr[s] = f;
    }
    cumGp[aid] = gpArr;
    cumFpts[aid] = fptsArr;
  }
  const obj = { cumGp, cumFpts };
  seasonCache[seasonKey] = obj;
  return obj;
}

function interpSlotAdj(anchors, adp) {
  if (adp === null || adp === undefined) return anchors[anchors.length - 1];
  if (adp <= ANCHORS_RK[0]) return anchors[0];
  const last = ANCHORS_RK.length - 1;
  if (adp >= ANCHORS_RK[last]) return anchors[last];
  for (let i = 0; i < last; i++) {
    const r0 = ANCHORS_RK[i], r1 = ANCHORS_RK[i + 1];
    if (adp >= r0 && adp <= r1) {
      const t = (adp - r0) / (r1 - r0);
      return anchors[i] + t * (anchors[i + 1] - anchors[i]);
    }
  }
  return anchors[last];
}

function estimateRankFromAnchors(anchors, adjVal) {
  if (adjVal >= anchors[0]) return 1;
  const last = anchors.length - 1;
  if (adjVal <= anchors[last]) return ANCHORS_RK[last] + 1;
  for (let i = 0; i < last; i++) {
    const v0 = anchors[i], v1 = anchors[i + 1];
    if (adjVal <= v0 && adjVal >= v1) {
      const span = Math.max(v0 - v1, 1e-6);
      const t = (v0 - adjVal) / span;
      return Math.max(1, Math.round(ANCHORS_RK[i] + t * (ANCHORS_RK[i + 1] - ANCHORS_RK[i])));
    }
  }
  return ANCHORS_RK[last] + 1;
}

function fmtSigned(v, dec = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return "&mdash;";
  return (v >= 0 ? "+" : "") + Number(v).toFixed(dec);
}
function fmtNum(v, dec = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return "&mdash;";
  return Number(v).toFixed(dec);
}
function fmtInt(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "&mdash;";
  return Math.round(Number(v)).toLocaleString("en-US");
}
function fmtSignedInt(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "&mdash;";
  const n = Math.round(Number(v));
  return (n >= 0 ? "+" : "") + n.toLocaleString("en-US");
}

function pearson(xs, ys) {
  const n = xs.length;
  if (n < 3) return 0;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += xs[i]; my += ys[i]; }
  mx /= n; my /= n;
  let num = 0, dx2 = 0, dy2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx, dy = ys[i] - my;
    num += dx * dy; dx2 += dx * dx; dy2 += dy * dy;
  }
  const den = Math.sqrt(dx2 * dy2);
  return den > 1e-9 ? num / den : 0;
}

function enrichRow(r, cum, cp) {
  const gpArr = cum.cumGp[r.id];
  const fptsArr = cum.cumFpts[r.id];
  const gp = gpArr ? gpArr[state.step] : 0;
  const fpts = fptsArr ? fptsArr[state.step] : 0;
  const raw = gp > 0 ? fpts / gp : 0;
  const effG = Math.min(gp, cp.ref_g);
  const actAdj = gp > 0 ? ((effG * raw + (cp.ref_g - effG) * cp.waiver) / cp.ref_g) : cp.waiver;
  const actRank = gp > 0 ? estimateRankFromAnchors(cp.anchors, actAdj) : (ANCHORS_RK[ANCHORS_RK.length - 1] + 1);
  const slotAdj = r.adp !== null ? interpSlotAdj(cp.anchors, r.adp) : cp.waiver;
  const actVsAdp = actAdj - slotAdj;
  return {
    ...r,
    gp,
    fpts,
    act_fppg: raw,
    act_adj: actAdj,
    act_rank: actRank,
    act_vs_adp: actVsAdp,
    in_t150: actAdj >= cp.t150_cut - 1e-6 && gp > 0,
  };
}

function evalCall(r) {
  if (r.e === null || r.e === undefined) return { label: "&mdash;", cls: "call-agree", score: 0 };
  const bigEdge = Math.abs(r.e) >= 0.25;
  const bigShift = r.drk !== null && Math.abs(r.drk) >= 3;
  if (!bigEdge && !bigShift) return { label: "AGREE", cls: "call-agree", score: 0 };
  const liked = r.e > 0;
  const hit = liked ? (r.act_vs_adp > 0) : (r.act_vs_adp < 0);
  if (liked) return hit ? { label: "TGT ✓", cls: "call-hit", score: 1 } : { label: "TGT ✗", cls: "call-miss", score: -1 };
  return hit ? { label: "FADE ✓", cls: "call-hit", score: 1 } : { label: "FADE ✗", cls: "call-miss", score: -1 };
}

function renderSeasonBar() {
  const cv = APP.season_order.filter(x => x.split === "cv");
  const ho = APP.season_order.filter(x => x.split === "holdout");
  const lv = APP.season_order.filter(x => x.split === "live");
  const S = APP.seasons[state.season];

  const mkBtns = (arr) => arr.map(x => `
    <button class="season-btn ${x.split} ${x.season === state.season ? "active" : ""}" data-season="${x.season}">${x.season}</button>
  `).join("");

  let simHtml = "";
  if (S.sim_wait && S.sim_greedy) {
    simHtml = `12-Team Sim (144 drafts): Greedy <strong>${fmtSignedInt(S.sim_greedy.diff)}</strong> | Wait <strong>${fmtSignedInt(S.sim_wait.diff)}</strong> FPTS (Win ${S.sim_wait.win_pct.toFixed(1)}%)`;
  } else {
    simHtml = `Live Out-of-Sample Season (${S.max_step} team-games played so far)`;
  }

  document.getElementById("season-bar").innerHTML = `
    <div class="season-group"><span class="season-group-lbl">CV Folds:</span>${mkBtns(cv)}</div>
    <div class="season-group"><span class="season-group-lbl">Holdout:</span>${mkBtns(ho)}</div>
    <div class="season-group"><span class="season-group-lbl">Live:</span>${mkBtns(lv)}</div>
    <div class="season-sim-note">${simHtml}</div>
  `;

  document.querySelectorAll(".season-btn").forEach(btn => {
    btn.addEventListener("click", () => selectSeason(btn.dataset.season));
  });
}

function renderChart() {
  const S = APP.seasons[state.season];
  const arr = S.ts[state.chartMode];
  const n = arr.length;
  const svg = document.getElementById("edge-chart");
  const W = 1000, H = 118, padL = 56, padR = 16, padT = 12, padB = 20;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  let minV = Math.min(0, ...arr);
  let maxV = Math.max(0, ...arr);
  if (maxV - minV < 1e-4) { maxV += 1; minV -= 1; }
  const margin = (maxV - minV) * 0.12;
  minV -= margin; maxV += margin;

  const xOf = (i) => padL + (n > 1 ? (i / (n - 1)) * plotW : plotW / 2);
  const yOf = (v) => padT + ((maxV - v) / (maxV - minV)) * plotH;
  const y0 = yOf(0);

  // Full background dim trace + active trace up to state.step
  const fullPts = arr.map((v, i) => `${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`).join(" ");
  const curPts = arr.slice(0, state.step + 1).map((v, i) => `${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`).join(" ");
  const curVal = arr[state.step];
  const curX = xOf(state.step);
  const curY = yOf(curVal);
  const strokeCol = curVal >= 0 ? "#4ade80" : "#f87171";

  const fmtTick = (v) => state.chartMode === "corr_r" ? (v >= 0 ? "+" : "") + v.toFixed(2) : fmtSignedInt(v);

  svg.innerHTML = `
    <line x1="${padL}" y1="${padT}" x2="${W - padR}" y2="${padT}" stroke="#232730" stroke-width="1"/>
    <line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" stroke="#232730" stroke-width="1"/>
    <line x1="${padL}" y1="${y0.toFixed(1)}" x2="${W - padR}" y2="${y0.toFixed(1)}" stroke="#464d5b" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="${padL - 6}" y="${padT + 9}" text-anchor="end" fill="#8b919c" font-family="ui-monospace,monospace" font-size="10">${fmtTick(maxV)}</text>
    <text x="${padL - 6}" y="${(y0 + 3).toFixed(1)}" text-anchor="end" fill="#8b919c" font-family="ui-monospace,monospace" font-size="10">0</text>
    <text x="${padL - 6}" y="${H - padB}" text-anchor="end" fill="#8b919c" font-family="ui-monospace,monospace" font-size="10">${fmtTick(minV)}</text>
    <polyline fill="none" stroke="#363c48" stroke-width="1.5" points="${fullPts}"/>
    <polyline fill="none" stroke="${strokeCol}" stroke-width="2.4" points="${curPts}"/>
    <line x1="${curX.toFixed(1)}" y1="${padT}" x2="${curX.toFixed(1)}" y2="${H - padB}" stroke="#e7e5dd" stroke-width="1" stroke-dasharray="2,2" opacity="0.65"/>
    <circle cx="${curX.toFixed(1)}" cy="${curY.toFixed(1)}" r="4.2" fill="${strokeCol}" stroke="#101215" stroke-width="1.5"/>
    <text x="${Math.min(W - padR - 4, Math.max(padL + 6, curX + 8)).toFixed(1)}" y="${Math.max(padT + 11, curY - 7).toFixed(1)}" fill="${strokeCol}" font-family="ui-monospace,monospace" font-size="11" font-weight="700">${fmtTick(curVal)}</text>
    <text x="${padL}" y="${H - 5}" fill="#8b919c" font-family="ui-monospace,monospace" font-size="10">Game 1</text>
    <text x="${W - padR}" y="${H - 5}" text-anchor="end" fill="#8b919c" font-family="ui-monospace,monospace" font-size="10">Game ${n}</text>
  `;
}

function renderCards(elId, cards) {
  document.getElementById(elId).innerHTML = cards.map(c => `
    <div class="stat-box ${c.featured ? "featured" : ""}">
      <div class="stat-label">${c.label}</div>
      <div class="stat-val ${c.cls || ""}">${c.val}</div>
      <div class="stat-sub">${c.sub}</div>
    </div>
  `).join("");
}

function renderAll() {
  const S = APP.seasons[state.season];
  const cum = getSeasonCum(state.season);
  const cp = S.checkpoints[state.step];
  const splitTag = S.split.toUpperCase();

  document.getElementById("page-title").innerHTML = `${S.season} Projected Top 150 &amp; Real-Time Edge Replay <span class="tag ${S.split === "live" ? "tag-rook" : (S.split === "holdout" ? "tag-inj" : "")}">${splitTag}</span>`;
  document.getElementById("page-subtitle").innerHTML =
    `Strict out-of-sample pre-season board (${S.split === "holdout" ? "trained strictly on 2016–17..2023–24" : "trained strictly on seasons before " + S.season}), ` +
    `evaluated at <strong>Game ${cp.g} / ${S.max_step}</strong> (<strong>${cp.date}</strong> &bull; ${cp.reg} NBA games &bull; ${cp.pct}% of season).`;

  document.getElementById("replay-clock").innerHTML =
    `<strong>Game ${cp.g} / ${S.max_step}</strong> <span>&bull; ${cp.date} &bull; ${cp.reg} NBA games (${cp.pct}%) &bull; Waiver ${cp.waiver.toFixed(2)}</span>`;
  document.getElementById("slider-max-lbl").textContent = `G ${S.max_step}`;
  document.getElementById("actuals-super-head").textContent = `Season Actuals Through Game ${cp.g} (${cp.date} • ${cp.reg} NBA Games)`;
  document.getElementById("hero-kicker").textContent = `Total Edge & Fantasy Points Gained — ${S.season} Through Game ${cp.g} (${cp.date})`;

  // Enrich Stage 2 (all 225) & Stage 1 (top 150) at current checkpoint
  const s2All = S.s2_all.map(r => enrichRow(r, cum, cp));
  const s2Top150 = s2All.slice(0, 150);
  const mktTop150 = s2All.filter(r => r.mrk !== null && r.mrk <= 150).sort((a, b) => a.mrk - b.mrk);
  const deepPool = s2All.filter(r => r.mrk !== null && r.mrk > 150);
  const s1Top150 = S.s1_150.map(r => enrichRow(r, cum, cp));

  // Hero Box 1: Top 150 Model vs ADP FPTS
  const s2_150_fpts = s2Top150.reduce((a, r) => a + r.fpts, 0);
  const mkt_150_fpts = mktTop150.reduce((a, r) => a + r.fpts, 0);
  const s2_150_diff = s2_150_fpts - mkt_150_fpts;
  const s2_150_adj_diff = s2Top150.reduce((a, r) => a + r.act_adj, 0) - mktTop150.reduce((a, r) => a + r.act_adj, 0);
  const s2_swaps = s2Top150.filter(r => r.mrk > 150).length;

  // Hero Box 2: Top 150 Targets (>= +0.25) vs Fades (<= -0.25)
  const tAll = mktTop150.filter(r => r.e >= 0.25);
  const fAll = mktTop150.filter(r => r.e <= -0.25);
  const tAvgVs = tAll.length ? tAll.reduce((a, r) => a + r.act_vs_adp, 0) / tAll.length : 0;
  const fAvgVs = fAll.length ? fAll.reduce((a, r) => a + r.act_vs_adp, 0) / fAll.length : 0;
  const tAvgFpts = tAll.length ? tAll.reduce((a, r) => a + r.fpts, 0) / tAll.length : 0;
  const fAvgFpts = fAll.length ? fAll.reduce((a, r) => a + r.fpts, 0) / fAll.length : 0;

  // Hero Box 3: Top 15 Highest-Conviction Targets vs Top 15 Biggest Fades
  const byE = [...mktTop150].sort((a, b) => b.e - a.e);
  const t15 = byE.slice(0, 15);
  const f15 = byE.slice(-15);
  const t15Fpts = t15.reduce((a, r) => a + r.fpts, 0);
  const f15Fpts = f15.reduce((a, r) => a + r.fpts, 0);
  const t15Diff = t15Fpts - f15Fpts;
  const t15VsAvg = t15.reduce((a, r) => a + r.act_vs_adp, 0) / 15;
  const f15VsAvg = f15.reduce((a, r) => a + r.act_vs_adp, 0) / 15;
  const t15VsSum = (t15VsAvg - f15VsAvg) * 15;

  // Hero Box 4: Top 15 Overall Board (Picks #1-#15)
  const s2_15_fpts = s2Top150.slice(0, 15).reduce((a, r) => a + r.fpts, 0);
  const mkt_15_fpts = mktTop150.slice(0, 15).reduce((a, r) => a + r.fpts, 0);
  const s1_15_fpts = s1Top150.slice(0, 15).reduce((a, r) => a + r.fpts, 0);

  renderCards("hero-cards", [
    {
      featured: true,
      label: "Top 150 — Model vs. ADP Total FPTS",
      val: `${fmtSignedInt(s2_150_diff)} FPTS`,
      cls: s2_150_diff >= 0 ? "pos" : "neg",
      sub: `Model Top 150: <strong>${fmtInt(s2_150_fpts)}</strong> vs ADP: <strong>${fmtInt(mkt_150_fpts)}</strong><br>Across ${s2_swaps} board swaps (${fmtSigned(s2_150_adj_diff)} sum AdjFPPG)`,
    },
    {
      featured: true,
      label: "Top 150 — Targets vs. Fades Edge",
      val: `${fmtSigned(tAvgVs - fAvgVs, 2)} AdjFPPG`,
      cls: (tAvgVs - fAvgVs) >= 0 ? "pos" : "neg",
      sub: `<strong>${tAll.length} Targets</strong> (&ge;+0.25): ${fmtSigned(tAvgVs)} vs ADP (${fmtInt(tAvgFpts)} FPTS/p)<br><strong>${fAll.length} Fades</strong> (&le;&minus;0.25): ${fmtSigned(fAvgVs)} vs ADP (${fmtInt(fAvgFpts)} FPTS/p) &bull; <strong>${fmtSignedInt(tAvgFpts - fAvgFpts)}</strong>/p`,
    },
    {
      featured: true,
      label: "Top 15 Targets vs. Top 15 Fades",
      val: `${fmtSignedInt(t15Diff)} FPTS`,
      cls: t15Diff >= 0 ? "pos" : "neg",
      sub: `Top 15 Targets: <strong>${fmtInt(t15Fpts)}</strong> (${fmtSigned(t15VsAvg)} vs ADP) &bull; Fades: <strong>${fmtInt(f15Fpts)}</strong> (${fmtSigned(f15VsAvg)})<br>Total AdjFPPG edge spread: <strong class="${t15VsSum >= 0 ? "pos" : "neg"}">${fmtSigned(t15VsSum)}</strong>`,
    },
    {
      featured: true,
      label: "Our Top 15 Overall (Picks #1–#15)",
      val: `${fmtInt(s2_15_fpts)} FPTS`,
      cls: (s2_15_fpts - mkt_15_fpts) >= 0 ? "pos" : "",
      sub: `Stage 2 vs ADP Top 15: <strong>${fmtSignedInt(s2_15_fpts - mkt_15_fpts)} FPTS</strong><br>Stage 1 (0% ADP) Top 15: <strong class="${(s1_15_fpts - mkt_15_fpts) >= 0 ? "pos" : "neg"}">${fmtSignedInt(s1_15_fpts - mkt_15_fpts)} FPTS</strong> (${fmtInt(s1_15_fpts)})`,
    },
  ]);

  // Diagnostics
  const corrAll = pearson(mktTop150.map(r => r.re), mktTop150.map(r => r.act_vs_adp));
  let flipsTotal = 0, flipsWon = 0;
  for (let i = 0; i < mktTop150.length; i++) {
    for (let j = i + 1; j < mktTop150.length; j++) {
      const a = mktTop150[i], b = mktTop150[j];
      if (b.pr > a.pr) {
        flipsTotal++;
        if (b.act_adj > a.act_adj) flipsWon++;
      }
    }
  }
  const flipsPct = flipsTotal ? (100 * flipsWon / flipsTotal) : 50;

  const posDeep = deepPool.filter(r => r.e > 0);
  const negDeep = deepPool.filter(r => r.e <= 0);
  const posDeepHit = posDeep.filter(r => r.in_t150).length;
  const negDeepHit = negDeep.filter(r => r.in_t150).length;
  const s1Cap = s1Top150.filter(r => r.in_t150).length;
  const mktCap = mktTop150.filter(r => r.in_t150).length;
  const s2Cap = s2Top150.filter(r => r.in_t150).length;
  const s1Gt = s1Top150.filter(r => r.mrk === null || r.mrk > 150);
  const s1GtHit = s1Gt.filter(r => r.in_t150).length;

  renderCards("detail-cards", [
    {
      label: "Top-150 Edge Correlation (r)",
      val: `${corrAll >= 0 ? "+" : ""}${corrAll.toFixed(3)}`,
      cls: corrAll >= 0 ? "pos" : "neg",
      sub: `Stage 2 True Top-150 Capture: <strong>${s2Cap}/150</strong> vs Market ${mktCap}/150`,
    },
    {
      label: "Head-to-Head Rank Flips vs. ADP",
      val: `${flipsPct.toFixed(1)}%`,
      cls: flipsPct >= 50 ? "pos" : "neg",
      sub: `<strong>${flipsWon} / ${flipsTotal}</strong> player-vs-player swaps won at Game ${cp.g}`,
    },
    {
      label: "ADP 151–225 Deep Sleepers",
      val: `${posDeepHit}/${posDeep.length} (${posDeep.length ? (100 * posDeepHit / posDeep.length).toFixed(1) : "0.0"}%)`,
      cls: "pos",
      sub: `Pos-edge Top-150 hit rate vs ${negDeepHit}/${negDeep.length} (${negDeep.length ? (100 * negDeepHit / negDeep.length).toFixed(1) : "0.0"}%) for neg-edge`,
    },
    {
      label: "Stage 1 Full-NBA Top 150 (0% ADP)",
      val: `${s1Cap}/150`,
      cls: s1Cap >= mktCap ? "pos" : "",
      sub: `vs Market ${mktCap}/150 &bull; ${s1GtHit}/${s1Gt.length} ADP&gt;150 hits`,
    },
  ]);

  // Round bands strip
  const bandDefs = [["R1–2 (1–24)", 1, 24], ["R3–5 (25–60)", 25, 60], ["R6–9 (61–108)", 61, 108], ["R10–13 (109–150)", 109, 150]];
  document.getElementById("band-strip").innerHTML =
    `<strong>${S.season} Edge Corr by Round (Game ${cp.g}):</strong>` +
    bandDefs.map(([lbl, lo, hi]) => {
      const sub = mktTop150.filter(r => r.mrk >= lo && r.mrk <= hi);
      const rc = pearson(sub.map(r => r.re), sub.map(r => r.act_vs_adp));
      return `<span>${lbl}: <strong class="${rc >= 0 ? "pos" : "neg"}">${rc >= 0 ? "+" : ""}${rc.toFixed(3)}</strong></span>`;
    }).join("");

  renderChart();

  // Render active tab table
  const baseRows = state.tab === "stage2_top150" ? s2Top150 : (state.tab === "stage1_top150" ? s1Top150 : deepPool);
  const q = state.search.trim().toLowerCase();
  const filtered = baseRows.filter(r => {
    if (q && !(r.p.toLowerCase().includes(q) || r.pos.toLowerCase().includes(q))) return false;
    if (state.filter === "top15" && !(r.rk <= 15)) return false;
    if (state.filter === "pos" && !(r.e >= 0.25)) return false;
    if (state.filter === "neg" && !(r.e <= -0.25)) return false;
    if (state.filter === "gt150" && !(r.mrk === null || r.mrk > 150)) return false;
    if (state.filter === "rook" && r.t !== "ROOK") return false;
    return true;
  });

  const col = state.sortCol, dir = state.sortDir;
  filtered.sort((a, b) => {
    const va = col === "call_score" ? evalCall(a).score : a[col];
    const vb = col === "call_score" ? evalCall(b).score : b[col];
    if (va === null || va === undefined) return 1;
    if (vb === null || vb === undefined) return -1;
    if (typeof va === "string") return dir * va.localeCompare(vb);
    return dir * (va - vb);
  });

  document.querySelectorAll("thead th").forEach(th => {
    const arrow = th.querySelector(".sort-arrow");
    if (!arrow) return;
    arrow.textContent = th.dataset.col === state.sortCol ? (state.sortDir === 1 ? "▲" : "▼") : "";
  });

  const hits = filtered.filter(r => evalCall(r).score === 1).length;
  const calls = filtered.filter(r => evalCall(r).score !== 0).length;
  const top150Now = filtered.filter(r => r.in_t150).length;
  const totFpts = filtered.reduce((a, r) => a + r.fpts, 0);
  const avgVs = filtered.length ? filtered.reduce((a, r) => a + r.act_vs_adp, 0) / filtered.length : 0;

  document.getElementById("row-count").textContent =
    `Showing ${filtered.length} players | Total FPTS: ${fmtInt(totFpts)} | Actual Top-150: ${top150Now}/${filtered.length}`;
  document.getElementById("filtered-summary").textContent =
    `Avg Actual vs ADP: ${fmtSigned(avgVs)} AdjFPPG | Conviction Calls: ${hits}/${calls} (${calls ? (100 * hits / calls).toFixed(1) : "0.0"}%)`;

  document.getElementById("board-body").innerHTML = filtered.map(r => {
    const call = evalCall(r);
    const typeTag = r.t === "ROOK" ? '<span class="tag tag-rook">ROOK</span>' : (r.t === "RET" ? '<span class="tag tag-ret">RET</span>' : "");
    const injTag = r.al ? `<span class="tag tag-inj">${r.al}</span>` : "";
    const adpRkStr = r.mrk !== null ? `#${r.mrk}` : '<span class="muted">UNR</span>';
    const adpValStr = r.adp !== null ? fmtNum(r.adp, 1) : '<span class="muted">&mdash;</span>';
    const deltaCls = r.drk === null ? "muted" : (r.drk > 0 ? "pos" : (r.drk < 0 ? "neg" : "muted"));
    const deltaStr = r.drk === null ? "&mdash;" : (r.drk > 0 ? "+" + r.drk : String(r.drk));
    const edgeCls = r.e >= 0.25 ? "edge-pos" : (r.e <= -0.25 ? "edge-neg" : "edge-flat");
    const actVsCls = r.act_vs_adp > 0.25 ? "pos" : (r.act_vs_adp < -0.25 ? "neg" : "muted");
    const gpCls = r.gp === 0 ? "neg" : "";

    return `<tr>
      <td><strong>#${r.rk}</strong></td>
      <td class="left player-cell">${r.p}${typeTag}${injTag}</td>
      <td class="left muted">${r.pos}</td>
      <td class="muted">${adpRkStr}</td>
      <td class="muted">${adpValStr}</td>
      <td class="${deltaCls}">${deltaStr}</td>
      <td class="muted">${r.mp !== null ? fmtNum(r.mp, 2) : "&mdash;"}</td>
      <td><span class="edge-pill ${edgeCls}">${fmtSigned(r.e, 2)}</span></td>
      <td><strong>${fmtNum(r.pr, 2)}</strong></td>
      <td class="group-sep ${gpCls}">${r.gp}</td>
      <td>${fmtInt(r.fpts)}</td>
      <td>${fmtNum(r.act_fppg, 1)}</td>
      <td>${fmtNum(r.act_adj, 2)}</td>
      <td>#${r.act_rank}</td>
      <td class="${actVsCls}">${fmtSigned(r.act_vs_adp, 2)}</td>
      <td><span class="call-badge ${call.cls}">${call.label}</span></td>
    </tr>`;
  }).join("");
}

function stopPlayback() {
  if (state.timer !== null) {
    clearInterval(state.timer);
    state.timer = null;
  }
  const btn = document.getElementById("play-btn");
  btn.classList.remove("playing");
  const gps = Math.round(1000 / state.speedMs);
  btn.innerHTML = `&#9654; Play (${gps} g/s)`;
}

function startPlayback() {
  const S = APP.seasons[state.season];
  if (state.step >= S.max_step - 1) {
    state.step = 0;
    document.getElementById("game-slider").value = 1;
    renderAll();
  }
  const btn = document.getElementById("play-btn");
  btn.classList.add("playing");
  btn.innerHTML = "&#10074;&#10074; Pause";
  state.timer = setInterval(() => {
    const curS = APP.seasons[state.season];
    if (state.step >= curS.max_step - 1) {
      stopPlayback();
      return;
    }
    state.step++;
    document.getElementById("game-slider").value = state.step + 1;
    renderAll();
  }, state.speedMs);
}

function selectSeason(seasonKey) {
  stopPlayback();
  state.season = seasonKey;
  const S = APP.seasons[seasonKey];
  state.step = S.max_step - 1;
  const slider = document.getElementById("game-slider");
  slider.min = 1;
  slider.max = S.max_step;
  slider.value = S.max_step;
  renderSeasonBar();
  renderAll();
}

// Wire controls
document.getElementById("play-btn").addEventListener("click", () => {
  if (state.timer !== null) stopPlayback();
  else startPlayback();
});
document.getElementById("reset-btn").addEventListener("click", () => {
  stopPlayback();
  state.step = 0;
  document.getElementById("game-slider").value = 1;
  renderAll();
});
document.getElementById("end-btn").addEventListener("click", () => {
  stopPlayback();
  const S = APP.seasons[state.season];
  state.step = S.max_step - 1;
  document.getElementById("game-slider").value = S.max_step;
  renderAll();
});
document.getElementById("game-slider").addEventListener("input", e => {
  stopPlayback();
  state.step = Math.max(0, parseInt(e.target.value, 10) - 1);
  renderAll();
});
document.querySelectorAll("[data-speed]").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("[data-speed]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.speedMs = parseInt(btn.dataset.speed, 10);
    if (state.timer !== null) {
      stopPlayback();
      startPlayback();
    } else {
      stopPlayback();
    }
  });
});
document.querySelectorAll("[data-chart]").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("[data-chart]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.chartMode = btn.dataset.chart;
    renderChart();
  });
});
document.getElementById("edge-chart").addEventListener("click", e => {
  stopPlayback();
  const rect = e.currentTarget.getBoundingClientRect();
  const xFrac = Math.max(0, Math.min(1, (e.clientX - rect.left - rect.width * 0.056) / (rect.width * 0.928)));
  const S = APP.seasons[state.season];
  state.step = Math.round(xFrac * (S.max_step - 1));
  document.getElementById("game-slider").value = state.step + 1;
  renderAll();
});

document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.tab = btn.dataset.tab;
    state.sortCol = state.tab === "deep_sleepers" ? "e" : "rk";
    state.sortDir = state.tab === "deep_sleepers" ? -1 : 1;
    renderAll();
  });
});
document.querySelectorAll(".pill").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".pill").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.filter = btn.dataset.filter;
    renderAll();
  });
});
document.getElementById("search").addEventListener("input", e => {
  state.search = e.target.value;
  renderAll();
});
document.querySelectorAll("thead th").forEach(th => {
  th.addEventListener("click", () => {
    const col = th.dataset.col;
    if (!col) return;
    if (state.sortCol === col) {
      state.sortDir *= -1;
    } else {
      state.sortCol = col;
      state.sortDir = (col === "rk" || col === "mrk" || col === "adp" || col === "act_rank" || col === "p" || col === "pos") ? 1 : -1;
    }
    renderAll();
  });
});

selectSeason(APP.default_season);
</script>
</body>
</html>
"""


def main():
    data = Path("data")
    print("Building multi-season strict out-of-sample replay payload (6 CV + 2 Holdout + Live 2026-27)...")
    payload = build_all_seasons_payload(data, scoring_name="default")
    json_blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__PAYLOAD_JSON__", json_blob)
    out_path = Path("board_2026_27.html")
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path.resolve()} ({len(html):,} bytes, {len(payload['season_order'])} seasons)")


if __name__ == "__main__":
    main()
