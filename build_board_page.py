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

import argparse
import json
import math
import random
from collections import defaultdict
from datetime import date
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

    # Collect per-player box scores keyed by date_idx (or exact team-game number 1..min_team_games when is_live)
    raw_boxes = list(de.player_box_scores(folder, game_dates))
    if is_live and raw_boxes:
        team_gids = defaultdict(set)
        for gid, _aid, _name, _stats, _pos, _starter, tid in raw_boxes:
            if tid:
                team_gids[tid].add(gid)
        team_game_num = {}
        for tid, gids in team_gids.items():
            for gnum, gid in enumerate(sorted(gids, key=lambda x: (game_dates[x], x)), 1):
                team_game_num[(tid, gid)] = gnum
        min_tg = max(1, min(len(gids) for gids in team_gids.values()))
        max_step = min_tg
        step_events = [defaultdict(lambda: [0, 0.0]) for _ in range(max_step)]
        player_per_s = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
        step_dates = [""] * max_step
        for gid, aid, _name, stats, _pos, _starter, tid in raw_boxes:
            gnum = team_game_num.get((tid, gid))
            if gnum is None or gnum > min_tg:
                continue
            s_idx = gnum - 1
            pts = de.score_box_stats(stats, scoring)
            step_events[s_idx][aid][0] += 1
            step_events[s_idx][aid][1] += pts
            player_per_s[aid][s_idx][0] += 1
            player_per_s[aid][s_idx][1] += pts
            gd = game_dates[gid]
            if gd > step_dates[s_idx]:
                step_dates[s_idx] = gd
        player_step_sparse = {
            aid: [[s_idx, val[0], round(val[1], 1)] for s_idx, val in sorted(smap.items())]
            for aid, smap in player_per_s.items()
        }
        cum_gp = defaultdict(int)
        cum_pts = defaultdict(float)
        checkpoints = []
        step_adj_by_id = []
        for s_idx in range(max_step):
            for aid, (gp, pts) in step_events[s_idx].items():
                cum_gp[aid] += gp
                cum_pts[aid] += pts
            avg_g = float(s_idx + 1)
            reg_g = int(15 * (s_idx + 1))
            ref_g = max(1.0, de.REF_GAMES * min(avg_g, de.SEASON_GAMES) / de.SEASON_GAMES)
            season_pct = round(100.0 * avg_g / de.SEASON_GAMES, 1)
            min_qual = max(1, int(round(20.0 * (ref_g / de.REF_GAMES))))
            qual_raw = sorted((cum_pts[aid] / g for aid, g in cum_gp.items() if g >= min_qual), reverse=True)
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
                "date": step_dates[s_idx],
                "reg": reg_g,
                "avg_g": round(avg_g, 1),
                "pct": season_pct,
                "ref_g": round(ref_g, 2),
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

    player_daily = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))  # aid -> di -> [gp, pts]
    seen_game_dates = set()
    for gid, aid, _name, stats, _pos, _starter, _tid in raw_boxes:
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
    adp_entries, tr_pv, tr_pr, vegas_odds=None,
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
                prior_s=prior_s, vegas_odds=vegas_odds,
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


def build_live_2026_27_s2_cands(data, prior_s, prev2_s, births_prior, target_profs, prior_profs, played_before, adp_entries, early_data, vegas_odds=None):
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
                vegas_odds=vegas_odds,
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


def simulate_live_points_season(folder: Path, scoring: dict, s2_cands: list, n_teams: int = 12, n_rounds: int = 13, trials_per_slot: int = 60, adp_var: float = 2.0, max_centers: int = 4, seed: int = 42):
    """Run 720 Pure Greedy 12-team draft-and-hold leagues on the in-progress live season (total FPTS + weekly H2H)."""
    if len(s2_cands) < n_teams * n_rounds:
        return None
    game_dates = {}
    for g in de.read_csv(folder / "games.csv"):
        if not de.regular_season_game(g):
            continue
        gid = g.get("game_id")
        gd = (g.get("game_date") or "")[:10]
        hs = de._safe_float(g.get("home_score"))
        as_ = de._safe_float(g.get("away_score"))
        if not gid or not gd or hs is None or as_ is None or (hs == 0 and as_ == 0):
            continue
        game_dates[gid] = gd

    raw_boxes = list(de.player_box_scores(folder, game_dates))
    team_gids = defaultdict(set)
    for gid, _aid, _name, _stats, _pos, _starter, tid in raw_boxes:
        if tid:
            team_gids[tid].add(gid)
    if not team_gids:
        return None
    team_game_num = {}
    for tid, gids in team_gids.items():
        for gnum, gid in enumerate(sorted(gids, key=lambda x: (game_dates[x], x)), 1):
            team_game_num[(tid, gid)] = gnum
    min_tg = max(1, min(len(gids) for gids in team_gids.values()))

    raw_daily = defaultdict(lambda: defaultdict(float))  # aid -> gd -> fpts
    seen_dates = set()
    for gid, aid, _name, stats, _pos, _starter, tid in raw_boxes:
        if team_game_num.get((tid, gid), 999) > min_tg:
            continue
        gd = game_dates[gid]
        raw_daily[aid][gd] += de.score_box_stats(stats, scoring)
        seen_dates.add(gd)

    if not seen_dates:
        return None
    dates_sorted = sorted(seen_dates)
    date_to_idx = {d: i for i, d in enumerate(dates_sorted)}
    weeks_sorted = sorted({date.fromisoformat(d).isocalendar()[:2] for d in dates_sorted})
    wk_to_idx = {wk: i for i, wk in enumerate(weeks_sorted)}
    day_to_week = [wk_to_idx[date.fromisoformat(d).isocalendar()[:2]] for d in dates_sorted]
    n_days, n_wks = len(dates_sorted), len(weeks_sorted)

    players = sorted(s2_cands, key=lambda r: r["market_rank"])
    n_p = len(players)
    played_mat = np.zeros((n_p, n_days), dtype=np.uint8)
    fpts_mat = np.zeros((n_p, n_days), dtype=np.float64)
    player_tot = np.zeros(n_p, dtype=np.float64)
    player_wk = np.zeros((n_p, n_wks), dtype=np.float64)
    masks7, masks10, is_c, prio = [], [], [], []
    for idx, p in enumerate(players):
        ptup = p["positions"]
        m7, m10 = de._pos_to_masks(ptup)
        masks7.append(m7)
        masks10.append(m10)
        is_c.append(ptup[0] == "C")
        dmap = raw_daily.get(p["athlete_id"], {})
        gp = len(dmap)
        tot = 0.0
        for gd, pts in dmap.items():
            di = date_to_idx[gd]
            played_mat[idx, di] = 1
            fpts_mat[idx, di] = pts
            player_wk[idx, day_to_week[di]] += pts
            tot += pts
        player_tot[idx] = tot
        prio.append((tot / gp) if gp > 0 else -99.0)

    played_lists = played_mat.tolist()
    fpts_lists = fpts_mat.tolist()
    roster_cache = {}

    def score_roster(roster):
        key = tuple(sorted(roster))
        cached = roster_cache.get(key)
        if cached is not None:
            return cached
        ordered = sorted(roster, key=lambda idx: -prio[idx])
        counts = played_mat[ordered].sum(axis=0)
        crowded_days = np.flatnonzero(counts >= 5).tolist()
        tot = float(player_tot[ordered].sum())
        wk = player_wk[ordered].sum(axis=0).copy()
        for d in crowded_days:
            free_bits = 0x3FF
            match_slot = None
            filled = 0
            w = day_to_week[d]
            for idx in ordered:
                if not played_lists[idx][d]:
                    continue
                if filled == 10:
                    v = fpts_lists[idx][d]
                    tot -= v
                    wk[w] -= v
                    continue
                m = masks10[idx]
                if match_slot is None:
                    avail = m & free_bits
                    if avail:
                        free_bits ^= (avail & -avail)
                        filled += 1
                        continue
                    match_slot = [-1] * 10
                    filled = 0
                    for j in ordered:
                        if not played_lists[j][d]:
                            continue
                        if j == idx:
                            break
                        if de._dfs_match(j, match_slot, masks10, [0]):
                            filled += 1
                if filled < 10 and de._dfs_match(idx, match_slot, masks10, [0]):
                    filled += 1
                else:
                    v = fpts_lists[idx][d]
                    tot -= v
                    wk[w] -= v
        res = (tot, wk)
        roster_cache[key] = res
        return res

    straight_order = list(range(n_p))
    hero_scored = sorted(
        [(float(p["model_proj"]), -p["market_rank"], i) for i, p in enumerate(players)],
        key=lambda t: (-t[0], -t[1], t[2]),
    )
    hero_order = [i for _, _, i in hero_scored]
    pick_order = [
        (rnd + 1, team)
        for rnd in range(n_rounds)
        for team in (range(n_teams - 1, -1, -1) if rnd % 2 == 1 else range(n_teams))
    ]
    rng = random.Random(seed)
    hero_tots, field_avgs, diffs = [], [], []
    for slot in range(n_teams):
        for _ in range(trials_per_slot):
            opp_noisy = []
            for _t in range(n_teams):
                noisy = [(float(players[i]["adp"]) + rng.uniform(-adp_var, adp_var), i) for i in range(n_p)]
                noisy.sort(key=lambda t: (t[0], t[1]))
                opp_noisy.append([i for _, i in noisy])

            taken = [False] * n_p
            rosters = [[] for _ in range(n_teams)]
            c_cnt = [0] * n_teams
            m_sz = [0] * n_teams
            for rnd, team in pick_order:
                roster = rosters[team]
                rem_after = n_rounds - len(roster) - 1
                cur_c, cur_m = c_cnt[team], m_sz[team]
                board = hero_order if team == slot else (straight_order if rnd == 1 else opp_noisy[team])
                choice, new_m = None, cur_m
                for idx in board:
                    if taken[idx] or (is_c[idx] and cur_c >= max_centers):
                        continue
                    if cur_m + rem_after >= 7:
                        choice = idx
                        new_m = de.max_matching_7(roster + [idx], masks7) if cur_m < 7 else 7
                        break
                    cm = de.max_matching_7(roster + [idx], masks7)
                    if cm + rem_after >= 7:
                        choice, new_m = idx, cm
                        break
                if choice is None:
                    choice = next(
                        (idx for idx in board if not taken[idx] and not (is_c[idx] and cur_c >= max_centers)),
                        next(idx for idx in board if not taken[idx]),
                    )
                    new_m = de.max_matching_7(roster + [choice], masks7)
                taken[choice] = True
                roster.append(choice)
                if is_c[choice]:
                    c_cnt[team] += 1
                m_sz[team] = new_m

            tots = [score_roster(rosters[t])[0] for t in range(n_teams)]
            hero_tot = tots[slot]
            field_tots = [tots[t] for t in range(n_teams) if t != slot]
            field_avg = sum(field_tots) / len(field_tots)
            hero_tots.append(hero_tot)
            field_avgs.append(field_avg)
            diffs.append(hero_tot - field_avg)

    h_avg = mean(hero_tots)
    f_avg = mean(field_avgs)
    d_avg = mean(diffs)
    return {
        "drafts": len(diffs),
        "weeks": n_wks,
        "min_games": min_tg,
        "last_date": dates_sorted[-1],
        "hero_fpts": round(h_avg, 1),
        "field_fpts": round(f_avg, 1),
        "diff": round(d_avg, 1),
        "lift_pct": round(100.0 * d_avg / f_avg, 2) if f_avg else 0.0,
    }


def build_all_seasons_payload(data: Path, scoring_name: str = "default", min_train_season: str = "2016-17", cv_folds: int = 6, recency_halflife=None, sim_csv_override=None):
    _, scoring = de.resolve_scoring(scoring_name)
    seasons_dirs, season_data, births_cache, profiles_cache = de.load_all_completed_seasons(data, scoring)
    pure_vet_by_s, pure_rk_by_s, adp_pool_by_s, played_before_all, _ = de.build_all_training_datasets(
        data, seasons_dirs, season_data, births_cache, profiles_cache,
    )
    vegas_odds = de.load_vegas_preseason_odds(data)

    # Load historical 12-team sim summary by season (if available) to display in season header
    sim_by_season = {}
    sim_csv = Path(sim_csv_override) if sim_csv_override else (data / f"draft_engine_sim_by_season_{scoring_name}.csv")
    if sim_csv.is_file():
        for row in de.read_csv(sim_csv):
            if row.get("strategy") in ("residual_unified", "residual_wait"):
                sim_by_season[(row["season"], row["strategy"])] = {
                    "diff": float(row["espn_diff"]),
                    "rank": float(row["avg_rank"]),
                    "win_pct": float(row["win_pct"]),
                    "top4_pct": float(row["top4_pct"]),
                    "drafts": int(row["drafts"]) if row.get("drafts") else 720,
                }

    eff_min_s = None if (not min_train_season or str(min_train_season).lower() == "all") else str(min_train_season)
    train_adp_seasons = sorted(s for s in adp_pool_by_s if eff_min_s is None or s >= eff_min_s)
    eval_seasons = [s for s in train_adp_seasons if len(adp_pool_by_s[s]) >= 156]
    if len(eval_seasons) <= 2:
        eval_seasons = train_adp_seasons
    development, holdout = eval_seasons[:-2], eval_seasons[-2:]
    cv_eligible = [s for s in development if sum(1 for ts in train_adp_seasons if ts < s) >= 2]
    cv_seasons = cv_eligible[-cv_folds:] if cv_eligible else development[1:]
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
            tr_pv = [r for s, grp in pure_vet_by_s.items() if (eff_min_s is None or s >= eff_min_s) and s <= cutoff for r in grp]
            tr_pr = [r for s, grp in pure_rk_by_s.items() if (eff_min_s is None or s >= eff_min_s) and s <= cutoff for r in grp]
            tr_adp = [r for s, grp in adp_pool_by_s.items() if (eff_min_s is None or s >= eff_min_s) and s <= cutoff for r in grp]
        else:
            tr_pv = [r for s, grp in pure_vet_by_s.items() if (eff_min_s is None or s >= eff_min_s) and s < target_s for r in grp]
            tr_pr = [r for s, grp in pure_rk_by_s.items() if (eff_min_s is None or s >= eff_min_s) and s < target_s for r in grp]
            tr_adp = [r for s, grp in adp_pool_by_s.items() if (eff_min_s is None or s >= eff_min_s) and s < target_s for r in grp]

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
                vegas_odds=vegas_odds,
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
        res_s2 = de.fit_predict_residual_unified(tr_adp, s2_cands, prior_s["waiver_fppg"], lam=0.50, recency_halflife=recency_halflife)
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
            target_profs, pb_prior, adp_entries, tr_pv, tr_pr, vegas_odds=vegas_odds,
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
        sim_live = simulate_live_points_season(data / target_s, scoring, s2_cands) if is_live else None

        seasons_payload[target_s] = {
            "season": target_s,
            "split": split,
            "max_step": tl["max_step"],
            "prior_waiver": round(prior_s["waiver_fppg"], 2),
            "peer_center": round(float(peer_center), 2),
            "sim_greedy": sim_g,
            "sim_wait": sim_w,
            "sim_live": sim_live,
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
<title>ADP-Based Pre-Draft Projections (Points)</title>
<style>
  :root {
    --bg: #f4f7fb;
    --card: #ffffff;
    --card-soft: #f8fafc;
    --ink: #0f172a;
    --ink-2: #334155;
    --muted: #64748b;
    --border: #e2e8f0;
    --border-strong: #cbd5e1;
    --row-alt: #f8fafc;
    --row-hover: #eff6ff;
    --pos: #059669;
    --pos-bg: #ecfdf5;
    --pos-bd: #a7f3d0;
    --neg: #e11d48;
    --neg-bg: #fff1f2;
    --neg-bd: #fecdd3;
    --amber: #d97706;
    --amber-bg: #fffbeb;
    --amber-bd: #fde68a;
    --accent: #2563eb;
    --accent-bg: #eff6ff;
    --accent-bd: #bfdbfe;
    --orange: #ea580c;
    --shadow: 0 1px 3px rgba(15, 23, 42, 0.06), 0 1px 2px rgba(15, 23, 42, 0.04);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 24px 18px 56px;
    background: linear-gradient(180deg, #eef4ff 0%, var(--bg) 220px);
    color: var(--ink);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.45;
  }
  .wrap { max-width: 1220px; margin: 0 auto; }

  /* Compact Header */
  header {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 16px;
  }
  .title-row {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }
  h1 {
    margin: 0;
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.025em;
    color: var(--ink);
  }
  .split-badge {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 3px 9px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--card);
    color: var(--muted);
  }
  .split-badge.live { background: var(--pos-bg); border-color: var(--pos-bd); color: var(--pos); }
  .split-badge.holdout { background: var(--amber-bg); border-color: var(--amber-bd); color: var(--amber); }
  .split-badge.cv { background: var(--accent-bg); border-color: var(--accent-bd); color: var(--accent); }
  .fmt-pill { font-size: 11px; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase; padding: 3px 10px; border-radius: 999px; background: #0f172a; color: #ffffff; }
  .header-meta {
    font-size: 13px;
    color: var(--muted);
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 5px 13px;
    box-shadow: var(--shadow);
  }
  .header-meta strong { color: var(--ink); }

  /* Season Bar */
  .season-bar {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 10px 14px;
    margin-bottom: 14px;
    box-shadow: var(--shadow);
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 12px;
  }
  .season-group {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 5px;
  }
  .season-group-lbl {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
    margin-right: 3px;
  }
  .season-btn {
    font-size: 12px;
    font-weight: 600;
    background: var(--card-soft);
    border: 1px solid var(--border);
    color: var(--ink-2);
    padding: 5px 10px;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.12s ease;
  }
  .season-btn:hover { background: var(--accent-bg); border-color: var(--accent-bd); color: var(--accent); }
  .season-btn.holdout { border-color: var(--amber-bd); color: var(--amber); background: var(--amber-bg); }
  .season-btn.live { border-color: var(--pos-bd); color: var(--pos); background: var(--pos-bg); }
  .season-btn.active {
    background: var(--ink);
    border-color: var(--ink);
    color: #ffffff;
  }
  .season-sim-note {
    margin-left: auto;
    font-size: 12.5px;
    color: var(--muted);
    font-weight: 500;
  }
  .season-sim-note strong { color: var(--pos); font-weight: 700; }

  /* Replay Card */
  .replay-panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 14px;
    box-shadow: var(--shadow);
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
    gap: 6px;
  }
  .play-btn {
    font-size: 12.5px;
    font-weight: 700;
    background: var(--pos);
    border: 1px solid var(--pos);
    color: #ffffff;
    padding: 6px 14px;
    border-radius: 8px;
    cursor: pointer;
    min-width: 108px;
    transition: opacity 0.12s ease;
  }
  .play-btn:hover { opacity: 0.92; }
  .play-btn.playing { background: var(--orange); border-color: var(--orange); }
  .mini-btn {
    font-size: 12px;
    font-weight: 600;
    background: var(--card-soft);
    border: 1px solid var(--border);
    color: var(--ink-2);
    padding: 5px 10px;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.12s ease;
  }
  .mini-btn:hover { background: var(--row-hover); border-color: var(--accent-bd); color: var(--accent); }
  .mini-btn.active {
    background: var(--accent-bg);
    border-color: var(--accent-bd);
    color: var(--accent);
  }
  .replay-clock {
    font-size: 13px;
    font-weight: 600;
    color: var(--ink);
    font-variant-numeric: tabular-nums;
  }
  .replay-clock span { color: var(--muted); font-weight: 500; }
  .slider-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
  }
  .slider-end {
    font-size: 11.5px;
    font-weight: 600;
    color: var(--muted);
    min-width: 34px;
    font-variant-numeric: tabular-nums;
  }
  .slider-end.right { text-align: right; }
  input[type="range"]#game-slider {
    flex: 1;
    accent-color: var(--accent);
    cursor: pointer;
    height: 6px;
  }
  .chart-header {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }
  .chart-title {
    font-size: 12px;
    font-weight: 600;
    color: var(--muted);
  }
  .chart-modes { display: flex; gap: 5px; flex-wrap: wrap; }
  #edge-chart {
    width: 100%;
    height: 116px;
    display: block;
    background: #fafcff;
    border: 1px solid var(--border);
    border-radius: 8px;
    cursor: crosshair;
  }

  /* 4 Crisp KPI Cards */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
    margin-bottom: 12px;
  }
  .stat-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: var(--shadow);
  }
  .stat-label {
    font-size: 11.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
    margin-bottom: 6px;
  }
  .stat-val {
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.15;
    color: var(--ink);
    font-variant-numeric: tabular-nums;
  }
  .stat-sub {
    font-size: 12.5px;
    color: var(--muted);
    margin-top: 5px;
  }
  .stat-sub strong { color: var(--ink-2); font-weight: 600; }

  /* Compact Round Strip */
  .bands {
    font-size: 12.5px;
    color: var(--muted);
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 9px 14px;
    margin-bottom: 14px;
    box-shadow: var(--shadow);
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    align-items: center;
    font-variant-numeric: tabular-nums;
  }
  .bands .sep { color: var(--border-strong); }
  .bands strong { color: var(--ink); font-weight: 700; }

  /* Controls & Table */
  .controls {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
  }
  .tabs { display: flex; flex-wrap: wrap; gap: 6px; }
  .tab-btn {
    font-size: 13px;
    font-weight: 600;
    background: var(--card);
    border: 1px solid var(--border);
    color: var(--ink-2);
    padding: 7px 13px;
    border-radius: 9px;
    cursor: pointer;
    box-shadow: var(--shadow);
    transition: all 0.12s ease;
  }
  .tab-btn:hover { border-color: var(--accent-bd); color: var(--accent); }
  .tab-btn.active {
    background: var(--accent);
    border-color: var(--accent);
    color: #ffffff;
  }
  .filters { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
  .filters input[type="text"] {
    font-size: 13px;
    padding: 6px 11px;
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    background: var(--card);
    color: var(--ink);
    width: 170px;
    outline: none;
  }
  .filters input[type="text"]:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-bg); }
  .filters input[type="text"]::placeholder { color: var(--muted); }
  .pill {
    font-size: 12px;
    font-weight: 600;
    background: var(--card);
    border: 1px solid var(--border);
    color: var(--ink-2);
    padding: 6px 10px;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.12s ease;
  }
  .pill:hover { background: var(--row-hover); color: var(--accent); }
  .pill.active {
    background: var(--accent-bg);
    border-color: var(--accent-bd);
    color: var(--accent);
  }

  /* Compact Pop-down Guide */
  .guide-drawer {
    display: none;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    box-shadow: var(--shadow);
    font-size: 12.5px;
    color: var(--ink-2);
  }
  .guide-drawer.open { display: block; }
  .guide-items {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 20px;
  }
  .guide-items span { white-space: nowrap; }
  .guide-items code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11.5px;
    font-weight: 700;
    background: var(--card-soft);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1px 5px;
    color: var(--ink);
    margin-right: 4px;
  }

  .table-meta {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
    font-size: 12.5px;
    color: var(--muted);
    margin-bottom: 8px;
    padding: 0 2px;
    font-variant-numeric: tabular-nums;
  }
  .table-meta strong { color: var(--ink-2); }
  .table-wrap {
    overflow-x: auto;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow);
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    font-variant-numeric: tabular-nums;
  }
  thead th {
    position: sticky;
    top: 0;
    background: #f8fafc;
    border-bottom: 1px solid var(--border-strong);
    padding: 10px 10px;
    text-align: right;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
  }
  thead th.left { text-align: left; }
  thead th.group-sep, tbody td.group-sep { border-left: 1px solid var(--border); }
  thead th:hover { background: #f1f5f9; color: var(--ink); }
  thead th .sort-arrow { color: var(--accent); font-size: 10px; margin-left: 3px; }
  tbody tr { border-bottom: 1px solid #f1f5f9; transition: background 0.08s ease; }
  tbody tr:nth-child(even) { background: var(--row-alt); }
  tbody tr:hover { background: var(--row-hover); }
  tbody td { padding: 8px 10px; text-align: right; white-space: nowrap; color: var(--ink-2); }
  tbody td.left { text-align: left; }
  td.player-cell {
    font-size: 13.5px;
    font-weight: 600;
    color: var(--ink);
  }
  .sub-num { color: var(--muted); font-size: 11.5px; margin-left: 3px; }
  .tag {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    padding: 1px 6px;
    border-radius: 999px;
    border: 1px solid var(--border);
    color: var(--muted);
    margin-left: 6px;
    vertical-align: 1px;
  }
  .tag-rook { border-color: var(--accent-bd); color: var(--accent); background: var(--accent-bg); }
  .tag-ret { border-color: #fed7aa; color: var(--orange); background: #fff7ed; }
  .tag-inj { border-color: var(--amber-bd); color: var(--amber); background: var(--amber-bg); }
  .pos { color: var(--pos); font-weight: 700; }
  .neg { color: var(--neg); font-weight: 700; }
  .muted { color: var(--muted); }
  .edge-pill {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 6px;
    min-width: 52px;
    text-align: right;
    font-weight: 700;
  }
  .edge-pos { background: var(--pos-bg); color: var(--pos); }
  .edge-neg { background: var(--neg-bg); color: var(--neg); }
  .edge-flat { color: var(--muted); font-weight: 500; }
  .call-badge {
    display: inline-block;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 999px;
    border: 1px solid transparent;
    font-weight: 700;
    min-width: 58px;
    text-align: center;
  }
  .call-hit { color: var(--pos); background: var(--pos-bg); border-color: var(--pos-bd); }
  .call-miss { color: var(--neg); background: var(--neg-bg); border-color: var(--neg-bd); }
  .call-agree { color: var(--muted); background: var(--card-soft); border-color: var(--border); font-weight: 600; }

  /* Plain-English Explainer Box */
  .title-block { display: flex; flex-direction: column; gap: 3px; }
  .page-tagline { font-size: 13px; color: var(--muted); font-weight: 500; }

  .intro-cards {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 14px;
    align-items: start;
  }
  @media (max-width: 860px) { .intro-cards { grid-template-columns: 1fr; } }
  .intro-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 12px;
    padding: 11px 14px;
    box-shadow: var(--shadow);
  }
  .intro-box.does { border-left-color: var(--pos); }
  .intro-box.doesnt { border-left-color: var(--neg); }
  .intro-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    cursor: pointer;
    user-select: none;
  }
  .intro-title {
    font-size: 13px;
    font-weight: 700;
    color: var(--ink);
  }
  .intro-title span {
    color: var(--accent);
    margin-right: 6px;
  }
  .intro-box.does .intro-title span { color: var(--pos); }
  .intro-box.doesnt .intro-title span { color: var(--neg); }
  .intro-toggle {
    font-size: 11.5px;
    font-weight: 600;
    color: var(--muted);
    background: var(--card-soft);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 2px 8px;
    cursor: pointer;
    flex-shrink: 0;
  }
  .intro-toggle:hover { color: var(--accent); border-color: var(--accent-bd); }
  .intro-body {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
    font-size: 12.5px;
    color: var(--ink-2);
    line-height: 1.48;
  }
  .intro-box.collapsed .intro-body { display: none; }

  /* Prominent Simulated Win-Rate Banner */
  .sim-hero {
    background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%);
    border: 1px solid var(--pos-bd);
    border-left: 4px solid var(--pos);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 14px;
    box-shadow: var(--shadow);
    display: grid;
    grid-template-columns: 1.25fr 1fr 1.25fr;
    gap: 16px;
    align-items: center;
  }
  @media (max-width: 860px) { .sim-hero { grid-template-columns: 1fr; } }
  .sim-hero-lbl {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
    margin-bottom: 4px;
  }
  .sim-hero-big {
    display: flex;
    align-items: baseline;
    gap: 9px;
    flex-wrap: wrap;
  }
  .sim-hero-num {
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--pos);
    line-height: 1.05;
    font-variant-numeric: tabular-nums;
  }
  .sim-hero-mult {
    font-size: 12px;
    font-weight: 700;
    color: var(--pos);
    background: var(--pos-bg);
    border: 1px solid var(--pos-bd);
    padding: 2px 8px;
    border-radius: 999px;
  }
  .sim-hero-sub {
    font-size: 12px;
    color: var(--ink-2);
    margin-top: 5px;
    font-variant-numeric: tabular-nums;
  }
  .sim-bars { margin-top: 7px; display: flex; flex-direction: column; gap: 4px; }
  .sim-bar-row { display: flex; align-items: center; gap: 8px; font-size: 11px; color: var(--muted); font-variant-numeric: tabular-nums; }
  .sim-bar-lbl { min-width: 72px; font-weight: 600; }
  .sim-bar-track { flex: 1; height: 7px; background: #e2e8f0; border-radius: 999px; overflow: hidden; }
  .sim-bar-fill { height: 100%; border-radius: 999px; }
  .sim-bar-fill.base { background: #94a3b8; }
  .sim-bar-fill.model { background: var(--pos); }
  .sim-hero-div { border-left: 1px solid var(--pos-bd); padding-left: 16px; }
  @media (max-width: 860px) { .sim-hero-div { border-left: none; padding-left: 0; border-top: 1px solid var(--pos-bd); padding-top: 12px; } }
  .sim-hero-season-num { font-size: 24px; font-weight: 800; color: var(--ink); line-height: 1.1; font-variant-numeric: tabular-nums; }
  .sim-hero-note { font-size: 12.5px; color: var(--ink-2); line-height: 1.45; }

  .call-legend {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px 16px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 8px 13px;
    margin-bottom: 10px;
    font-size: 12px;
    color: var(--ink-2);
    box-shadow: var(--shadow);
  }
  .call-legend-lbl {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
  }
  .call-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    flex-wrap: wrap;
  }
  .call-legend-item .call-badge { min-width: auto; padding: 1px 6px; font-size: 10px; }

  footer {
    margin-top: 18px;
    font-size: 12px;
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
    <div class="title-block">
      <div class="title-row">
        <h1 id="page-title">2026&ndash;27 ADP-Based Pre-Draft Projections</h1>
        <span class="fmt-pill">POINTS</span>
        <span class="split-badge live" id="split-badge">LIVE</span>
      </div>
      <div class="page-tagline">Pre-draft player projections built to beat consensus ADP on average &mdash; tested out-of-sample below.</div>
    </div>
    <div class="header-meta" id="header-meta"></div>
  </header>

  <section class="sim-hero" id="sim-hero"></section>

  <div class="intro-cards">
    <section class="intro-box why collapsed">
      <div class="intro-head">
        <div class="intro-title"><span>&#9679;</span>Why This Exists</div>
        <button class="intro-toggle">Show</button>
      </div>
      <div class="intro-body">Fantasy leagues are already <strong>well projected in general</strong>&mdash;consensus ADP gets most players roughly right, and throwing it away to guess from scratch loses badly. The edge comes from repeatable <strong>human biases on a small group of players</strong>: overpaying for rookie hype, aging/injury-prone names, and hot streaks while overlooking durable, rising starters. Avoiding a few landmines and hitting a few undervalued targets is what creates league-winning teams.</div>
    </section>
    <section class="intro-box does collapsed">
      <div class="intro-head">
        <div class="intro-title"><span>&#9679;</span>What This Board Is</div>
        <button class="intro-toggle">Show</button>
      </div>
      <div class="intro-body"><strong>Pre-draft player projections</strong> built on top of ADP (scored on a full-season 65-game reliability scale) that we believe outperform consensus ADP on average&mdash;backed by out-of-sample historical results and live tracking below. Green <strong style="color:var(--pos)">Edge</strong> highlights players we project higher than ADP; red highlights players we project lower.</div>
    </section>
    <section class="intro-box doesnt collapsed">
      <div class="intro-head">
        <div class="intro-title"><span>&#9679;</span>What It Doesn&rsquo;t Cover</div>
        <button class="intro-toggle">Show</button>
      </div>
      <div class="intro-body">Anything after draft day&mdash;<strong>no</strong> waiver pickups, trades, mid-season injuries, or coaching/rotation changes. It is strictly a pre-draft projection to help you leave the draft with a better roster; managing the season is up to you.</div>
    </section>
  </div>

  <div class="season-bar" id="season-bar"></div>

  <section class="replay-panel">
    <div class="replay-top">
      <div class="replay-controls">
        <button class="play-btn" id="play-btn">&#9654; Play</button>
        <button class="mini-btn" id="reset-btn">G 1</button>
        <button class="mini-btn" id="end-btn">End</button>
        <button class="mini-btn active" data-speed="250">4&times;</button>
        <button class="mini-btn" data-speed="125">8&times;</button>
      </div>
      <div class="replay-clock" id="replay-clock"></div>
    </div>
    <div class="slider-row">
      <span class="slider-end">G 1</span>
      <input type="range" id="game-slider" min="1" max="82" value="82">
      <span class="slider-end right" id="slider-max-lbl">G 82</span>
    </div>
    <div class="chart-header">
      <div class="chart-title">Season Trajectory (click chart to scrub)</div>
      <div class="chart-modes">
        <button class="mini-btn active" data-chart="t15_fpts">Top 15 Calls</button>
        <button class="mini-btn" data-chart="top150_fpts">Top 150 FPTS</button>
        <button class="mini-btn" data-chart="corr_r">Edge Corr (r)</button>
      </div>
    </div>
    <svg id="edge-chart" viewBox="0 0 1000 116" preserveAspectRatio="none"></svg>
  </section>

  <section class="stats-grid" id="hero-cards"></section>
  <div class="bands" id="band-strip"></div>

  <div class="controls">
    <div class="tabs" id="tabs">
      <button class="tab-btn active" data-tab="stage2_top150">Top 150</button>
      <button class="tab-btn" data-tab="stage1_top150">Pure Stats (0% ADP)</button>
      <button class="tab-btn" data-tab="deep_sleepers">Sleepers (151&ndash;225)</button>
    </div>
    <div class="filters">
      <input type="text" id="search" placeholder="Search player or pos...">
      <button class="pill active" data-filter="all">All</button>
      <button class="pill" data-filter="top15">Top 15</button>
      <button class="pill" data-filter="pos">Targets</button>
      <button class="pill" data-filter="neg">Fades</button>
      <button class="pill" data-filter="rook">Rookies</button>
      <button class="pill" id="guide-toggle">? Guide</button>
    </div>
  </div>

  <div class="guide-drawer" id="guide-drawer">
    <div class="guide-items">
      <span><code>&Delta;Rk</code> ADP rank minus model rank (+)</span>
      <span><code>Edge</code> Predicted AdjFPPG vs. peers</span>
      <span><code>Proj</code> Pre-season AdjFPPG (65g scale)</span>
      <span><code>Act Rk</code> Actual NBA rank at game</span>
      <span><code>vs ADP</code> Actual AdjFPPG minus ADP slot</span>
      <span><code>Call</code> Grades |Edge| &ge; 0.25 or |&Delta;Rk| &ge; 3</span>
      <span><code>Rank Flips</code> % of player pairs where the model reversed ADP order and picked the higher scorer</span>
    </div>
  </div>

  <div class="call-legend" aria-label="Call badge legend">
    <span class="call-legend-lbl">Call Key:</span>
    <span class="call-legend-item"><span class="call-badge call-hit">TGT &#10003;</span><span class="call-badge call-miss">TGT &#10007;</span> <strong>Target</strong> (projected above ADP; &#10003; beat ADP, &#10007; fell short)</span>
    <span class="call-legend-item"><span class="call-badge call-hit">FADE &#10003;</span><span class="call-badge call-miss">FADE &#10007;</span> <strong>Fade</strong> (projected below ADP; &#10003; underperformed ADP, &#10007; beat ADP)</span>
    <span class="call-legend-item"><span class="call-badge call-agree">AGREE</span> <strong>Agree</strong> (matches ADP within |Edge| &lt; 0.25 &amp; |&Delta;Rk| &lt; 3)</span>
  </div>

  <div class="table-meta">
    <span id="row-count"></span>
    <span id="filtered-summary"></span>
  </div>

  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th data-col="rk">Rk<span class="sort-arrow"></span></th>
          <th class="left" data-col="p">Player<span class="sort-arrow"></span></th>
          <th class="left" data-col="pos">Pos<span class="sort-arrow"></span></th>
          <th data-col="mrk">ADP<span class="sort-arrow"></span></th>
          <th data-col="drk">&Delta;Rk<span class="sort-arrow"></span></th>
          <th data-col="e">Edge<span class="sort-arrow"></span></th>
          <th data-col="pr">Proj<span class="sort-arrow"></span></th>
          <th class="group-sep" data-col="gp">GP<span class="sort-arrow"></span></th>
          <th data-col="fpts">FPTS<span class="sort-arrow"></span></th>
          <th data-col="act_rank">Act Rk<span class="sort-arrow"></span></th>
          <th data-col="act_vs_adp">vs ADP<span class="sort-arrow"></span></th>
          <th data-col="call_score">Call<span class="sort-arrow"></span></th>
        </tr>
      </thead>
      <tbody id="board-body"></tbody>
    </table>
  </div>

  <footer>
    <span>Strict out-of-sample pre-season projections (ESPN default scoring)</span>
    <span id="footer-sim-summary"></span>
  </footer>
</div>

<script>
const APP = __PAYLOAD_JSON__;
const ANCHORS_RK = APP.anchor_ranks;

const state = {
  season: APP.default_season,
  step: 0,
  tab: "stage2_top150",
  filter: "all",
  search: "",
  sortCol: "rk",
  sortDir: 1,
  chartMode: "t15_fpts",
  speedMs: 250,
  timer: null,
};

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

function renderSimHero() {
  const S = APP.seasons[state.season];
  const allRows = APP.season_order.map(o => APP.seasons[o.season]?.sim_greedy).filter(Boolean);
  const el = document.getElementById("sim-hero");
  if (!el || !allRows.length) return;

  const nYrs = allRows.length;
  const totalDr = allRows.reduce((a, r) => a + (r.drafts || 720), 0);
  const perYr = Math.round(totalDr / nYrs);
  const avgWin = allRows.reduce((a, r) => a + r.win_pct, 0) / nYrs;
  const avgDiff = allRows.reduce((a, r) => a + r.diff, 0) / nYrs;
  const avgRk = allRows.reduce((a, r) => a + r.rank, 0) / nYrs;
  const avgT4 = allRows.reduce((a, r) => a + r.top4_pct, 0) / nYrs;
  const wins = allRows.filter(r => r.diff > 0).length;
  const mult = avgWin / (100 / 12);
  const maxBar = Math.max(35, avgWin * 1.15);

  const cur = S.sim_greedy;
  const live = S.sim_live;
  let seasonHtml = "";
  if (cur) {
    const cls = cur.diff >= 0 ? "pos" : "neg";
    seasonHtml = `
      <div class="sim-hero-lbl">Selected Season (${S.season} &middot; ${cur.drafts || 720} drafts)</div>
      <div class="sim-hero-season-num ${cls}">${cur.win_pct.toFixed(1)}% Win Rate</div>
      <div class="sim-hero-sub"><strong>${fmtSignedInt(cur.diff)} FPTS</strong> vs. field &middot; Avg Rank <strong>#${cur.rank.toFixed(2)}</strong> &middot; Top-4 <strong>${cur.top4_pct.toFixed(1)}%</strong></div>
    `;
  } else if (live) {
    const cls = live.diff >= 0 ? "pos" : "neg";
    seasonHtml = `
      <div class="sim-hero-lbl">Selected Season (${S.season} &middot; Live Draft vs. ADP &middot; ${live.min_games || S.max_step} Equalized G)</div>
      <div class="sim-hero-season-num ${cls}">${fmtSignedInt(live.diff)} FPTS <span style="font-size:15px;font-weight:700">(${fmtSigned(live.lift_pct, 1)}%)</span></div>
      <div class="sim-hero-sub">Avg Projected Draft <strong>${fmtInt(live.hero_fpts)}</strong> vs. Avg ADP Draft <strong>${fmtInt(live.field_fpts)}</strong> FPTS &middot; ${live.drafts} draft-only leagues (first ${live.min_games || S.max_step} games/team, early sample)</div>
    `;
  } else {
    seasonHtml = `
      <div class="sim-hero-lbl">Selected Season (${S.season} &middot; Live)</div>
      <div class="sim-hero-season-num muted">Season In Progress</div>
      <div class="sim-hero-sub">Zero-leakage live board &middot; Tracked in real time below (${S.max_step} team-games played)</div>
    `;
  }

  el.innerHTML = `
    <div>
      <div class="sim-hero-lbl">${nYrs}-Year Draft-Only Simulated Win Rate (${totalDr.toLocaleString()} Leagues)</div>
      <div class="sim-hero-big">
        <span class="sim-hero-num">${avgWin.toFixed(1)}%</span>
        <span class="sim-hero-mult">${mult.toFixed(1)}&times; baseline</span>
      </div>
      <div class="sim-bars">
        <div class="sim-bar-row"><span class="sim-bar-lbl">ADP (1/12)</span><div class="sim-bar-track"><div class="sim-bar-fill base" style="width:${(8.33 / maxBar * 100).toFixed(1)}%"></div></div><span>8.3%</span></div>
        <div class="sim-bar-row"><span class="sim-bar-lbl">This Board</span><div class="sim-bar-track"><div class="sim-bar-fill model" style="width:${(avgWin / maxBar * 100).toFixed(1)}%"></div></div><strong>${avgWin.toFixed(1)}%</strong></div>
      </div>
      <div class="sim-hero-sub"><strong>${fmtSignedInt(avgDiff)} FPTS</strong> avg &middot; Rank <strong>#${avgRk.toFixed(2)}</strong> &middot; Top-4 <strong>${avgT4.toFixed(1)}%</strong> &middot; <strong>${wins}/${nYrs}</strong> winning yrs</div>
    </div>
    <div class="sim-hero-div">${seasonHtml}</div>
    <div class="sim-hero-div">
      <div class="sim-hero-lbl">What Simulated Win Rate Means</div>
      <div class="sim-hero-note">In a fair 12-team league, everyone starts with an <strong>8.3% (1 in 12)</strong> chance to finish 1st. We simulate <strong>${perYr} draft-and-hold leagues/yr</strong> where 11 teams draft by ADP and 1 team blindly picks the highest-ranked available player here&mdash;<strong>zero waiting tricks, trades, or mid-season moves</strong>.</div>
    </div>
  `;
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
  const simG = S.sim_greedy;
  const simL = S.sim_live;
  if (simG) {
    const nDr = simG.drafts || 720;
    simHtml = `<span title="720 draft-and-hold 12-team leagues blindly picking the top-ranked available player (baseline 12-team win rate = 8.3%; no trades or mid-season moves).">Draft-Only Sim (${nDr} dr): <strong>${fmtSignedInt(simG.diff)} FPTS</strong> &middot; ${simG.win_pct.toFixed(1)}% Win <span class="muted">(vs 8.3% base)</span></span>`;
  } else if (simL) {
    simHtml = `<span title="Early-season draft edge: average team drafted from this board vs. average ADP-drafted team across ${simL.drafts} draft-and-hold leagues so far (excluded from multi-year completed-season win rate).">Live Draft vs. ADP (${simL.drafts} dr): <strong>${fmtSignedInt(simL.diff)} FPTS</strong> <span class="muted">(${fmtSigned(simL.lift_pct, 1)}% &middot; ${fmtInt(simL.hero_fpts)} vs ${fmtInt(simL.field_fpts)})</span></span>`;
  } else {
    simHtml = `Live season &middot; ${S.max_step} team-games`;
  }

  document.getElementById("season-bar").innerHTML = `
    <div class="season-group"><span class="season-group-lbl">CV</span>${mkBtns(cv)}</div>
    <div class="season-group"><span class="season-group-lbl">Holdout</span>${mkBtns(ho)}</div>
    <div class="season-group"><span class="season-group-lbl">Live</span>${mkBtns(lv)}</div>
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
  const W = 1000, H = 116, padL = 56, padR = 16, padT = 14, padB = 20;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  let minV = Math.min(0, ...arr);
  let maxV = Math.max(0, ...arr);
  if (maxV - minV < 1e-4) { maxV += 1; minV -= 1; }
  const margin = (maxV - minV) * 0.14;
  minV -= margin; maxV += margin;

  const xOf = (i) => padL + (n > 1 ? (i / (n - 1)) * plotW : plotW / 2);
  const yOf = (v) => padT + ((maxV - v) / (maxV - minV)) * plotH;
  const y0 = yOf(0);

  const fullPts = arr.map((v, i) => `${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`).join(" ");
  const curSlice = arr.slice(0, state.step + 1);
  const curPts = curSlice.map((v, i) => `${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`).join(" ");
  const curVal = arr[state.step];
  const curX = xOf(state.step);
  const curY = yOf(curVal);
  const strokeCol = curVal >= 0 ? "#059669" : "#e11d48";
  const fillCol = curVal >= 0 ? "rgba(5,150,105,0.12)" : "rgba(225,29,72,0.10)";
  const areaPts = `${xOf(0).toFixed(1)},${y0.toFixed(1)} ${curPts} ${curX.toFixed(1)},${y0.toFixed(1)}`;

  const fmtTick = (v) => state.chartMode === "corr_r" ? (v >= 0 ? "+" : "") + v.toFixed(2) : fmtSignedInt(v);

  svg.innerHTML = `
    <line x1="${padL}" y1="${padT}" x2="${W - padR}" y2="${padT}" stroke="#eef2f7" stroke-width="1"/>
    <line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" stroke="#eef2f7" stroke-width="1"/>
    <line x1="${padL}" y1="${y0.toFixed(1)}" x2="${W - padR}" y2="${y0.toFixed(1)}" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="${padL - 8}" y="${padT + 9}" text-anchor="end" fill="#64748b" font-size="10.5">${fmtTick(maxV)}</text>
    <text x="${padL - 8}" y="${(y0 + 3).toFixed(1)}" text-anchor="end" fill="#64748b" font-size="10.5">0</text>
    <text x="${padL - 8}" y="${H - padB}" text-anchor="end" fill="#64748b" font-size="10.5">${fmtTick(minV)}</text>
    <polyline fill="none" stroke="#cbd5e1" stroke-width="1.5" points="${fullPts}"/>
    <polygon fill="${fillCol}" points="${areaPts}"/>
    <polyline fill="none" stroke="${strokeCol}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" points="${curPts}"/>
    <line x1="${curX.toFixed(1)}" y1="${padT}" x2="${curX.toFixed(1)}" y2="${H - padB}" stroke="#94a3b8" stroke-width="1" stroke-dasharray="2,2"/>
    <circle cx="${curX.toFixed(1)}" cy="${curY.toFixed(1)}" r="4.5" fill="${strokeCol}" stroke="#ffffff" stroke-width="2"/>
    <text x="${Math.min(W - padR - 4, Math.max(padL + 6, curX + 9)).toFixed(1)}" y="${Math.max(padT + 12, curY - 7).toFixed(1)}" fill="${strokeCol}" font-size="11.5" font-weight="700">${fmtTick(curVal)}</text>
    <text x="${padL}" y="${H - 5}" fill="#64748b" font-size="10.5">Game 1</text>
    <text x="${W - padR}" y="${H - 5}" text-anchor="end" fill="#64748b" font-size="10.5">Game ${n}</text>
  `;
}

function renderAll() {
  const S = APP.seasons[state.season];
  const cum = getSeasonCum(state.season);
  const cp = S.checkpoints[state.step];

  document.getElementById("page-title").textContent = `${S.season} ADP-Based Pre-Draft Projections`;
  const badge = document.getElementById("split-badge");
  badge.textContent = S.split.toUpperCase();
  badge.className = `split-badge ${S.split}`;

  document.getElementById("header-meta").innerHTML =
    `<strong>Game ${cp.g}/${S.max_step}</strong> &middot; ${cp.date} &middot; ${cp.pct}% of season`;
  document.getElementById("replay-clock").innerHTML =
    `Game ${cp.g} / ${S.max_step} <span>&middot; ${cp.date} &middot; ${cp.reg} NBA games</span>`;
  document.getElementById("slider-max-lbl").textContent = `G ${S.max_step}`;

  const s2All = S.s2_all.map(r => enrichRow(r, cum, cp));
  const s2Top150 = s2All.slice(0, 150);
  const mktTop150 = s2All.filter(r => r.mrk !== null && r.mrk <= 150).sort((a, b) => a.mrk - b.mrk);
  const deepPool = s2All.filter(r => r.mrk !== null && r.mrk > 150);
  const s1Top150 = S.s1_150.map(r => enrichRow(r, cum, cp));

  const s2_150_diff = s2Top150.reduce((a, r) => a + r.fpts, 0) - mktTop150.reduce((a, r) => a + r.fpts, 0);
  const s2_swaps = s2Top150.filter(r => r.mrk > 150).length;

  const byE = [...mktTop150].sort((a, b) => b.e - a.e);
  const t15 = byE.slice(0, 15);
  const f15 = byE.slice(-15);
  const t15Diff = t15.reduce((a, r) => a + r.fpts, 0) - f15.reduce((a, r) => a + r.fpts, 0);
  const t15VsSpread = (t15.reduce((a, r) => a + r.act_vs_adp, 0) - f15.reduce((a, r) => a + r.act_vs_adp, 0)) / 15;

  const corrAll = pearson(mktTop150.map(r => r.re), mktTop150.map(r => r.act_vs_adp));
  let flipsTotal = 0, flipsWon = 0;
  for (let i = 0; i < mktTop150.length; i++) {
    for (let j = i + 1; j < mktTop150.length; j++) {
      if (mktTop150[j].pr > mktTop150[i].pr) {
        flipsTotal++;
        if (mktTop150[j].act_adj > mktTop150[i].act_adj) flipsWon++;
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

  const cards = [
    {
      label: "Top-150 Cutoff Swaps",
      val: `${fmtSignedInt(s2_150_diff)} FPTS`,
      cls: s2_150_diff >= 0 ? "pos" : "neg",
      sub: `<span title="Measures only the ~10-15 borderline swaps across the #150 cut line (the ~138 shared players inside Top 150 cancel out). See Top 15 Calls, Correlation, and Sim for full Rounds 1-13 performance."><strong>${s2Cap}/150</strong> hits (Mkt ${mktCap}) &middot; ${s2_swaps} cut-line swaps</span>`,
    },
    {
      label: "Top 15 Targets vs. Fades",
      val: `${fmtSignedInt(t15Diff)} FPTS`,
      cls: t15Diff >= 0 ? "pos" : "neg",
      sub: `<strong>${fmtSigned(t15VsSpread)}</strong> AdjFPPG avg spread`,
    },
    {
      label: "Edge Correlation (r)",
      val: `${corrAll >= 0 ? "+" : ""}${corrAll.toFixed(3)}`,
      cls: corrAll >= 0 ? "pos" : "neg",
      sub: `<span title="Out of every pair of Top-150 players where our model flipped ADP order (ranked a later-ADP player ahead of an earlier-ADP player), how often did our higher-ranked player actually outscore them?"><strong>${flipsPct.toFixed(1)}%</strong> rank flips won (${flipsWon}/${flipsTotal})</span>`,
    },
    {
      label: "Deep Sleepers (151–225)",
      val: `${posDeepHit}/${posDeep.length} (${posDeep.length ? (100 * posDeepHit / posDeep.length).toFixed(0) : "0"}%)`,
      cls: "pos",
      sub: `vs. <strong>${negDeep.length ? (100 * negDeepHit / negDeep.length).toFixed(0) : "0"}%</strong> for neg-edge`,
    },
  ];
  document.getElementById("hero-cards").innerHTML = cards.map(c => `
    <div class="stat-box">
      <div class="stat-label">${c.label}</div>
      <div class="stat-val ${c.cls}">${c.val}</div>
      <div class="stat-sub">${c.sub}</div>
    </div>
  `).join("");

  const bandDefs = [["R1–2", 1, 24], ["R3–5", 25, 60], ["R6–9", 61, 108], ["R10–13", 109, 150]];
  document.getElementById("band-strip").innerHTML =
    `<strong>By Round:</strong>` +
    bandDefs.map(([lbl, lo, hi]) => {
      const sub = mktTop150.filter(r => r.mrk >= lo && r.mrk <= hi);
      const rc = pearson(sub.map(r => r.re), sub.map(r => r.act_vs_adp));
      return `<span>${lbl} <strong class="${rc >= 0 ? "pos" : "neg"}">${rc >= 0 ? "+" : ""}${rc.toFixed(2)}</strong></span>`;
    }).join("") +
    `<span class="sep">|</span><span>Pure Stats (0% ADP): <strong>${s1Cap}/150</strong> hits</span>`;

  renderChart();

  const baseRows = state.tab === "stage2_top150" ? s2Top150 : (state.tab === "stage1_top150" ? s1Top150 : deepPool);
  const q = state.search.trim().toLowerCase();
  const filtered = baseRows.filter(r => {
    if (q && !(r.p.toLowerCase().includes(q) || r.pos.toLowerCase().includes(q))) return false;
    if (state.filter === "top15" && !(r.rk <= 15)) return false;
    if (state.filter === "pos" && !(r.e >= 0.25)) return false;
    if (state.filter === "neg" && !(r.e <= -0.25)) return false;
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

  document.getElementById("row-count").innerHTML =
    `<strong>${filtered.length}</strong> players &middot; <strong>${top150Now}</strong> actual Top-150`;
  document.getElementById("filtered-summary").innerHTML =
    calls ? `Calls: <strong>${hits}/${calls} (${(100 * hits / calls).toFixed(0)}%)</strong>` : "";

  document.getElementById("board-body").innerHTML = filtered.map(r => {
    const call = evalCall(r);
    const typeTag = r.t === "ROOK" ? '<span class="tag tag-rook">ROOK</span>' : (r.t === "RET" ? '<span class="tag tag-ret">RET</span>' : "");
    const injTag = r.al ? `<span class="tag tag-inj">${r.al}</span>` : "";
    const adpStr = r.mrk !== null
      ? `#${r.mrk}<span class="sub-num">(${fmtNum(r.adp, 1)})</span>`
      : '<span class="muted">UNR</span>';
    const deltaCls = r.drk === null ? "muted" : (r.drk > 0 ? "pos" : (r.drk < 0 ? "neg" : "muted"));
    const deltaStr = r.drk === null ? "&mdash;" : (r.drk > 0 ? "+" + r.drk : String(r.drk));
    const edgeCls = r.e >= 0.25 ? "edge-pos" : (r.e <= -0.25 ? "edge-neg" : "edge-flat");
    const actVsCls = r.act_vs_adp > 0.25 ? "pos" : (r.act_vs_adp < -0.25 ? "neg" : "muted");
    const gpCls = r.gp === 0 ? "neg" : "";
    const projTip = r.mp !== null ? `ADP Proj: ${fmtNum(r.mp, 2)}` : "";
    const actTip = `FPPG: ${fmtNum(r.act_fppg, 1)} | Act Adj: ${fmtNum(r.act_adj, 2)}`;

    return `<tr>
      <td><strong>#${r.rk}</strong></td>
      <td class="left player-cell">${r.p}${typeTag}${injTag}</td>
      <td class="left muted">${r.pos}</td>
      <td>${adpStr}</td>
      <td class="${deltaCls}">${deltaStr}</td>
      <td><span class="edge-pill ${edgeCls}">${fmtSigned(r.e, 2)}</span></td>
      <td title="${projTip}"><strong>${fmtNum(r.pr, 1)}</strong></td>
      <td class="group-sep ${gpCls}">${r.gp}</td>
      <td>${fmtInt(r.fpts)}</td>
      <td title="${actTip}">#${r.act_rank}</td>
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
  btn.innerHTML = "&#9654; Play";
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
  renderSimHero();
  renderSeasonBar();
  renderAll();
}

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
document.querySelectorAll(".intro-box .intro-head").forEach(head => {
  head.addEventListener("click", () => {
    const box = head.closest(".intro-box");
    const collapsed = box.classList.toggle("collapsed");
    const btn = box.querySelector(".intro-toggle");
    if (btn) btn.textContent = collapsed ? "Show" : "Hide";
  });
});
document.getElementById("guide-toggle").addEventListener("click", e => {
  const drawer = document.getElementById("guide-drawer");
  const open = drawer.classList.toggle("open");
  e.currentTarget.classList.toggle("active", open);
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
document.querySelectorAll(".pill[data-filter]").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".pill[data-filter]").forEach(b => b.classList.remove("active"));
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

(() => {
  const simRows = APP.season_order.map(o => APP.seasons[o.season]?.sim_greedy).filter(Boolean);
  const el = document.getElementById("footer-sim-summary");
  if (el && simRows.length) {
    const nYrs = simRows.length;
    const totalDrafts = simRows.reduce((a, r) => a + (r.drafts || 720), 0);
    const perYr = Math.round(totalDrafts / nYrs);
    const avgDiff = simRows.reduce((a, r) => a + r.diff, 0) / nYrs;
    const avgWin = simRows.reduce((a, r) => a + r.win_pct, 0) / nYrs;
    const wins = simRows.filter(r => r.diff > 0).length;
    el.textContent = `${nYrs}-Yr Backtest (${totalDrafts.toLocaleString()} drafts, ${perYr}/yr): ${avgDiff >= 0 ? "+" : ""}${avgDiff.toFixed(1)} FPTS avg | ${avgWin.toFixed(1)}% Win | ${wins}/${nYrs} winning yrs (Greedy — Pure Board)`;
  }
})();

selectSeason(APP.default_season);
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Build interactive multi-season draft board & replay HTML.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--scoring", default="default")
    parser.add_argument("--min-train-season", default="2016-17", help="earliest training season (default: 2016-17; use 2014-15 or 2008-09 for expanded archive)")
    parser.add_argument("--cv-folds", type=int, default=6, help="number of CV seasons to include (default: 6; use 10 with --min-train-season 2008-09 for all 13 seasons 2014-15..2026-27)")
    parser.add_argument("--recency-halflife", type=float, default=None, help="optional exponential recency half-life in seasons")
    parser.add_argument("--sim-csv", type=Path, default=None, help="optional sim_by_season CSV override matching this training window")
    parser.add_argument("--out", type=Path, default=Path("board_2026_27.html"), help="output HTML file path")
    args = parser.parse_args()

    print(f"Building multi-season strict out-of-sample replay payload (min_train={args.min_train_season}, cv_folds={args.cv_folds}, hl={args.recency_halflife})...")
    payload = build_all_seasons_payload(
        args.data,
        scoring_name=args.scoring,
        min_train_season=args.min_train_season,
        cv_folds=args.cv_folds,
        recency_halflife=args.recency_halflife,
        sim_csv_override=args.sim_csv,
    )
    json_blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__PAYLOAD_JSON__", json_blob)
    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out.resolve()} ({len(html):,} bytes, {len(payload['season_order'])} seasons)")


if __name__ == "__main__":
    main()
