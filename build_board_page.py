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


def build_live_2026_27_s2_cands(data, prior_s, prev2_s, births_prior, target_profs, prior_profs, played_before, adp_entries, early_data, vegas_odds=None, preseason_rosters=None):
    target_season = "2026-27"
    target_year = 2026
    prior_year = 2025
    prior_name = "2025-26"
    rt_map = de.compute_season_roster_turnover(target_season, prior_s, preseason_rosters)

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
            row = de.make_vet_feature_row(aid, prior_s, prev2_s, births_prior, target_season, prior_name, roster_turnover=rt_map)
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


def simulate_live_points_season(folder: Path, scoring: dict, s2_cands: list, n_teams: int = 14, n_rounds: int = 13, trials_per_slot: int = 52, adp_var: float = 2.0, max_centers: int = 4, seed: int = 42):
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
    preseason_rosters = de.load_preseason_rosters(data)

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
                vegas_odds=vegas_odds, preseason_rosters=preseason_rosters,
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
        mkt_150_pr = np.array([float(r["model_proj"]) for r in mkt_top150], dtype=np.float64)
        flip_mask = np.triu(np.ones((len(mkt_top150), len(mkt_top150)), dtype=bool), k=1) & (mkt_150_pr[None, :] > mkt_150_pr[:, None])
        flips_total = int(flip_mask.sum())
        ts_flips_pct = []

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

            act_arr = np.array([adj_by_id.get(aid, waiver_s) for aid in mkt_150_ids], dtype=np.float64)
            flips_won = int((flip_mask & (act_arr[None, :] > act_arr[:, None])).sum())
            ts_flips_pct.append(round(100.0 * flips_won / flips_total, 1) if flips_total else 50.0)

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

        seasons_payload[target_s] = {
            "season": target_s,
            "split": split,
            "max_step": tl["max_step"],
            "prior_waiver": round(prior_s["waiver_fppg"], 2),
            "peer_center": round(float(peer_center), 2),
            "checkpoints": tl["checkpoints"],
            "ts": {
                "t15_fpts": ts_t15_fpts_diff,
                "t15_vs": ts_t15_vs_diff,
                "top150_fpts": ts_top150_fpts_diff,
                "corr_r": ts_corr_r,
                "flips_pct": ts_flips_pct,
            },
            "logs": logs,
            "s2_all": ser_rows(s2_ordered),     # full 225 Stage-2 pool (top 150 + 151-225 sleepers + dropped)
            "s1_150": ser_rows(s1_top150),      # Stage-1 Pure Stats Top 150
        }
        season_order.append({"season": target_s, "split": split, "max_step": tl["max_step"]})
        print(
            f"  Built {target_s} ({split:7s}, {tl['max_step']:2d} steps) | "
            f"End Top150 FPTS diff: {ts_top150_fpts_diff[-1]:+8.1f} | "
            f"End Top15 Tgt-Fade FPTS: {ts_t15_fpts_diff[-1]:+8.1f} | End Calls Right={ts_flips_pct[-1]:.1f}%"
        )

    return {
        "anchor_ranks": ANCHOR_RANKS,
        "season_order": season_order,
        "default_season": season_order[-1]["season"],
        "seasons": seasons_payload,
    }


from board_templates import POINTS_HTML_TEMPLATE as HTML_TEMPLATE


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
