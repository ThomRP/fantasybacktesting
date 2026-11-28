"""Generate a clean, bright, self-contained HTML 9-Cat Draft Board & Multi-Season Replay Dashboard (`board_cat9_2026_27.html`).

Covers ALL out-of-sample seasons (strict zero-leakage expanding-window training):
  - 6 CV Folds (2018-19 .. 2023-24)
  - 2 Untouched Holdout Seasons (2024-25, 2025-26)
  - Live Out-of-Sample Season (2026-27 through latest games)

Features:
  - Season switcher (CV / Holdout / Live) with 3-format sim stats (Roto, H2H-Each, H2H-Most)
  - Interactive client-side Punt Toggle Chips (PTS, REB, AST, STL, BLK, 3PM, FG%, FT%, TO)
  - Real-time game-by-game 9-Cat Z-score replay (4x / 8x) + clickable SVG edge trajectory chart
  - Toggleable 9-category Z-score heatmap columns (Projected vs. Actual)

Usage:
  .\\.venv\\Scripts\\python.exe build_cat9_board_page.py
"""

import argparse
import json
import random
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import mean, median

import numpy as np

import cat9_engine as c9
import draft_engine as de


ANCHOR_RANKS = (
    list(range(1, 21))
    + list(range(25, 155, 5))
    + [160, 175, 200, 225, 250]
)


def _sample_anchors(lb_desc):
    n = len(lb_desc)
    if n == 0:
        return [0.0] * len(ANCHOR_RANKS)
    return [round(float(lb_desc[min(rk - 1, n - 1)]), 2) for rk in ANCHOR_RANKS]


def load_cat9_timeline(folder: Path, w9: np.ndarray, is_live: bool = False):
    """Build per-team-game checkpoints (1..max_step) and exact cumulative 9-Cat Z trajectories."""
    game_dates = {}
    games_per_date = defaultdict(int)
    for g in de.read_csv(folder / "games.csv"):
        if not de.regular_season_game(g):
            continue
        gid = g.get("game_id")
        gd = (g.get("game_date") or "")[:10]
        if not gid or not gd:
            continue
        if is_live:
            hs = de._safe_float(g.get("home_score"))
            as_ = de._safe_float(g.get("away_score"))
            if hs is None or as_ is None or (hs == 0 and as_ == 0):
                continue
        game_dates[gid] = gd
        games_per_date[gd] += 1

    dates_all = sorted(games_per_date)
    date_to_idx = {d: i for i, d in enumerate(dates_all)}

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
        step_events = [defaultdict(lambda: [0, np.zeros(11, dtype=np.float64)]) for _ in range(max_step)]
        step_dates = [""] * max_step
        for gid, aid, _name, stats, _pos, _starter, tid in raw_boxes:
            gnum = team_game_num.get((tid, gid))
            if gnum is None or gnum > min_tg:
                continue
            s_idx = gnum - 1
            vals = np.array([float(stats.get(k) or 0.0) for k in c9.STAT_KEYS_11], dtype=np.float64)
            slot = step_events[s_idx][aid]
            slot[0] += 1
            slot[1] += vals
            gd = game_dates[gid]
            if gd > step_dates[s_idx]:
                step_dates[s_idx] = gd
        cum_gp = defaultdict(int)
        cum_11 = defaultdict(lambda: np.zeros(11, dtype=np.float64))
        checkpoints = []
        step_z_by_id = []
        final_cat_z = {}
        final_zero_vec = [0.0] * 9
        for s_idx in range(max_step):
            for aid, (gp, vals) in step_events[s_idx].items():
                cum_gp[aid] += gp
                cum_11[aid] += vals
            avg_g = float(s_idx + 1)
            frac = min(1.0, avg_g / 82.0)
            min_qual = max(1, int(round(20.0 * frac)))
            zinfo = c9.compute_cat9_zscores(cum_11, cum_gp, top_pool_n=180, min_qual_games=min_qual)
            z_scores = {aid: float(vec @ w9) for aid, vec in zinfo["z_tot"].items()}
            zero_z = float(zinfo["zero_tot_vec"] @ w9)
            z_lb = sorted(z_scores.values(), reverse=True)
            t150_cut = z_lb[min(149, len(z_lb) - 1)] if z_lb else zero_z
            checkpoints.append({
                "g": s_idx + 1,
                "date": step_dates[s_idx],
                "reg": int(15 * (s_idx + 1)),
                "avg_g": round(avg_g, 1),
                "pct": round(100.0 * frac, 1),
                "zero_z": round(zero_z, 2),
                "t150_cut": round(t150_cut, 2),
                "anchors": _sample_anchors(z_lb),
            })
            step_z_by_id.append((z_scores, z_lb, zero_z, dict(cum_gp)))
            if s_idx == max_step - 1:
                final_cat_z = {aid: [round(float(x), 2) for x in vec] for aid, vec in zinfo["z_tot"].items()}
                final_zero_vec = [round(float(x), 2) for x in zinfo["zero_tot_vec"]]
        return {
            "max_step": max_step,
            "checkpoints": checkpoints,
            "step_z_by_id": step_z_by_id,
            "final_cat_z": final_cat_z,
            "final_zero_vec": final_zero_vec,
        }

    player_daily = defaultdict(dict)  # aid -> di -> (gp, ndarray(11))
    seen_dates = set()
    for gid, aid, _name, stats, _pos, _starter, _tid in raw_boxes:
        gd = game_dates[gid]
        di = date_to_idx[gd]
        vals = np.array([float(stats.get(k) or 0.0) for k in c9.STAT_KEYS_11], dtype=np.float64)
        prev = player_daily[aid].get(di)
        if prev is None:
            player_daily[aid][di] = [1, vals]
        else:
            prev[0] += 1
            prev[1] += vals
        seen_dates.add(gd)

    if is_live and seen_dates:
        dates_sorted = [d for d in dates_all if d in seen_dates]
        new_idx = {d: i for i, d in enumerate(dates_sorted)}
        remapped = defaultdict(dict)
        for aid, dmap in player_daily.items():
            for old_di, val in dmap.items():
                gd = dates_all[old_di]
                if gd in new_idx:
                    remapped[aid][new_idx[gd]] = val
        player_daily = remapped
    else:
        dates_sorted = dates_all

    cum_games = []
    running = 0
    for d in dates_sorted:
        running += games_per_date[d]
        cum_games.append(running)

    avg_team_g = [(2.0 * cg) / 30.0 for cg in cum_games]
    final_avg_g = avg_team_g[-1] if avg_team_g else 1.0
    max_step = max(1, int(round(final_avg_g)))

    step_date_idx = []
    for g_num in range(1, max_step + 1):
        if g_num == max_step:
            di = len(dates_sorted) - 1
        else:
            target = g_num - 0.15
            di = next((i for i, av in enumerate(avg_team_g) if av >= target), len(dates_sorted) - 1)
            if step_date_idx and di < step_date_idx[-1]:
                di = step_date_idx[-1]
        step_date_idx.append(di)

    di_to_step = [max_step - 1] * len(dates_sorted)
    prev_di = -1
    for s_idx, cut_di in enumerate(step_date_idx):
        for di in range(prev_di + 1, cut_di + 1):
            di_to_step[di] = s_idx
        prev_di = cut_di

    step_events = [defaultdict(lambda: [0, np.zeros(11, dtype=np.float64)]) for _ in range(max_step)]
    for aid, dmap in player_daily.items():
        for di, (gp, vals) in dmap.items():
            s_idx = di_to_step[di]
            slot = step_events[s_idx][aid]
            slot[0] += gp
            slot[1] += vals

    cum_gp = defaultdict(int)
    cum_11 = defaultdict(lambda: np.zeros(11, dtype=np.float64))
    checkpoints = []
    step_z_by_id = []
    final_cat_z = {}

    for s_idx in range(max_step):
        for aid, (gp, vals) in step_events[s_idx].items():
            cum_gp[aid] += gp
            cum_11[aid] += vals

        di = step_date_idx[s_idx]
        avg_g = avg_team_g[di]
        frac = min(1.0, avg_g / (82.0 if is_live else max(final_avg_g, 1.0)))
        min_qual = max(1, int(round(20.0 * frac)))

        zinfo = c9.compute_cat9_zscores(cum_11, cum_gp, top_pool_n=180, min_qual_games=min_qual)
        z_scores = {aid: float(vec @ w9) for aid, vec in zinfo["z_tot"].items()}
        zero_z = float(zinfo["zero_tot_vec"] @ w9)
        z_lb = sorted(z_scores.values(), reverse=True)
        t150_cut = z_lb[min(149, len(z_lb) - 1)] if z_lb else zero_z

        checkpoints.append({
            "g": s_idx + 1,
            "date": dates_sorted[di],
            "reg": int(cum_games[di]),
            "avg_g": round(avg_g, 1),
            "pct": round(100.0 * frac, 1),
            "zero_z": round(zero_z, 2),
            "t150_cut": round(t150_cut, 2),
            "anchors": _sample_anchors(z_lb),
        })
        step_z_by_id.append((z_scores, z_lb, zero_z, dict(cum_gp)))
        if s_idx == max_step - 1:
            final_cat_z = {aid: [round(float(x), 2) for x in vec] for aid, vec in zinfo["z_tot"].items()}
            final_zero_vec = [round(float(x), 2) for x in zinfo["zero_tot_vec"]]

    return {
        "max_step": max_step,
        "checkpoints": checkpoints,
        "step_z_by_id": step_z_by_id,
        "final_cat_z": final_cat_z,
        "final_zero_vec": final_zero_vec,
    }


def simulate_live_cat9_season(folder: Path, cands: list, w9: np.ndarray, n_teams: int = 14, n_rounds: int = 13, trials_per_slot: int = 52, adp_var: float = 2.0, max_centers: int = 4, seed: int = 42):
    """Run 720 Pure Greedy 12-team 9-Cat draft-and-hold leagues on the in-progress live season (Roto + H2H-E + H2H-M)."""
    if len(cands) < n_teams * n_rounds:
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

    raw_daily = defaultdict(dict)  # aid -> gd -> ndarray(11)
    p_tot = defaultdict(lambda: np.zeros(11, dtype=np.float64))
    p_g = defaultdict(int)
    seen_dates = set()
    for gid, aid, _name, stats, _pos, _starter, tid in raw_boxes:
        if team_game_num.get((tid, gid), 999) > min_tg:
            continue
        gd = game_dates[gid]
        vals = np.array([float(stats.get(k) or 0.0) for k in c9.STAT_KEYS_11], dtype=np.float64)
        prev = raw_daily[aid].get(gd)
        raw_daily[aid][gd] = vals if prev is None else (prev + vals)
        p_tot[aid] += vals
        p_g[aid] += 1
        seen_dates.add(gd)

    if not seen_dates:
        return None
    dates_sorted = sorted(seen_dates)
    date_to_idx = {d: i for i, d in enumerate(dates_sorted)}
    weeks_sorted = sorted({date.fromisoformat(d).isocalendar()[:2] for d in dates_sorted})
    wk_to_idx = {wk: i for i, wk in enumerate(weeks_sorted)}
    day_to_week = np.array([wk_to_idx[date.fromisoformat(d).isocalendar()[:2]] for d in dates_sorted], dtype=np.int32)
    n_days, n_wks = len(dates_sorted), len(weeks_sorted)
    valid_wks = list(range(n_wks))

    # Actual per-game Z so far for active lineup priority on crowded days
    z_live = c9.compute_cat9_zscores(dict(p_tot), dict(p_g), min_qual_games=3)

    players = sorted(cands, key=lambda r: r["market_rank"])
    n_p = len(players)
    played_mat = np.zeros((n_p, n_days), dtype=np.uint8)
    player_tot11 = np.zeros((n_p, 11), dtype=np.float64)
    player_wk11 = np.zeros((n_p, n_wks, 11), dtype=np.float64)
    daily_11 = []
    masks7, masks10, is_c, prio = [], [], [], []
    for idx, p in enumerate(players):
        ptup = p["positions"]
        m7, m10 = de._pos_to_masks(ptup)
        masks7.append(m7)
        masks10.append(m10)
        is_c.append(ptup[0] == "C")
        zr = z_live["z_raw"].get(p["athlete_id"])
        prio.append(float(zr @ w9) if zr is not None else -99.0)
        dmap_idx = {}
        for gd, v11 in raw_daily.get(p["athlete_id"], {}).items():
            di = date_to_idx[gd]
            dmap_idx[di] = v11
            played_mat[idx, di] = 1
            player_tot11[idx] += v11
            player_wk11[idx, int(day_to_week[di])] += v11
        daily_11.append(dmap_idx)

    bundle = {
        "played_mat": played_mat,
        "played_lists": played_mat.tolist(),
        "daily_11": daily_11,
        "player_tot11": player_tot11,
        "player_wk11": player_wk11,
        "day_to_week": day_to_week,
        "masks10": masks10,
        "prio": prio,
        "roster_cache": {},
    }

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
    roto_pts_l, field_roto_l, roto_diff_l = [], [], []

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
                choice, new_m = c9._pick_legal_from_board(board, taken, roster, cur_c, cur_m, rem_after, masks7, is_c, max_centers)
                taken[choice] = True
                roster.append(choice)
                if is_c[choice]:
                    c_cnt[team] += 1
                m_sz[team] = new_m

            team_tot11 = np.zeros((n_teams, 11), dtype=np.float64)
            team_wk11 = np.zeros((n_teams, n_wks, 11), dtype=np.float64)
            for t in range(n_teams):
                t11, w11, _bg = c9.score_roster_cat9_active(rosters[t], bundle)
                team_tot11[t] = t11
                team_wk11[t] = w11

            roto, _h2he, _h2hm = c9.score_cat9_league(team_tot11, team_wk11, valid_wks)
            field_roto = float(np.mean([roto[t] for t in range(n_teams) if t != slot]))
            roto_pts_l.append(float(roto[slot]))
            field_roto_l.append(field_roto)
            roto_diff_l.append(float(roto[slot]) - field_roto)

    return {
        "drafts": len(roto_pts_l),
        "weeks": n_wks,
        "min_games": min_tg,
        "last_date": dates_sorted[-1],
        "hero_roto": round(mean(roto_pts_l), 1),
        "field_roto": round(mean(field_roto_l), 1),
        "roto_diff": round(mean(roto_diff_l), 1),
    }


def build_all_cat9_seasons_payload(
    data: Path,
    *,
    to_weight: float = c9.DEFAULT_TO_WEIGHT,
    min_train_season: str = "2016-17",
    cv_folds: int = 6,
    recency_halflife=None,
    sim_csv_override=None,
):
    w9 = c9.make_cat_weights(to_weight=to_weight)
    seasons_dirs, season_data, births_cache, profiles_cache, cat9_by_year = c9.load_all_cat9_seasons(data)
    cat9_pools, lb_cache, played_before_all, history_available = c9.build_cat9_adp_pools(
        data, seasons_dirs, season_data, births_cache, profiles_cache, cat9_by_year, w9,
    )
    vegas_odds = de.load_vegas_preseason_odds(data)

    eff_min_s = None if (not min_train_season or str(min_train_season).lower() == "all") else str(min_train_season)
    train_seasons = sorted(s for s in cat9_pools if eff_min_s is None or s >= eff_min_s)
    eval_seasons = [s for s in train_seasons if len(cat9_pools[s]) >= 156]
    if len(eval_seasons) <= 2:
        eval_seasons = train_seasons
    development, holdout = eval_seasons[:-2], eval_seasons[-2:]
    cv_eligible = [s for s in development if sum(1 for ts in train_seasons if ts < s) >= 2]
    cv_seasons = cv_eligible[-cv_folds:] if cv_eligible else development[1:]
    all_targets = [(s, "cv") for s in cv_seasons] + [(s, "holdout") for s in holdout]
    if (data / "2026-27" / "adp.csv").is_file():
        all_targets.append(("2026-27", "live"))

    seasons_payload = {}
    season_order = []

    for target_s, split in all_targets:
        target_yr = int(target_s[:4])
        prior_yr = target_yr - 1
        is_live = (split == "live")

        if split == "holdout":
            cutoff = development[-1]
            tr_rows = [r for s, grp in cat9_pools.items() if (eff_min_s is None or s >= eff_min_s) and s <= cutoff for r in grp]
        else:
            tr_rows = [r for s, grp in cat9_pools.items() if (eff_min_s is None or s >= eff_min_s) and s < target_s for r in grp]

        if is_live:
            base_cands = de.build_eval_adp_pool(
                data, prior_yr, seasons_dirs, season_data, births_cache, profiles_cache,
                set(played_before_all), history_available, max_pool_adp=225.0, vegas_odds=vegas_odds,
            )
            pri_c9 = cat9_by_year[prior_yr]
            pri_tot_map, pri_raw_map, pri_lb, pri_raw_rk, pri_zero, pri_waiv_raw = lb_cache[prior_yr]
            p2_tot_map = lb_cache[prior_yr - 1][0] if (prior_yr - 1) in lb_cache else {}
            spread = max(1.0, c9.z_slot_value(pri_lb, 10.0) - c9.z_slot_value(pri_lb, 150.0))
            col_thresh = c9.COLLAPSE_SPREAD_RATIO * spread
            cands = []
            for r in base_cands:
                row = r.copy()
                aid = row["athlete_id"]
                adp = float(row["adp"])
                z_mkt = c9.z_slot_value(pri_lb, adp)
                pri_tot_vec = pri_c9["z_tot"].get(aid)
                pri_raw_vec = pri_c9["z_raw"].get(aid)
                pri_tot = pri_tot_map.get(aid)
                pri_raw = pri_raw_map.get(aid)
                p2_tot = p2_tot_map.get(aid)
                row["z_mkt"] = z_mkt
                row["z_zero_floor"] = pri_zero
                row["z_spread"] = spread
                row["z_waiver_raw"] = pri_waiv_raw
                row["z_prior_tot"] = pri_tot
                row["z_prior_raw"] = pri_raw
                row["z_prior_vec"] = pri_tot_vec.copy() if pri_tot_vec is not None else None
                row["z_prior_raw_vec"] = pri_raw_vec.copy() if pri_raw_vec is not None else None
                row["z_prior_fg_imp"] = float(pri_raw_vec[6]) if pri_raw_vec is not None else None
                row["z_prior_ft_imp"] = float(pri_raw_vec[7]) if pri_raw_vec is not None else None
                row["z_prior_to"] = float(pri_raw_vec[8]) if pri_raw_vec is not None else None
                row["z_yoy_delta"] = (pri_tot - p2_tot) if (pri_tot is not None and p2_tot is not None) else None
                if row["player_type"] == "returning_vet" and pri_tot is not None:
                    raw_gap = float(pri_tot) - z_mkt
                    p_rk = pri_raw_rk.get(pri_raw, 500) if pri_raw is not None else 500
                    slot_drop = adp - p_rk
                    is_col = slot_drop >= c9.COLLAPSE_MIN_SLOT_DROP and raw_gap > col_thresh
                    row["z_prior_minus_mkt_capped"] = max(-1.25 * col_thresh, min(col_thresh, raw_gap))
                    row["z_collapse_excess"] = (raw_gap - col_thresh) if is_col else 0.0
                else:
                    row["z_prior_minus_mkt_capped"] = None
                    row["z_collapse_excess"] = 0.0
                cands.append(row)
        else:
            cands = [r.copy() for r in cat9_pools[target_s]]

        res = c9.fit_predict_cat9_unified(tr_rows, cands, w9, lam=c9.DEFAULT_LAM, recency_halflife=recency_halflife)
        peer_center = median(res["scaled_edges"][i] for i, r in enumerate(cands) if r["market_rank"] <= 150)

        for i, r in enumerate(cands):
            r["model_proj"] = round(float(res["residual_proj"][i]), 2)
            r["mkt_proj"] = round(float(r["z_mkt"]), 2)
            r["raw_edge"] = round(float(res["raw_residuals"][i]), 2)
            r["edge_vs_peers"] = round(float(res["scaled_edges"][i] - peer_center), 2)
            r["proj_cat_z"] = [round(float(x), 2) for x in res["proj_cat_z"][i]]
            r["alert"] = "CHECK_INJURY" if r.get("z_collapse_excess", 0.0) > 0 else ""

        ordered = sorted(cands, key=lambda r: (-r["model_proj"], r["market_rank"]))
        for rk, r in enumerate(ordered, 1):
            r["model_rank"] = rk
            r["rank_delta"] = int(r["market_rank"] - rk)

        tl = load_cat9_timeline(data / target_s, w9, is_live=is_live)

        s2_top150 = ordered[:150]
        mkt_top150 = sorted([r for r in cands if r["market_rank"] <= 150], key=lambda r: r["market_rank"])
        t15 = sorted(mkt_top150, key=lambda r: -r["edge_vs_peers"])[:15]
        f15 = sorted(mkt_top150, key=lambda r: r["edge_vs_peers"])[:15]

        s2_150_ids = [r["athlete_id"] for r in s2_top150]
        mkt_150_ids = [r["athlete_id"] for r in mkt_top150]
        t15_ids = [r["athlete_id"] for r in t15]
        f15_ids = [r["athlete_id"] for r in f15]
        mkt_150_adps = [float(r["adp"]) for r in mkt_top150]
        mkt_150_re = [float(r["raw_edge"]) for r in mkt_top150]
        t15_adps = [float(r["adp"]) for r in t15]
        f15_adps = [float(r["adp"]) for r in f15]

        ts_top150_z = []
        ts_t15_vs_z = []
        ts_corr_r = []
        ts_flips_pct = []
        mkt_150_pr = np.array([float(r["model_proj"]) for r in mkt_top150], dtype=np.float64)
        flip_mask = np.triu(np.ones((len(mkt_top150), len(mkt_top150)), dtype=bool), k=1) & (mkt_150_pr[None, :] > mkt_150_pr[:, None])
        flips_total = int(flip_mask.sum())

        # Track compact per-player step series (gp, z_tot) for replay scrubbing
        all_ids = [r["athlete_id"] for r in ordered]
        gp_series = {aid: [] for aid in all_ids}
        z_series = {aid: [] for aid in all_ids}

        for s_idx in range(tl["max_step"]):
            z_scores, z_lb, zero_z, gp_map = tl["step_z_by_id"][s_idx]
            s2_z = sum(z_scores.get(aid, zero_z) for aid in s2_150_ids)
            mk_z = sum(z_scores.get(aid, zero_z) for aid in mkt_150_ids)
            ts_top150_z.append(round(s2_z - mk_z, 2))

            t15_vs = sum(z_scores.get(aid, zero_z) - c9.z_slot_value(z_lb, a) for aid, a in zip(t15_ids, t15_adps))
            f15_vs = sum(z_scores.get(aid, zero_z) - c9.z_slot_value(z_lb, a) for aid, a in zip(f15_ids, f15_adps))
            ts_t15_vs_z.append(round(t15_vs - f15_vs, 2))

            vs_vec = [z_scores.get(aid, zero_z) - c9.z_slot_value(z_lb, a) for aid, a in zip(mkt_150_ids, mkt_150_adps)]
            rc = float(np.corrcoef(mkt_150_re, vs_vec)[0, 1]) if (np.std(vs_vec) > 1e-6 and np.std(mkt_150_re) > 1e-6) else 0.0
            ts_corr_r.append(round(rc, 3))

            act_arr = np.array([z_scores.get(aid, zero_z) for aid in mkt_150_ids], dtype=np.float64)
            flips_won = int((flip_mask & (act_arr[None, :] > act_arr[:, None])).sum())
            ts_flips_pct.append(round(100.0 * flips_won / flips_total, 1) if flips_total else 50.0)

            for aid in all_ids:
                gp_series[aid].append(int(gp_map.get(aid, 0)))
                z_series[aid].append(round(z_scores.get(aid, zero_z), 2))

        type_short = {"returning_vet": "VET", "rookie": "ROOK", "returning_absence": "RET"}
        rows_ser = []
        for r in ordered:
            aid = r["athlete_id"]
            rows_ser.append({
                "id": aid,
                "rk": r["model_rank"],
                "p": r["player"],
                "pos": "/".join(r["positions"][:2]) if isinstance(r.get("positions"), (tuple, list)) else str(r.get("positions") or ""),
                "t": type_short.get(r["player_type"], "VET"),
                "mrk": r["market_rank"],
                "adp": round(float(r["adp"]), 1),
                "drk": r["rank_delta"],
                "mp": r["mkt_proj"],
                "e": r["edge_vs_peers"],
                "re": r["raw_edge"],
                "pr": r["model_proj"],
                "al": r["alert"],
                "pz": r["proj_cat_z"],
                "az": tl["final_cat_z"].get(aid, tl["final_zero_vec"]),
            })

        seasons_payload[target_s] = {
            "season": target_s,
            "split": split,
            "max_step": tl["max_step"],
            "peer_center": round(float(peer_center), 2),
            "checkpoints": tl["checkpoints"],
            "ts": {
                "top150_z": ts_top150_z,
                "t15_vs": ts_t15_vs_z,
                "corr_r": ts_corr_r,
                "flips_pct": ts_flips_pct,
            },
            "gp": gp_series,
            "zs": z_series,
            "rows": rows_ser,
        }
        season_order.append({"season": target_s, "split": split, "max_step": tl["max_step"]})
        print(
            f"  Built {target_s} ({split:7s}, {tl['max_step']:2d} steps) | "
            f"End Top150 Z diff: {ts_top150_z[-1]:+7.2f} | "
            f"End Top15 Tgt-Fade Z: {ts_t15_vs_z[-1]:+7.2f} | End Calls Right={ts_flips_pct[-1]:.1f}%"
        )

    return {
        "to_weight": float(to_weight),
        "cat_names": list(c9.CAT_NAMES),
        "anchor_ranks": ANCHOR_RANKS,
        "season_order": season_order,
        "default_season": season_order[-1]["season"],
        "seasons": seasons_payload,
    }


from board_templates import CAT9_HTML_TEMPLATE as HTML_TEMPLATE


def main():
    parser = argparse.ArgumentParser(description="Build interactive multi-season 9-Cat draft board & replay HTML.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--to-weight", type=float, default=c9.DEFAULT_TO_WEIGHT)
    parser.add_argument("--min-train-season", default="2016-17")
    parser.add_argument("--cv-folds", type=int, default=6)
    parser.add_argument("--recency-halflife", type=float, default=None)
    parser.add_argument("--sim-csv", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("board_cat9_2026_27.html"))
    args = parser.parse_args()

    print(f"Building multi-season 9-Cat strict out-of-sample replay payload (TO={args.to_weight:.2f}, min_train={args.min_train_season}, cv_folds={args.cv_folds})...")
    payload = build_all_cat9_seasons_payload(
        args.data,
        to_weight=args.to_weight,
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
