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

    player_daily = defaultdict(dict)  # aid -> di -> (gp, ndarray(11))
    seen_dates = set()
    for gid, aid, _name, stats, _pos, _starter, _tid in de.player_box_scores(folder, game_dates):
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


def simulate_live_cat9_season(folder: Path, cands: list, w9: np.ndarray, n_teams: int = 12, n_rounds: int = 13, trials_per_slot: int = 60, adp_var: float = 2.0, max_centers: int = 4, seed: int = 42):
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

    raw_daily = defaultdict(dict)  # aid -> gd -> ndarray(11)
    p_tot = defaultdict(lambda: np.zeros(11, dtype=np.float64))
    p_g = defaultdict(int)
    seen_dates = set()
    for gid, aid, _name, stats, _pos, _starter, _tid in de.player_box_scores(folder, game_dates):
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

    # Load season-by-season multi-format sim results if available
    sim_by_season = {}
    tag = f"to{int(round(to_weight * 100))}"
    sim_csv = Path(sim_csv_override) if sim_csv_override else (data / f"cat9_engine_sim_by_season_{tag}.csv")
    if sim_csv.is_file():
        for row in de.read_csv(sim_csv):
            if row.get("strategy") in ("residual_unified", "residual_wait"):
                sim_by_season[(row["season"], row["strategy"])] = {
                    "roto_pts": float(row["roto_pts"]),
                    "roto_rk": float(row["roto_rk"]),
                    "roto_win": float(row["roto_win_pct"]),
                    "roto_t4": float(row["roto_top4_pct"]),
                    "h2he_rk": float(row["h2he_rk"]),
                    "h2he_win": float(row["h2he_win_pct"]),
                    "h2hm_rk": float(row["h2hm_rk"]),
                    "h2hm_win": float(row["h2hm_win_pct"]),
                    "drafts": int(row["drafts"]) if row.get("drafts") else 360,
                }

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

        sim_live = simulate_live_cat9_season(data / target_s, cands, w9) if is_live else None

        seasons_payload[target_s] = {
            "season": target_s,
            "split": split,
            "max_step": tl["max_step"],
            "peer_center": round(float(peer_center), 2),
            "sim_greedy": sim_by_season.get((target_s, "residual_unified")),
            "sim_wait": sim_by_season.get((target_s, "residual_wait")),
            "sim_live": sim_live,
            "checkpoints": tl["checkpoints"],
            "ts": {
                "top150_z": ts_top150_z,
                "t15_vs": ts_t15_vs_z,
                "corr_r": ts_corr_r,
            },
            "gp": gp_series,
            "zs": z_series,
            "rows": rows_ser,
        }
        season_order.append({"season": target_s, "split": split, "max_step": tl["max_step"]})
        print(
            f"  Built {target_s} ({split:7s}, {tl['max_step']:2d} steps) | "
            f"End Top150 Z diff: {ts_top150_z[-1]:+7.2f} | "
            f"End Top15 Tgt-Fade Z: {ts_t15_vs_z[-1]:+7.2f} | End r={ts_corr_r[-1]:+.3f}"
        )

    return {
        "to_weight": float(to_weight),
        "cat_names": list(c9.CAT_NAMES),
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
<title>ADP-Based Pre-Draft Projections (9-Cat)</title>
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
  .wrap { max-width: 1360px; margin: 0 auto; }

  header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 16px; }
  .title-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  h1 { margin: 0; font-size: 24px; font-weight: 800; letter-spacing: -0.025em; color: var(--ink); }
  .split-badge { font-size: 11px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; padding: 3px 9px; border-radius: 999px; border: 1px solid var(--border); background: var(--card); color: var(--muted); }
  .split-badge.live { background: var(--pos-bg); border-color: var(--pos-bd); color: var(--pos); }
  .split-badge.holdout { background: var(--amber-bg); border-color: var(--amber-bd); color: var(--amber); }
  .split-badge.cv { background: var(--accent-bg); border-color: var(--accent-bd); color: var(--accent); }
  .cat9-pill { font-size: 11px; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase; padding: 3px 10px; border-radius: 999px; background: #0f172a; color: #fff; }
  .header-meta { font-size: 13px; color: var(--muted); background: var(--card); border: 1px solid var(--border); border-radius: 999px; padding: 5px 13px; box-shadow: var(--shadow); }
  .header-meta strong { color: var(--ink); }

  .intro-box { background: var(--card); border: 1px solid var(--accent-bd); border-left: 4px solid var(--accent); border-radius: 12px; padding: 13px 16px; margin-bottom: 14px; box-shadow: var(--shadow); }
  .intro-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
  .intro-title { font-size: 13.5px; font-weight: 700; color: var(--ink); }
  .intro-title span { color: var(--accent); margin-right: 6px; }
  .intro-toggle { font-size: 11.5px; font-weight: 600; color: var(--muted); background: var(--card-soft); border: 1px solid var(--border); border-radius: 6px; padding: 3px 9px; cursor: pointer; }
  .intro-toggle:hover { color: var(--accent); border-color: var(--accent-bd); }
  .intro-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 14px 20px; margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border); }
  .intro-box.collapsed .intro-grid { display: none; }
  .intro-step-title { font-size: 12.5px; font-weight: 700; color: var(--ink); margin-bottom: 3px; }
  .intro-step-body { font-size: 12.5px; color: var(--ink-2); line-height: 1.45; }

  .sim-hero {
    background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%);
    border: 1px solid var(--pos-bd);
    border-left: 4px solid var(--pos);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 14px;
    box-shadow: var(--shadow);
    display: grid;
    grid-template-columns: 1.35fr 1fr 1.2fr;
    gap: 16px;
    align-items: center;
  }
  @media (max-width: 900px) { .sim-hero { grid-template-columns: 1fr; } }
  .sim-hero-lbl { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin-bottom: 4px; }
  .sim-hero-big { display: flex; align-items: baseline; gap: 9px; flex-wrap: wrap; }
  .sim-hero-num { font-size: 30px; font-weight: 800; letter-spacing: -0.03em; color: var(--pos); line-height: 1.05; font-variant-numeric: tabular-nums; }
  .sim-hero-mult { font-size: 12px; font-weight: 700; color: var(--pos); background: var(--pos-bg); border: 1px solid var(--pos-bd); padding: 2px 8px; border-radius: 999px; }
  .sim-hero-sub { font-size: 12px; color: var(--ink-2); margin-top: 5px; font-variant-numeric: tabular-nums; }
  .sim-bars { margin-top: 7px; display: flex; flex-direction: column; gap: 4px; }
  .sim-bar-row { display: flex; align-items: center; gap: 8px; font-size: 11px; color: var(--muted); font-variant-numeric: tabular-nums; }
  .sim-bar-lbl { min-width: 78px; font-weight: 600; }
  .sim-bar-track { flex: 1; height: 7px; background: #e2e8f0; border-radius: 999px; overflow: hidden; }
  .sim-bar-fill { height: 100%; border-radius: 999px; }
  .sim-bar-fill.base { background: #94a3b8; }
  .sim-bar-fill.model { background: var(--pos); }
  .sim-bar-fill.h2h { background: var(--accent); }
  .sim-hero-div { border-left: 1px solid var(--pos-bd); padding-left: 16px; }
  @media (max-width: 900px) { .sim-hero-div { border-left: none; padding-left: 0; border-top: 1px solid var(--pos-bd); padding-top: 12px; } }
  .sim-hero-season-num { font-size: 22px; font-weight: 800; color: var(--ink); line-height: 1.15; font-variant-numeric: tabular-nums; }
  .sim-hero-note { font-size: 12.5px; color: var(--ink-2); line-height: 1.45; }

  .season-bar { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 10px 14px; margin-bottom: 12px; box-shadow: var(--shadow); display: flex; flex-wrap: wrap; align-items: center; gap: 12px; }
  .season-group { display: flex; flex-wrap: wrap; align-items: center; gap: 5px; }
  .season-group-lbl { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin-right: 3px; }
  .season-btn { font-size: 12px; font-weight: 600; background: var(--card-soft); border: 1px solid var(--border); color: var(--ink-2); padding: 5px 10px; border-radius: 8px; cursor: pointer; transition: all 0.12s ease; }
  .season-btn:hover { background: var(--accent-bg); border-color: var(--accent-bd); color: var(--accent); }
  .season-btn.holdout { border-color: var(--amber-bd); color: var(--amber); background: var(--amber-bg); }
  .season-btn.live { border-color: var(--pos-bd); color: var(--pos); background: var(--pos-bg); }
  .season-btn.active { background: var(--ink); border-color: var(--ink); color: #ffffff; }
  .season-sim-note { margin-left: auto; font-size: 12.5px; color: var(--muted); font-weight: 500; }
  .season-sim-note strong { color: var(--pos); font-weight: 700; }

  /* Interactive Punt Bar */
  .punt-bar { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 10px 14px; margin-bottom: 14px; box-shadow: var(--shadow); display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
  .punt-lbl { font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin-right: 4px; }
  .punt-chip { font-size: 12px; font-weight: 700; background: var(--pos-bg); border: 1px solid var(--pos-bd); color: var(--pos); padding: 4px 10px; border-radius: 8px; cursor: pointer; transition: all 0.12s ease; }
  .punt-chip:hover { opacity: 0.85; }
  .punt-chip.punted { background: var(--neg-bg); border-color: var(--neg-bd); color: var(--neg); text-decoration: line-through; }
  .punt-reset { font-size: 11.5px; font-weight: 600; background: var(--card-soft); border: 1px solid var(--border); color: var(--muted); padding: 4px 9px; border-radius: 7px; cursor: pointer; }
  .punt-reset:hover { color: var(--ink); border-color: var(--border-strong); }
  .punt-hint { margin-left: auto; font-size: 12px; color: var(--muted); }
  .punt-hint strong { color: var(--ink-2); }

  .replay-panel { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; margin-bottom: 14px; box-shadow: var(--shadow); }
  .replay-top { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 10px; }
  .replay-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
  .play-btn { font-size: 12.5px; font-weight: 700; background: var(--pos); border: 1px solid var(--pos); color: #fff; padding: 6px 14px; border-radius: 8px; cursor: pointer; min-width: 108px; }
  .play-btn.playing { background: var(--orange); border-color: var(--orange); }
  .mini-btn { font-size: 12px; font-weight: 600; background: var(--card-soft); border: 1px solid var(--border); color: var(--ink-2); padding: 5px 10px; border-radius: 8px; cursor: pointer; }
  .mini-btn:hover { background: var(--row-hover); border-color: var(--accent-bd); color: var(--accent); }
  .mini-btn.active { background: var(--accent-bg); border-color: var(--accent-bd); color: var(--accent); }
  .replay-clock { font-size: 13px; font-weight: 600; color: var(--ink); font-variant-numeric: tabular-nums; }
  .replay-clock span { color: var(--muted); font-weight: 500; }
  .slider-row { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
  .slider-end { font-size: 11.5px; font-weight: 600; color: var(--muted); min-width: 34px; font-variant-numeric: tabular-nums; }
  .slider-end.right { text-align: right; }
  input[type="range"]#game-slider { flex: 1; accent-color: var(--accent); cursor: pointer; height: 6px; }
  .chart-header { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 8px; }
  .chart-title { font-size: 12px; font-weight: 600; color: var(--muted); }
  .chart-modes { display: flex; gap: 5px; flex-wrap: wrap; }
  #edge-chart { width: 100%; height: 112px; display: block; background: #fafcff; border: 1px solid var(--border); border-radius: 8px; cursor: crosshair; }

  .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; margin-bottom: 14px; }
  .stat-box { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; box-shadow: var(--shadow); }
  .stat-label { font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); margin-bottom: 6px; }
  .stat-val { font-size: 22px; font-weight: 800; letter-spacing: -0.02em; line-height: 1.18; color: var(--ink); font-variant-numeric: tabular-nums; }
  .stat-sub { font-size: 12px; color: var(--muted); margin-top: 5px; }
  .stat-sub strong { color: var(--ink-2); font-weight: 600; }

  .controls { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px; margin-bottom: 10px; }
  .tabs { display: flex; flex-wrap: wrap; gap: 6px; }
  .tab-btn { font-size: 13px; font-weight: 600; background: var(--card); border: 1px solid var(--border); color: var(--ink-2); padding: 7px 13px; border-radius: 9px; cursor: pointer; box-shadow: var(--shadow); }
  .tab-btn:hover { border-color: var(--accent-bd); color: var(--accent); }
  .tab-btn.active { background: var(--accent); border-color: var(--accent); color: #fff; }
  .filters { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
  .filters input[type="text"] { font-size: 13px; padding: 6px 11px; border: 1px solid var(--border-strong); border-radius: 8px; background: var(--card); color: var(--ink); width: 165px; outline: none; }
  .filters input[type="text"]:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-bg); }
  .pill { font-size: 12px; font-weight: 600; background: var(--card); border: 1px solid var(--border); color: var(--ink-2); padding: 6px 10px; border-radius: 8px; cursor: pointer; }
  .pill:hover { background: var(--row-hover); color: var(--accent); }
  .pill.active { background: var(--accent-bg); border-color: var(--accent-bd); color: var(--accent); }

  .table-meta { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; font-size: 12.5px; color: var(--muted); margin-bottom: 8px; padding: 0 2px; font-variant-numeric: tabular-nums; }
  .table-wrap { overflow-x: auto; background: var(--card); border: 1px solid var(--border); border-radius: 12px; box-shadow: var(--shadow); }
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; font-variant-numeric: tabular-nums; }
  thead th { position: sticky; top: 0; background: #f8fafc; border-bottom: 1px solid var(--border-strong); padding: 9px 7px; text-align: right; font-size: 10.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); cursor: pointer; user-select: none; white-space: nowrap; }
  thead th.left { text-align: left; padding-left: 10px; }
  thead th.group-sep, tbody td.group-sep { border-left: 1px solid var(--border); }
  thead th.punted-col, tbody td.punted-col { opacity: 0.35; text-decoration: line-through; }
  thead th:hover { background: #f1f5f9; color: var(--ink); }
  tbody tr { border-bottom: 1px solid #f1f5f9; }
  tbody tr:nth-child(even) { background: var(--row-alt); }
  tbody tr:hover { background: var(--row-hover); }
  tbody td { padding: 7px 7px; text-align: right; white-space: nowrap; color: var(--ink-2); }
  tbody td.left { text-align: left; padding-left: 10px; }
  td.player-cell { font-size: 13px; font-weight: 600; color: var(--ink); }
  .tag { display: inline-block; font-size: 9.5px; font-weight: 700; padding: 1px 5px; border-radius: 999px; border: 1px solid var(--border); color: var(--muted); margin-left: 5px; vertical-align: 1px; }
  .tag-rook { border-color: var(--accent-bd); color: var(--accent); background: var(--accent-bg); }
  .tag-ret { border-color: #fed7aa; color: var(--orange); background: #fff7ed; }
  .tag-inj { border-color: var(--amber-bd); color: var(--amber); background: var(--amber-bg); }
  .pos { color: var(--pos); font-weight: 700; }
  .neg { color: var(--neg); font-weight: 700; }
  .muted { color: var(--muted); }
  .edge-pill { display: inline-block; padding: 2px 6px; border-radius: 6px; min-width: 46px; text-align: right; font-weight: 700; }
  .edge-pos { background: var(--pos-bg); color: var(--pos); }
  .edge-neg { background: var(--neg-bg); color: var(--neg); }
  .edge-flat { color: var(--muted); font-weight: 500; }
  .cat-cell { font-size: 11.5px; font-weight: 600; border-radius: 4px; padding: 2px 4px; display: inline-block; min-width: 40px; text-align: right; }
  .cat-hi { background: #d1fae5; color: #065f46; }
  .cat-midpos { background: #ecfdf5; color: #047857; }
  .cat-lo { background: #ffe4e6; color: #9f1239; }
  .cat-midneg { background: #fff1f2; color: #be123c; }
  .call-badge { display: inline-block; font-size: 10.5px; padding: 2px 7px; border-radius: 999px; border: 1px solid transparent; font-weight: 700; min-width: 58px; text-align: center; }
  .call-hit { color: var(--pos); background: var(--pos-bg); border-color: var(--pos-bd); }
  .call-miss { color: var(--neg); background: var(--neg-bg); border-color: var(--neg-bd); }
  .call-agree { color: var(--muted); background: var(--card-soft); border-color: var(--border); font-weight: 600; }

  .call-legend { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 8px 13px; margin-bottom: 10px; font-size: 12px; color: var(--ink-2); box-shadow: var(--shadow); }
  .call-legend-lbl { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }
  .call-legend-item { display: inline-flex; align-items: center; gap: 5px; flex-wrap: wrap; }
  .call-legend-item .call-badge { min-width: auto; padding: 1px 6px; font-size: 10px; }

  footer { margin-top: 18px; font-size: 12px; color: var(--muted); display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="title-row">
      <h1 id="page-title">2026&ndash;27 ADP-Based Pre-Draft Projections</h1>
      <span class="cat9-pill">9-CAT (ROTO + H2H)</span>
      <span class="split-badge live" id="split-badge">LIVE</span>
    </div>
    <div class="header-meta" id="header-meta"></div>
  </header>

  <section class="sim-hero" id="sim-hero"></section>

  <section class="intro-box collapsed" id="intro-box">
    <div class="intro-head">
      <div class="intro-title"><span>&#9679;</span>Why This Exists, What It Does &amp; What It Doesn&rsquo;t Do</div>
      <button class="intro-toggle" id="intro-toggle">Show</button>
    </div>
    <div class="intro-grid">
      <div>
        <div class="intro-step-title">Why This Exists (And Why Beating ADP Is Possible)</div>
        <div class="intro-step-body">Fantasy leagues are actually <strong>well projected in general</strong>&mdash;consensus ADP gets most players roughly right, and throwing it away to guess from scratch loses badly. The edge comes from repeatable <strong>human error and biases on a small group of players</strong>: overpaying for rookie hype, aging/injury-prone names, and flashy scoring while overlooking durable, efficient multi-category starters. Avoiding a few landmines and hitting a few undervalued targets is what builds league-winning teams.</div>
      </div>
      <div>
        <div class="intro-step-title">What This Board Does</div>
        <div class="intro-step-body">Publishes <strong>balanced, ADP-based pre-draft 9-Cat projections</strong> (Z-scores across PTS, REB, AST, STL, BLK, 3PM, volume-weighted FG%/FT%, and 0.75&times; TO) that power both Roto and H2H. Optional <strong>Punt chips</strong> let punters re-rank as a reference lens alongside ADP&mdash;your draft-day tactics are up to you.</div>
      </div>
      <div>
        <div class="intro-step-title">What It Doesn&rsquo;t Do</div>
        <div class="intro-step-body">It does <strong>not</strong> consider mid-season moves, waiver streaming, trades, mid-season injuries, or coaching/rotation changes after Opening Night, and it is not an explicit draft script. This is purely a way to get an edge on draft day&mdash;the rest of the season is up to the manager.</div>
      </div>
      <div>
        <div class="intro-step-title">How to Read Z-Scores &amp; Replay</div>
        <div class="intro-step-body">Each category is shown in standard deviations (<strong>Z-score</strong>) vs. the Top-180 pool: green (<code>+1.0</code>+) is an elite strength, red (<code>-1.0</code>) is a drag. Green <strong style="color:var(--pos)">Edge</strong> marks players projected ahead of ADP; scrub the slider below to replay any season game by game.</div>
      </div>
    </div>
  </section>

  <section class="season-bar">
    <div class="season-group">
      <span class="season-group-lbl">CV</span>
      <span id="btns-cv"></span>
    </div>
    <div class="season-group">
      <span class="season-group-lbl">Holdout</span>
      <span id="btns-holdout"></span>
    </div>
    <div class="season-group">
      <span class="season-group-lbl">Live</span>
      <span id="btns-live"></span>
    </div>
    <div class="season-sim-note" id="season-sim-note"></div>
  </section>

  <section class="punt-bar">
    <span class="punt-lbl">Punt Filter:</span>
    <span id="punt-chips"></span>
    <button class="punt-reset" id="punt-reset">Reset All (Balanced)</button>
    <span class="punt-hint" id="punt-hint">Click any category to punt it &amp; re-rank instantly</span>
  </section>

  <section class="replay-panel">
    <div class="replay-top">
      <div class="replay-controls">
        <button class="play-btn" id="play-btn">&#9654; Play 4x</button>
        <button class="mini-btn active" data-speed="250">4 g/s</button>
        <button class="mini-btn" data-speed="125">8 g/s</button>
        <button class="mini-btn" id="btn-start">G1</button>
        <button class="mini-btn" id="btn-end">Final</button>
      </div>
      <div class="replay-clock" id="replay-clock"></div>
    </div>
    <div class="slider-row">
      <span class="slider-end">G1</span>
      <input type="range" id="game-slider" min="0" max="15" value="15">
      <span class="slider-end right" id="slider-max-lbl">G16</span>
    </div>
    <div class="chart-header">
      <div class="chart-title" id="chart-title">9-Cat Edge Trajectory &mdash; click chart to jump to any game</div>
      <div class="chart-modes">
        <button class="mini-btn active" data-chart="t15_vs">Top-15 Targets vs. Fades (Z)</button>
        <button class="mini-btn" data-chart="top150_z">Top-150 Swaps (Z)</button>
        <button class="mini-btn" data-chart="corr_r">Correlation r</button>
      </div>
    </div>
    <svg id="edge-chart" viewBox="0 0 1000 112" preserveAspectRatio="none"></svg>
  </section>

  <section class="stats-grid">
    <div class="stat-box" title="Net 9-Cat Z-score gained by the ~15-20 borderline players our model swapped into the Top 150 over Market ADP.">
      <div class="stat-label">Top-150 Cutoff Swaps</div>
      <div class="stat-val" id="kpi-top150">&mdash;</div>
      <div class="stat-sub" id="kpi-top150-sub">&mdash;</div>
    </div>
    <div class="stat-box" title="Actual 9-Cat Z-score vs. ADP of our Top-15 highest-edge Targets minus our Top-15 steepest Fades.">
      <div class="stat-label">Top-15 Targets vs. Fades</div>
      <div class="stat-val" id="kpi-t15">&mdash;</div>
      <div class="stat-sub" id="kpi-t15-sub">&mdash;</div>
    </div>
    <div class="stat-box" title="Out-of-sample Pearson correlation between pre-draft predicted 9-Cat edge and actual Z-score vs. ADP across the Top 150.">
      <div class="stat-label">Edge Correlation (r)</div>
      <div class="stat-val" id="kpi-corr">&mdash;</div>
      <div class="stat-sub" id="kpi-corr-sub">&mdash;</div>
    </div>
    <div class="stat-box" title="12-team x 13-round draft-and-hold simulator blindly drafting the highest-ranked available player (Pure Greedy) across Roto, H2H Each-Cat, and H2H Most-Cat.">
      <div class="stat-label" id="kpi-sim-lbl">Draft-Only Sim Win% (720 dr)</div>
      <div class="stat-val" id="kpi-sim">&mdash;</div>
      <div class="stat-sub" id="kpi-sim-sub">&mdash;</div>
    </div>
  </section>

  <div class="controls">
    <div class="tabs">
      <button class="tab-btn active" data-tab="top150">Top-150 Board</button>
      <button class="tab-btn" data-tab="targets">Top Targets</button>
      <button class="tab-btn" data-tab="fades">Top Fades</button>
      <button class="tab-btn" data-tab="sleepers">Deep Sleepers (151&ndash;225)</button>
      <button class="tab-btn" data-tab="all">All 225</button>
    </div>
    <div class="filters">
      <input type="text" id="search" placeholder="Search player / pos...">
      <button class="pill active" id="cat-mode-btn" title="Switch the 9 category columns between pre-draft Projected Z and end-of-season Actual Z">Cats: Projected</button>
      <button class="pill" data-filter="hits">Hits</button>
      <button class="pill" data-filter="misses">Misses</button>
      <button class="pill" data-filter="rookies">Rookies</button>
    </div>
  </div>

  <div class="call-legend" aria-label="Call badge legend">
    <span class="call-legend-lbl">Call Key:</span>
    <span class="call-legend-item"><span class="call-badge call-hit">TGT &#10003;</span><span class="call-badge call-miss">TGT &#10007;</span> <strong>Target</strong> (projected above ADP; &#10003; beat ADP Z, &#10007; fell short)</span>
    <span class="call-legend-item"><span class="call-badge call-hit">FADE &#10003;</span><span class="call-badge call-miss">FADE &#10007;</span> <strong>Fade</strong> (projected below ADP; &#10003; underperformed ADP Z, &#10007; beat ADP Z)</span>
    <span class="call-legend-item"><span class="call-badge call-agree">AGREE</span> <strong>Agree</strong> (matches ADP within |Edge| &lt; 0.30 &amp; |&Delta;Rk| &lt; 3)</span>
  </div>

  <div class="table-meta">
    <div id="row-count">Showing 150 players</div>
    <div id="table-note">Click any header to sort &middot; Balanced TO weight: 0.75&times;</div>
  </div>

  <div class="table-wrap">
    <table id="board-table">
      <thead><tr id="thead-row"></tr></thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>

  <footer>
    <div>Zero-leakage expanding-window 9-Cat engine &middot; Volume-weighted FG%/FT% &middot; 12T daily active lineups</div>
    <div id="footer-sim-summary"></div>
  </footer>
</div>

<script>
const APP = __PAYLOAD_JSON__;
const CATS = APP.cat_names;
const BASE_W = [1, 1, 1, 1, 1, 1, 1, 1, APP.to_weight];

let state = {
  season: APP.default_season,
  step: 0,
  tab: "top150",
  filter: "all",
  search: "",
  sortKey: "dyn_rk",
  sortDir: 1,
  chartMode: "t15_vs",
  speedMs: 250,
  playing: false,
  timer: null,
  punted: new Set(),
  catView: "proj", // "proj" or "act"
};

function fmtSigned(v, dec = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return "&mdash;";
  return (v >= 0 ? "+" : "") + v.toFixed(dec);
}

function interpAnchor(anchors, adp) {
  const rk = APP.anchor_ranks;
  if (!anchors || !anchors.length) return 0;
  if (adp <= rk[0]) return anchors[0];
  if (adp >= rk[rk.length - 1]) return anchors[anchors.length - 1];
  for (let i = 0; i < rk.length - 1; i++) {
    if (adp >= rk[i] && adp <= rk[i + 1]) {
      const f = (adp - rk[i]) / (rk[i + 1] - rk[i]);
      return anchors[i] + f * (anchors[i + 1] - anchors[i]);
    }
  }
  return anchors[anchors.length - 1];
}

function activeWeights() {
  return BASE_W.map((w, i) => state.punted.has(i) ? 0 : w);
}

function buildSeasonButtons() {
  const groups = { cv: document.getElementById("btns-cv"), holdout: document.getElementById("btns-holdout"), live: document.getElementById("btns-live") };
  Object.values(groups).forEach(el => el.innerHTML = "");
  for (const item of APP.season_order) {
    const btn = document.createElement("button");
    btn.className = "season-btn " + item.split + (item.season === state.season ? " active" : "");
    btn.textContent = item.season;
    btn.onclick = () => selectSeason(item.season);
    groups[item.split].appendChild(btn);
  }
}

function buildPuntChips() {
  const wrap = document.getElementById("punt-chips");
  wrap.innerHTML = "";
  CATS.forEach((name, idx) => {
    const btn = document.createElement("button");
    btn.className = "punt-chip" + (state.punted.has(idx) ? " punted" : "");
    btn.textContent = name;
    btn.onclick = () => {
      if (state.punted.has(idx)) state.punted.delete(idx);
      else if (state.punted.size < 4) state.punted.add(idx);
      buildPuntChips();
      renderTable();
    };
    wrap.appendChild(btn);
  });
  const hint = document.getElementById("punt-hint");
  if (state.punted.size === 0) {
    hint.innerHTML = "Active: <strong>Balanced 9-Cat</strong> (click any category to punt &amp; re-rank)";
  } else {
    const names = Array.from(state.punted).sort().map(i => CATS[i]).join(", ");
    hint.innerHTML = `Punting: <strong style="color:var(--neg)">${names}</strong> (rankings &amp; edges updated live)`;
  }
}

function stopPlay() {
  state.playing = false;
  if (state.timer) { clearInterval(state.timer); state.timer = null; }
  const pb = document.getElementById("play-btn");
  pb.textContent = state.speedMs === 125 ? "\\u25B6 Play 8x" : "\\u25B6 Play 4x";
  pb.classList.remove("playing");
}

function togglePlay() {
  if (state.playing) { stopPlay(); return; }
  const S = APP.seasons[state.season];
  if (state.step >= S.max_step - 1) state.step = 0;
  state.playing = true;
  const pb = document.getElementById("play-btn");
  pb.textContent = "\\u23F8 Pause";
  pb.classList.add("playing");
  renderStep();
  state.timer = setInterval(() => {
    const curS = APP.seasons[state.season];
    if (state.step >= curS.max_step - 1) { stopPlay(); return; }
    state.step += 1;
    renderStep();
  }, state.speedMs);
}

function renderSimHero() {
  const S = APP.seasons[state.season];
  const allRows = APP.season_order.map(o => APP.seasons[o.season]?.sim_greedy).filter(Boolean);
  const el = document.getElementById("sim-hero");
  if (!el || !allRows.length) return;

  const nYrs = allRows.length;
  const totalDr = allRows.reduce((a, r) => a + (r.drafts || 720), 0);
  const perYr = Math.round(totalDr / nYrs);
  const avgRotoW = allRows.reduce((a, r) => a + r.roto_win, 0) / nYrs;
  const avgH2heW = allRows.reduce((a, r) => a + r.h2he_win, 0) / nYrs;
  const avgH2hmW = allRows.reduce((a, r) => a + r.h2hm_win, 0) / nYrs;
  const avgRotoRk = allRows.reduce((a, r) => a + r.roto_rk, 0) / nYrs;
  const mult = avgRotoW / (100 / 12);
  const maxBar = Math.max(38, avgRotoW * 1.15);

  const cur = S.sim_greedy;
  const live = S.sim_live;
  let seasonHtml = "";
  if (cur) {
    seasonHtml = `
      <div class="sim-hero-lbl">Selected Season (${S.season} &middot; ${cur.drafts || 720} drafts)</div>
      <div class="sim-hero-season-num pos">${cur.roto_win.toFixed(1)}% Roto &middot; ${cur.h2hm_win.toFixed(1)}% H2H-M</div>
      <div class="sim-hero-sub">H2H-Each <strong>${cur.h2he_win.toFixed(1)}%</strong> &middot; Roto <strong>${cur.roto_pts.toFixed(1)}/108</strong> (Avg Rk <strong>#${cur.roto_rk.toFixed(2)}</strong>)</div>
    `;
  } else if (live) {
    const cls = live.roto_diff >= 0 ? "pos" : "neg";
    seasonHtml = `
      <div class="sim-hero-lbl">Selected Season (${S.season} &middot; Live Draft vs. ADP)</div>
      <div class="sim-hero-season-num ${cls}">${fmtSigned(live.roto_diff, 1)} Roto Pts</div>
      <div class="sim-hero-sub">Avg Projected Draft <strong>${live.hero_roto.toFixed(1)}</strong> vs. Avg ADP Draft <strong>${live.field_roto.toFixed(1)}</strong> / 108 &middot; ${live.drafts} draft-only leagues (through G${S.max_step}, early sample)</div>
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
        <span class="sim-hero-num">${avgRotoW.toFixed(1)}%</span>
        <span class="sim-hero-mult">${mult.toFixed(1)}&times; baseline</span>
      </div>
      <div class="sim-bars">
        <div class="sim-bar-row"><span class="sim-bar-lbl">ADP (1/12)</span><div class="sim-bar-track"><div class="sim-bar-fill base" style="width:${(8.33 / maxBar * 100).toFixed(1)}%"></div></div><span>8.3%</span></div>
        <div class="sim-bar-row"><span class="sim-bar-lbl">9-Cat Roto</span><div class="sim-bar-track"><div class="sim-bar-fill model" style="width:${(avgRotoW / maxBar * 100).toFixed(1)}%"></div></div><strong>${avgRotoW.toFixed(1)}%</strong></div>
        <div class="sim-bar-row"><span class="sim-bar-lbl">H2H Each</span><div class="sim-bar-track"><div class="sim-bar-fill h2h" style="width:${(avgH2heW / maxBar * 100).toFixed(1)}%"></div></div><strong>${avgH2heW.toFixed(1)}%</strong></div>
        <div class="sim-bar-row"><span class="sim-bar-lbl">H2H Most</span><div class="sim-bar-track"><div class="sim-bar-fill h2h" style="width:${(avgH2hmW / maxBar * 100).toFixed(1)}%"></div></div><strong>${avgH2hmW.toFixed(1)}%</strong></div>
      </div>
      <div class="sim-hero-sub">Avg Roto Rank <strong>#${avgRotoRk.toFixed(2)}</strong> &middot; <strong>${nYrs}/${nYrs}</strong> winning seasons across all 3 formats</div>
    </div>
    <div class="sim-hero-div">${seasonHtml}</div>
    <div class="sim-hero-div">
      <div class="sim-hero-lbl">What Simulated Win Rate Means</div>
      <div class="sim-hero-note">In a fair 12-team league, everyone starts with an <strong>8.3% (1 in 12)</strong> chance to finish 1st. We simulate <strong>${perYr} draft-and-hold leagues/yr</strong> where 11 teams draft by ADP and 1 team blindly picks the highest-ranked available player here&mdash;<strong>zero waiting, punting, trades, or mid-season moves</strong>.</div>
    </div>
  `;
}

function selectSeason(seasonKey) {
  stopPlay();
  state.season = seasonKey;
  const S = APP.seasons[seasonKey];
  state.step = S.max_step - 1;
  buildSeasonButtons();

  document.getElementById("page-title").textContent = `${seasonKey} ADP-Based Pre-Draft Projections`;
  const badge = document.getElementById("split-badge");
  badge.className = "split-badge " + S.split;
  badge.textContent = S.split.toUpperCase();

  const slider = document.getElementById("game-slider");
  slider.min = 0;
  slider.max = S.max_step - 1;
  slider.value = state.step;
  document.getElementById("slider-max-lbl").textContent = `G${S.max_step}`;

  const simNote = document.getElementById("season-sim-note");
  const simG = S.sim_greedy;
  const simL = S.sim_live;
  if (simG) {
    simNote.innerHTML = `<span title="720 draft-and-hold 12-team leagues blindly picking the top-ranked available player (baseline 12-team win rate = 8.3%; no trades or mid-season moves).">Draft-Only Sim (${simG.drafts} dr): <strong>${simG.roto_pts.toFixed(1)}</strong> Roto (${simG.roto_win.toFixed(1)}% Win) &middot; H2H-M <strong>${simG.h2hm_win.toFixed(1)}%</strong> Win <span class="muted">(vs 8.3% base)</span></span>`;
  } else if (simL) {
    simNote.innerHTML = `<span title="Early-season draft edge: average team drafted from this board vs. average ADP-drafted team across ${simL.drafts} draft-and-hold leagues so far (excluded from 8-yr completed-season win rate).">Live Draft vs. ADP (${simL.drafts} dr): <strong>${fmtSigned(simL.roto_diff, 1)} Roto Pts</strong> <span class="muted">(${simL.hero_roto.toFixed(1)} vs ${simL.field_roto.toFixed(1)} / 108)</span></span>`;
  } else {
    simNote.innerHTML = `Live Out-of-Sample Season (Zero Leakage)`;
  }

  renderSimHero();
  renderStep();
}

function drawChart() {
  const S = APP.seasons[state.season];
  const svg = document.getElementById("edge-chart");
  const arr = S.ts[state.chartMode] || [];
  const n = arr.length;
  if (!n) { svg.innerHTML = ""; return; }

  const W = 1000, H = 112, padL = 12, padR = 12, padT = 12, padB = 14;
  const minV = Math.min(0, ...arr);
  const maxV = Math.max(0, ...arr);
  const span = Math.max(maxV - minV, 0.05);
  const xOf = i => n === 1 ? W / 2 : padL + (i / (n - 1)) * (W - padL - padR);
  const yOf = v => padT + (1 - (v - minV) / span) * (H - padT - padB);
  const y0 = yOf(0);

  let pts = arr.map((v, i) => `${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`).join(" ");
  let area = `${xOf(0).toFixed(1)},${y0.toFixed(1)} ` + pts + ` ${xOf(n - 1).toFixed(1)},${y0.toFixed(1)}`;
  const curV = arr[state.step];
  const cx = xOf(state.step), cy = yOf(curV);
  const col = curV >= 0 ? "#059669" : "#e11d48";
  const fill = curV >= 0 ? "rgba(5,150,105,0.12)" : "rgba(225,29,72,0.12)";

  svg.innerHTML = `
    <line x1="${padL}" y1="${y0.toFixed(1)}" x2="${W - padR}" y2="${y0.toFixed(1)}" stroke="#cbd5e1" stroke-dasharray="4,4" stroke-width="1"/>
    <polygon points="${area}" fill="${fill}"/>
    <polyline points="${pts}" fill="none" stroke="${col}" stroke-width="2.4" stroke-linejoin="round"/>
    <line x1="${cx.toFixed(1)}" y1="${padT}" x2="${cx.toFixed(1)}" y2="${H - padB}" stroke="#2563eb" stroke-width="1.4" stroke-dasharray="3,3"/>
    <circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="4.5" fill="${col}" stroke="#fff" stroke-width="1.6"/>
  `;
}

function renderKPIs() {
  const S = APP.seasons[state.season];
  const cp = S.checkpoints[state.step];
  const top150Z = S.ts.top150_z[state.step];
  const t15Vs = S.ts.t15_vs[state.step];
  const rVal = S.ts.corr_r[state.step];

  document.getElementById("header-meta").innerHTML =
    `Through <strong>${cp.date}</strong> &middot; <strong>${cp.reg.toLocaleString()}</strong> NBA games (${cp.pct.toFixed(0)}%)`;
  document.getElementById("replay-clock").innerHTML =
    `Game ${cp.g} / ${S.max_step} <span>(${cp.date} &middot; Top-150 cut ${fmtSigned(cp.t150_cut)} Z)</span>`;
  document.getElementById("game-slider").value = state.step;

  const k1 = document.getElementById("kpi-top150");
  k1.innerHTML = `${fmtSigned(top150Z, 1)} Z`;
  k1.className = "stat-val " + (top150Z >= 0 ? "pos" : "neg");
  // Count Top-150 hits at current step
  let modHits = 0, mktHits = 0;
  for (const r of S.rows) {
    const az = S.zs[r.id][state.step];
    const hit = az >= cp.t150_cut;
    if (r.rk <= 150 && hit) modHits++;
    if (r.mrk <= 150 && hit) mktHits++;
  }
  document.getElementById("kpi-top150-sub").innerHTML = `Top-150 hits: <strong>${modHits}/150</strong> vs. ADP ${mktHits}/150`;

  const k2 = document.getElementById("kpi-t15");
  k2.innerHTML = `${fmtSigned(t15Vs, 1)} Z`;
  k2.className = "stat-val " + (t15Vs >= 0 ? "pos" : "neg");
  document.getElementById("kpi-t15-sub").innerHTML = `Top-15 Targets minus Top-15 Fades (vs. ADP)`;

  const k3 = document.getElementById("kpi-corr");
  k3.innerHTML = fmtSigned(rVal, 3);
  k3.className = "stat-val " + (rVal >= 0 ? "pos" : "neg");
  document.getElementById("kpi-corr-sub").innerHTML = `Top-150 pre-draft edge vs. actual 9-Cat Z`;

  const k4 = document.getElementById("kpi-sim");
  const k4s = document.getElementById("kpi-sim-sub");
  const simG = S.sim_greedy;
  const simL = S.sim_live;
  if (simG) {
    document.getElementById("kpi-sim-lbl").textContent = `Draft-Only Sim Win% (${simG.drafts} dr)`;
    k4.innerHTML = `${simG.roto_win.toFixed(1)}% / ${simG.h2he_win.toFixed(1)}% / ${simG.h2hm_win.toFixed(1)}%`;
    k4.className = "stat-val pos";
    k4s.innerHTML = `Roto / H2H-E / H2H-M (vs <strong>8.3%</strong> 12T base &middot; Rk ${simG.roto_rk.toFixed(2)})`;
  } else if (simL) {
    document.getElementById("kpi-sim-lbl").textContent = `Live Draft vs. ADP (${simL.drafts} dr)`;
    k4.innerHTML = `${fmtSigned(simL.roto_diff, 1)} Roto Pts`;
    k4.className = "stat-val " + (simL.roto_diff >= 0 ? "pos" : "neg");
    k4s.innerHTML = `Avg Draft <strong>${simL.hero_roto.toFixed(1)}</strong> vs. ADP <strong>${simL.field_roto.toFixed(1)}</strong> / 108 (early sample)`;
  } else {
    const allSims = APP.season_order.map(o => APP.seasons[o.season]?.sim_greedy).filter(Boolean);
    const nS = allSims.length || 1;
    const aR = allSims.reduce((a, r) => a + r.roto_win, 0) / nS;
    const aE = allSims.reduce((a, r) => a + r.h2he_win, 0) / nS;
    const aM = allSims.reduce((a, r) => a + r.h2hm_win, 0) / nS;
    document.getElementById("kpi-sim-lbl").textContent = "Draft-Only Sim Win% (Live)";
    k4.innerHTML = "In Progress";
    k4.className = "stat-val muted";
    k4s.innerHTML = `${nS}-Yr Avg (vs 8.3% base): <strong>${aR.toFixed(1)}%</strong> Roto &middot; <strong>${aE.toFixed(1)}%</strong> H2H-E &middot; <strong>${aM.toFixed(1)}%</strong> H2H-M`;
  }
}

function catClass(v) {
  if (v >= 1.2) return "cat-cell cat-hi";
  if (v >= 0.35) return "cat-cell cat-midpos";
  if (v <= -1.2) return "cat-cell cat-lo";
  if (v <= -0.35) return "cat-cell cat-midneg";
  return "cat-cell muted";
}

function buildRowsWithPunt() {
  const S = APP.seasons[state.season];
  const cp = S.checkpoints[state.step];
  const w = activeWeights();
  const isPunted = state.punted.size > 0;

  // Recompute projected Z under active punt weights
  const enriched = S.rows.map(r => {
    let projZ = r.pr;
    if (isPunted) {
      projZ = 0;
      for (let c = 0; c < 9; c++) projZ += r.pz[c] * w[c];
    }
    const actG = S.gp[r.id][state.step];
    const actZ = S.zs[r.id][state.step];
    const slotZ = interpAnchor(cp.anchors, r.adp);
    const vsAdp = actZ - slotZ;
    return { ...r, dyn_pr: projZ, act_g: actG, act_z: actZ, vs_adp: vsAdp };
  });

  // Assign dynamic rank & peer-centered edge
  const sorted = [...enriched].sort((a, b) => (b.dyn_pr - a.dyn_pr) || (a.mrk - b.mrk));
  const diffs150 = sorted.filter(r => r.mrk <= 150).map(r => r.dyn_pr - r.mp).sort((a, b) => a - b);
  const medDiff = diffs150.length ? diffs150[Math.floor(diffs150.length / 2)] : 0;

  sorted.forEach((r, i) => {
    r.dyn_rk = i + 1;
    r.dyn_drk = r.mrk - r.dyn_rk;
    r.dyn_e = isPunted ? (r.dyn_pr - r.mp - medDiff) : r.e;
    const bigEdge = Math.abs(r.dyn_e) >= 0.30;
    const bigShift = Math.abs(r.dyn_drk) >= 3;
    if (!bigEdge && !bigShift) {
      r.call_label = "AGREE";
      r.call_cls = "call-agree";
      r.call_score = 0;
    } else {
      const liked = r.dyn_e !== 0 ? (r.dyn_e > 0) : (r.dyn_drk > 0);
      const hit = liked ? (r.vs_adp > 0) : (r.vs_adp < 0);
      if (liked) {
        r.call_label = hit ? "TGT \u2713" : "TGT \u2717";
      } else {
        r.call_label = hit ? "FADE \u2713" : "FADE \u2717";
      }
      r.call_cls = hit ? "call-hit" : "call-miss";
      r.call_score = hit ? 1 : -1;
    }
  });
  return sorted;
}

function renderTable() {
  const rowsAll = buildRowsWithPunt();
  const isActCat = state.catView === "act";

  // Build header
  const cols = [
    { k: "dyn_rk", l: "Rk" },
    { k: "p", l: "Player", left: true },
    { k: "pos", l: "Pos", left: true },
    { k: "adp", l: "ADP" },
    { k: "dyn_drk", l: "&Delta;Rk" },
    { k: "dyn_pr", l: "Proj Z" },
    { k: "dyn_e", l: "Edge" },
    ...CATS.map((c, idx) => ({ k: `c_${idx}`, l: c, sep: idx === 0, catIdx: idx })),
    { k: "act_g", l: "G", sep: true },
    { k: "act_z", l: "Act Z" },
    { k: "vs_adp", l: "vs ADP" },
    { k: "call_score", l: "Call" },
  ];
  const thead = document.getElementById("thead-row");
  thead.innerHTML = cols.map(c => {
    const cls = [c.left ? "left" : "", c.sep ? "group-sep" : "", (c.catIdx !== undefined && state.punted.has(c.catIdx)) ? "punted-col" : ""].filter(Boolean).join(" ");
    const arr = state.sortKey === c.k ? (state.sortDir === 1 ? " &#9650;" : " &#9660;") : "";
    return `<th class="${cls}" data-key="${c.k}">${c.l}<span class="sort-arrow">${arr}</span></th>`;
  }).join("");
  thead.querySelectorAll("th").forEach(th => {
    th.onclick = () => {
      const k = th.dataset.key;
      if (state.sortKey === k) state.sortDir *= -1;
      else { state.sortKey = k; state.sortDir = (k === "dyn_rk" || k === "adp" || k === "p" || k === "pos") ? 1 : -1; }
      renderTable();
    };
  });

  // Tab filter
  let view = rowsAll;
  if (state.tab === "top150") view = rowsAll.filter(r => r.dyn_rk <= 150);
  else if (state.tab === "targets") view = rowsAll.filter(r => r.mrk <= 180 && r.dyn_e > 0).sort((a, b) => b.dyn_e - a.dyn_e).slice(0, 40);
  else if (state.tab === "fades") view = rowsAll.filter(r => r.mrk <= 150 && r.dyn_e < 0).sort((a, b) => a.dyn_e - b.dyn_e).slice(0, 40);
  else if (state.tab === "sleepers") view = rowsAll.filter(r => r.mrk > 150);

  if (state.filter === "hits") view = view.filter(r => r.call_score === 1);
  else if (state.filter === "misses") view = view.filter(r => r.call_score === -1);
  else if (state.filter === "rookies") view = view.filter(r => r.t === "ROOK");

  if (state.search.trim()) {
    const q = state.search.trim().toLowerCase();
    view = view.filter(r => r.p.toLowerCase().includes(q) || r.pos.toLowerCase().includes(q));
  }

  // Sort
  const sk = state.sortKey, sd = state.sortDir;
  view = [...view].sort((a, b) => {
    let va, vb;
    if (sk.startsWith("c_")) {
      const ci = parseInt(sk.slice(2), 10);
      va = (isActCat ? a.az : a.pz)[ci];
      vb = (isActCat ? b.az : b.pz)[ci];
    } else {
      va = a[sk]; vb = b[sk];
    }
    if (typeof va === "string") return sd * va.localeCompare(vb);
    return sd * ((va ?? 0) - (vb ?? 0));
  });

  document.getElementById("row-count").textContent = `Showing ${view.length} players (${isActCat ? "Actual" : "Projected"} category Z-scores)`;

  const tbody = document.getElementById("tbody");
  tbody.innerHTML = view.map(r => {
    let tags = "";
    if (r.t === "ROOK") tags += `<span class="tag tag-rook">ROOK</span>`;
    else if (r.t === "RET") tags += `<span class="tag tag-ret">RET</span>`;
    if (r.al) tags += `<span class="tag tag-inj">INJ?</span>`;
    const drkCls = r.dyn_drk >= 5 ? "pos" : (r.dyn_drk <= -5 ? "neg" : "muted");
    const eCls = r.dyn_e >= 0.30 ? "edge-pill edge-pos" : (r.dyn_e <= -0.30 ? "edge-pill edge-neg" : "edge-pill edge-flat");
    const vsCls = r.vs_adp >= 0.5 ? "pos" : (r.vs_adp <= -0.5 ? "neg" : "muted");
    const callCls = `call-badge ${r.call_cls}`;
    const cArr = isActCat ? r.az : r.pz;
    const catTds = cArr.map((v, ci) => {
      const cls = [ci === 0 ? "group-sep" : "", state.punted.has(ci) ? "punted-col" : ""].filter(Boolean).join(" ");
      return `<td class="${cls}"><span class="${catClass(v)}">${fmtSigned(v, 1)}</span></td>`;
    }).join("");
    return `<tr>
      <td><strong>${r.dyn_rk}</strong></td>
      <td class="left player-cell">${r.p}${tags}</td>
      <td class="left muted">${r.pos}</td>
      <td>${r.adp.toFixed(1)}</td>
      <td class="${drkCls}">${r.dyn_drk >= 0 ? "+" : ""}${r.dyn_drk}</td>
      <td><strong>${fmtSigned(r.dyn_pr, 2)}</strong></td>
      <td><span class="${eCls}">${fmtSigned(r.dyn_e, 2)}</span></td>
      ${catTds}
      <td class="group-sep muted">${r.act_g}</td>
      <td>${fmtSigned(r.act_z, 2)}</td>
      <td class="${vsCls}">${fmtSigned(r.vs_adp, 2)}</td>
      <td><span class="${callCls}">${r.call_label}</span></td>
    </tr>`;
  }).join("");
}

function renderStep() {
  renderKPIs();
  drawChart();
  renderTable();
}

// Wire controls
document.getElementById("intro-toggle").onclick = () => {
  const box = document.getElementById("intro-box");
  box.classList.toggle("collapsed");
  document.getElementById("intro-toggle").textContent = box.classList.contains("collapsed") ? "Show" : "Hide";
};
document.getElementById("play-btn").onclick = togglePlay;
document.getElementById("btn-start").onclick = () => { stopPlay(); state.step = 0; renderStep(); };
document.getElementById("btn-end").onclick = () => { stopPlay(); state.step = APP.seasons[state.season].max_step - 1; renderStep(); };
document.getElementById("game-slider").oninput = e => { stopPlay(); state.step = parseInt(e.target.value, 10); renderStep(); };
document.getElementById("punt-reset").onclick = () => { state.punted.clear(); buildPuntChips(); renderTable(); };
document.getElementById("cat-mode-btn").onclick = () => {
  state.catView = state.catView === "proj" ? "act" : "proj";
  document.getElementById("cat-mode-btn").textContent = state.catView === "proj" ? "Cats: Projected" : "Cats: Actual";
  renderTable();
};
document.querySelectorAll("[data-speed]").forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll("[data-speed]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.speedMs = parseInt(btn.dataset.speed, 10);
    if (state.playing) { stopPlay(); togglePlay(); }
    else document.getElementById("play-btn").textContent = state.speedMs === 125 ? "\\u25B6 Play 8x" : "\\u25B6 Play 4x";
  };
});
document.querySelectorAll("[data-chart]").forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll("[data-chart]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.chartMode = btn.dataset.chart;
    drawChart();
  };
});
document.getElementById("edge-chart").onclick = e => {
  const S = APP.seasons[state.season];
  const rect = e.currentTarget.getBoundingClientRect();
  const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  stopPlay();
  state.step = Math.round(frac * (S.max_step - 1));
  renderSimHero();
  renderStep();
};
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.tab = btn.dataset.tab;
    state.sortKey = state.tab === "targets" ? "dyn_e" : (state.tab === "fades" ? "dyn_e" : "dyn_rk");
    state.sortDir = state.tab === "targets" ? -1 : 1;
    renderTable();
  };
});
document.querySelectorAll("[data-filter]").forEach(btn => {
  btn.onclick = () => {
    const f = btn.dataset.filter;
    if (state.filter === f) { state.filter = "all"; btn.classList.remove("active"); }
    else {
      document.querySelectorAll("[data-filter]").forEach(b => b.classList.remove("active"));
      state.filter = f; btn.classList.add("active");
    }
    renderTable();
  };
});
document.getElementById("search").oninput = e => { state.search = e.target.value; renderTable(); };

// Dynamic footer summary from embedded sim rows (Pure Greedy)
(() => {
  const simRows = APP.season_order.map(o => APP.seasons[o.season]?.sim_greedy).filter(Boolean);
  const el = document.getElementById("footer-sim-summary");
  if (el && simRows.length) {
    const n = simRows.length;
    const dr = simRows.reduce((a, r) => a + r.drafts, 0);
    const rW = simRows.reduce((a, r) => a + r.roto_win, 0) / n;
    const eW = simRows.reduce((a, r) => a + r.h2he_win, 0) / n;
    const mW = simRows.reduce((a, r) => a + r.h2hm_win, 0) / n;
    el.textContent = `${n}-Yr 9-Cat Backtest (${dr.toLocaleString()} drafts): ${rW.toFixed(1)}% Roto | ${eW.toFixed(1)}% H2H-E | ${mW.toFixed(1)}% H2H-M Win (Greedy — Pure Board)`;
  }
})();

buildPuntChips();
selectSeason(APP.default_season);
</script>
</body>
</html>
"""


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
