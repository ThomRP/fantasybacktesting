"""Generate the 2026-27 full-pool draft board and run a 'too early' comparison against in-progress games."""

import argparse
import csv
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np

from backtest_common import (
    PlayerSeason,
    adp_slot_points,
    fantasy_points,
    load_player_profiles,
    name_key,
    player_box_scores,
    read_csv,
    regular_season_game,
    write_csv,
)
from backtest_full_pool import (
    BOARD_SIZE,
    RESIDUAL_CLIP,
    STRATEGIES,
    WAIVER_FACTOR,
    _pos_to_masks,
    assign_lineup_slots,
    build_completed_dataset,
    build_global_board,
    build_unified_season_rows,
    fit_and_score_full_pool,
    load_season_cache,
    load_season_daily_logs,
    max_matching_7,
    parse_draft_info_from_profiles,
    score_roster_espn,
    season_label,
    tune_global_lambdas,
)
from backtest_rookies import FIRST_ARCHIVE_YEAR, players_with_appearances


def read_in_progress_season(folder):
    """Read completed regular-season games from an in-progress season (no playoff or All-Star game required)."""
    games_path = folder / "games.csv"
    box_path = folder / "player_box_scores.csv"
    if not games_path.is_file() or not box_path.is_file():
        return None

    schedule = list(read_csv(games_path))
    reg_games = [g for g in schedule if regular_season_game(g)]
    if not reg_games:
        return None

    game_dates = {g["game_id"]: g["game_date"] for g in reg_games}
    team_games = Counter()
    for g in reg_games:
        team_games[str(g["home_id"])] += 1
        team_games[str(g["away_id"])] += 1
    avg_team_games = mean(team_games.values()) if team_games else 0.0
    latest_date = max(game_dates.values())

    players = defaultdict(PlayerSeason)
    for gid, aid, name, stats, pos, starter, _ in player_box_scores(folder, game_dates):
        p = players[aid]
        p.names.add(name)
        p.pre_games += 1
        p.pre_points += fantasy_points(stats)
        if pos:
            p.positions[pos] += 1
        st = str(starter).strip().lower()
        if st == "true":
            p.starts += 1
        elif st != "false":
            p.games_without_start_status += 1

    if not players:
        return None

    return {
        "players": players,
        "reg_games_count": len(reg_games),
        "avg_team_games": avg_team_games,
        "latest_date": latest_date,
        "season_pct": min(100.0, 100.0 * avg_team_games / 82.0),
    }


def compute_scaled_adj_leaderboard(players, avg_team_games):
    """Scale availability-adjusted FPPG proportionally to games elapsed so far (ref = 65 * avg_team_games / 82)."""
    scale = min(1.0, max(avg_team_games / 82.0, 1.0 / 82.0))
    ref_games = max(1.0, 65.0 * scale)
    qual_min = max(1, int(round(20.0 * scale)))

    qual_raw = sorted(
        (p.total_points / p.total_games for p in players.values() if p.total_games >= qual_min),
        reverse=True,
    )
    if not qual_raw:
        qual_raw = sorted((p.total_points / p.total_games for p in players.values() if p.total_games > 0), reverse=True)
    repl_fppg = qual_raw[min(149, len(qual_raw) - 1)] if qual_raw else 20.0
    waiver_fppg = WAIVER_FACTOR * repl_fppg

    adj_by_id, raw_by_id = {}, {}
    for aid, p in players.items():
        g = p.total_games
        if g <= 0:
            adj_by_id[aid] = waiver_fppg
            raw_by_id[aid] = 0.0
        else:
            raw = p.total_points / g
            raw_by_id[aid] = raw
            eff_g = min(float(g), ref_games)
            adj_by_id[aid] = (eff_g * raw + (ref_games - eff_g) * waiver_fppg) / ref_games

    adj_lb = sorted(adj_by_id.values(), reverse=True)
    adj_rank_by_score = {}
    for place, score in enumerate(adj_lb, start=1):
        adj_rank_by_score.setdefault(score, place)

    raw_lb = sorted((v for v in raw_by_id.values() if v > 0), reverse=True)
    raw_rank_by_score = {}
    for place, score in enumerate(raw_lb, start=1):
        raw_rank_by_score.setdefault(score, place)

    return {
        "ref_games": ref_games,
        "qual_min": qual_min,
        "waiver_fppg": waiver_fppg,
        "adj_by_id": adj_by_id,
        "raw_by_id": raw_by_id,
        "adj_lb": adj_lb,
        "adj_rank_by_score": adj_rank_by_score,
        "raw_lb": raw_lb,
        "raw_rank_by_score": raw_rank_by_score,
    }


def resolve_early_returning_absences(data, prior_year, rows, played_before):
    """If player_core for target_season isn't published yet, rescue veterans who missed prior_year from earlier profiles."""
    try:
        prior_profiles = parse_draft_info_from_profiles(load_player_profiles(data, prior_year + 1))
    except Exception:
        return

    ids_by_name = defaultdict(set)
    for aid, prof in prior_profiles.items():
        for n in prof["names"]:
            ids_by_name[name_key(n)].add(aid)

    for r in rows:
        if r["player_type"] == "rookie" and str(r["athlete_id"]).startswith("unmatched_"):
            cands = ids_by_name.get(name_key(r["player"]), set())
            if len(cands) == 1:
                (aid,) = cands
                if aid in played_before:
                    r["athlete_id"] = aid
                    r["player_type"] = "returning_absence"


def run_early_espn_sim(data, target_season, rows, lambdas, *, n_teams=12, n_rounds=13, trials_per_slot=12, adp_var=2.0, max_centers=4, seed=None):
    """Run 12-team x 13-round ESPN snake draft sim scored on actual games played so far in target_season."""
    if seed is None:
        seed = random.SystemRandom().randint(1, 999_999)
    rng = random.Random(seed)

    players = sorted(rows, key=lambda r: (r["adp"], r["market_rank"], r["athlete_id"]))
    total_picks = n_teams * n_rounds
    if len(players) < total_picks:
        print(f"Skipping early ESPN sim: only {len(players)} players in {target_season} pool, need {total_picks}")
        return None

    s_folder = data / target_season
    needed_ids = {p["athlete_id"] for p in players}
    n_days, daily, _, fallback_pos = load_season_daily_logs(s_folder, needed_ids)

    n_p = len(players)
    played_mat = np.zeros((n_p, max(n_days, 1)), dtype=np.uint8)
    fpts_mat = np.zeros((n_p, max(n_days, 1)), dtype=np.int16)
    masks7, masks10, is_center, prio, tot_fpts = [], [], [], [], []
    for idx, p in enumerate(players):
        aid = p["athlete_id"]
        ptup = p.get("adp_positions") or fallback_pos.get(aid) or ("SG", "SF")
        m7, m10 = _pos_to_masks(ptup)
        p["pos_str"] = ",".join(ptup)
        masks7.append(m7)
        masks10.append(m10)
        is_center.append(ptup[0] == "C")
        prio.append(p["early_raw_fppg"] if p["early_games"] > 0 else -1.0)
        day_map = daily.get(aid, {})
        for d, pts in day_map.items():
            played_mat[idx, d] = 1
            fpts_mat[idx, d] = pts
        tot_fpts.append(int(sum(day_map.values())))

    bundle = {
        "has_daily": True,
        "played_mat": played_mat,
        "played_lists": played_mat.tolist(),
        "fpts_lists": fpts_mat.tolist(),
        "masks7": masks7,
        "masks10": masks10,
        "is_center": is_center,
        "prio": prio,
        "tot_fpts": tot_fpts,
        "roster_cache": {},
    }

    pick_order = [
        (rnd + 1, team)
        for rnd in range(n_rounds)
        for team in (range(n_teams - 1, -1, -1) if rnd % 2 == 1 else range(n_teams))
    ]
    straight_order = list(range(len(players)))
    hero_orders = {}
    for strat in STRATEGIES:
        if strat == "market":
            hero_orders[strat] = straight_order
        else:
            lam = lambdas[strat]
            scored = [
                (p["mkt_expected_adj_fppg"] + lam * max(-RESIDUAL_CLIP, min(RESIDUAL_CLIP, p[f"{strat}_score"])), -p["adp"], idx)
                for idx, p in enumerate(players)
            ]
            scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
            hero_orders[strat] = [idx for _, _, idx in scored]

    sample_slot = rng.randrange(n_teams)
    sample_trial = rng.randrange(trials_per_slot)
    sample_info = None
    records = defaultdict(list)

    for slot in range(n_teams):
        for trial in range(trials_per_slot):
            opp_noisy = []
            for _ in range(n_teams):
                noisy = [(players[i]["adp"] + rng.uniform(-adp_var, adp_var), i) for i in range(len(players))]
                noisy.sort(key=lambda t: (t[0], t[1]))
                opp_noisy.append([i for _, i in noisy])

            for strat in STRATEGIES:
                taken = [False] * len(players)
                rosters = [[] for _ in range(n_teams)]
                c_cnt = [0] * n_teams
                m_sz = [0] * n_teams
                hero_picks = []
                for pick_num, (rnd, team) in enumerate(pick_order, start=1):
                    board = hero_orders[strat] if team == slot else (straight_order if rnd == 1 else opp_noisy[team])
                    roster = rosters[team]
                    rem_after = n_rounds - len(roster) - 1
                    cur_c, cur_m = c_cnt[team], m_sz[team]
                    choice, new_m = None, cur_m
                    for idx in board:
                        if taken[idx] or (is_center[idx] and cur_c >= max_centers):
                            continue
                        if cur_m + rem_after >= 7:
                            choice = idx
                            new_m = max_matching_7(roster + [idx], masks7) if cur_m < 7 else 7
                            break
                        cm = max_matching_7(roster + [idx], masks7)
                        if cm + rem_after >= 7:
                            choice, new_m = idx, cm
                            break
                    if choice is None:
                        choice = next(
                            (idx for idx in board if not taken[idx] and not (is_center[idx] and cur_c >= max_centers)),
                            next(idx for idx in board if not taken[idx]),
                        )
                        new_m = max_matching_7(roster + [choice], masks7)
                    taken[choice] = True
                    roster.append(choice)
                    if is_center[choice]:
                        c_cnt[team] += 1
                    m_sz[team] = new_m
                    if team == slot:
                        hero_picks.append((rnd, pick_num, choice))

                afs, bgs = [], []
                for t in range(n_teams):
                    af, _, bg, _ = score_roster_espn(rosters[t], bundle)
                    afs.append(af)
                    bgs.append(bg)
                hero_af = afs[slot]
                field_afs = [afs[t] for t in range(n_teams) if t != slot]
                field_avg = sum(field_afs) / len(field_afs)
                better = sum(1 for x in field_afs if x > hero_af)
                tied = sum(1 for x in field_afs if x == hero_af)
                rank = 1.0 + better + 0.5 * tied
                rec = {
                    "hero_af": hero_af,
                    "field_af": field_avg,
                    "diff": hero_af - field_avg,
                    "rank": rank,
                    "won": (1.0 / (1 + tied)) if better == 0 else 0.0,
                    "top4": 1.0 if rank <= 4.0 else 0.0,
                    "beat_field": 1.0 if hero_af > field_avg else (0.5 if hero_af == field_avg else 0.0),
                    "hero_rk": sum(1 for i in rosters[slot] if players[i]["player_type"] == "rookie"),
                    "field_rk": sum(1 for t in range(n_teams) if t != slot for i in rosters[t] if players[i]["player_type"] == "rookie") / (n_teams - 1),
                    "hero_0g": sum(1 for i in rosters[slot] if players[i]["early_games"] == 0),
                    "field_0g": sum(1 for t in range(n_teams) if t != slot for i in rosters[t] if players[i]["early_games"] == 0) / (n_teams - 1),
                    "hero_bg": bgs[slot],
                    "hero_c": c_cnt[slot],
                }
                records[strat].append(rec)
                if slot == sample_slot and trial == sample_trial and strat == "unified_equal":
                    sample_info = (slot + 1, bundle, players, hero_orders[strat], hero_picks, rosters[slot], rec)

    print(f"\n=== TOO-EARLY 12-Team ESPN Snake Draft Sim on {target_season} Games So Far ({trials_per_slot * n_teams} drafts/strategy, Seed: {seed}) ===")
    print(
        f"  {'Strategy':20s} {'Lam':>4s} | {'Hero ESPN':>9s} {'Field ESPN':>10s} {'Diff':>8s} {'Lift%':>6s} | "
        f"{'Rank':>5s} {'Win%':>5s} {'Top4%':>6s} {'BeatFld':>7s} | {'Rk(H/F)':>9s} {'0g(H/F)':>9s}"
    )
    print("  " + "-" * 116)
    sim_rows = []
    for strat in STRATEGIES:
        recs = records[strat]
        h_avg = mean(r["hero_af"] for r in recs)
        f_avg = mean(r["field_af"] for r in recs)
        d = h_avg - f_avg
        lift = 100.0 * d / f_avg if f_avg else 0.0
        row = {
            "strategy": strat,
            "lambda": lambdas[strat],
            "drafts": len(recs),
            "hero_espn_fpts": round(h_avg, 1),
            "field_espn_fpts": round(f_avg, 1),
            "espn_diff": round(d, 1),
            "espn_lift_pct": round(lift, 2),
            "avg_rank": round(mean(r["rank"] for r in recs), 2),
            "win_pct": round(100.0 * mean(r["won"] for r in recs), 1),
            "top4_pct": round(100.0 * mean(r["top4"] for r in recs), 1),
            "beat_field_pct": round(100.0 * mean(r["beat_field"] for r in recs), 1),
            "hero_rk": round(mean(r["hero_rk"] for r in recs), 2),
            "field_rk": round(mean(r["field_rk"] for r in recs), 2),
            "hero_0g": round(mean(r["hero_0g"] for r in recs), 2),
            "field_0g": round(mean(r["field_0g"] for r in recs), 2),
        }
        sim_rows.append(row)
        print(
            f"  {strat:20s} {lambdas[strat]:4.2f} | "
            f"{row['hero_espn_fpts']:9.1f} {row['field_espn_fpts']:10.1f} {row['espn_diff']:+8.1f} {row['espn_lift_pct']:+5.2f}% | "
            f"{row['avg_rank']:5.2f} {row['win_pct']:4.1f}% {row['top4_pct']:5.1f}% {row['beat_field_pct']:6.1f}% | "
            f"{row['hero_rk']:4.2f}/{row['field_rk']:4.2f} {row['hero_0g']:4.2f}/{row['field_0g']:4.2f}"
        )

    if sample_info:
        slot_num, b, players_s, hero_order, picks, roster, rec = sample_info
        rank_of = {idx: r for r, idx in enumerate(hero_order, start=1)}
        slot_of = assign_lineup_slots(roster, b["masks10"], b["prio"])
        tag_map = {"returning_vet": "VET", "rookie": "ROOK", "returning_absence": "RET"}
        print(f"\n=== Sample {target_season} Too-Early Roster — Hero (unified_equal) at Random Slot #{slot_num}/{n_teams} ===")
        print(
            f"  ESPN Active FPTS So Far: {rec['hero_af']:,} vs Field Avg {rec['field_af']:,.1f} ({rec['diff']:+,.1f}) | "
            f"Finish: #{rec['rank']:.1f}/{n_teams} | Rookies: {rec['hero_rk']} | Primary C: {rec['hero_c']}/{max_centers}"
        )
        print(f"  {'Rd':>2s} {'Pk':>3s} {'Slot':>4s} | {'Player':23s} {'Type':>4s} {'Pos':>8s} | {'ADP':>5s} {'HeroRk':>6s} | {'Gm':>3s} {'FPPG':>5s} {'FPTS':>6s}")
        print("  " + "-" * 85)
        for rnd, pick_num, idx in picks:
            p = players_s[idx]
            print(
                f"  {rnd:2d} {pick_num:3d} {slot_of[idx]:>4s} | {p['player'][:23]:23s} {tag_map[p['player_type']]:>4s} {p['pos_str'][:8]:>8s} | "
                f"{p['adp']:5.1f} {rank_of[idx]:6d} | {p['early_games']:3d} {p['early_raw_fppg']:5.1f} {p['early_total_fpts']:6,d}"
            )
    return sim_rows


def run_comparison(
    data,
    target_season="2026-27",
    primary_strategy="unified_equal",
    board_lambda=None,
    max_train_adp=175.0,
    max_pool_adp=225.0,
    cv_folds=6,
    trials_per_slot=12,
    adp_var=2.0,
    skip_sim=False,
    exclude_players="",
    seed=None,
):
    if not re.fullmatch(r"\d{4}-\d{2}", target_season):
        raise ValueError(f"--season expects YYYY-YY format (got {target_season!r})")
    target_year = int(target_season[:4])
    prior_year = target_year - 1
    target_folder = data / target_season
    adp_path = target_folder / "adp.csv"
    if not adp_path.is_file():
        raise ValueError(
            f"missing {adp_path}. Download {target_season} first with:\n"
            f"  .\\.venv\\Scripts\\python.exe download.py {target_year} --adp-year-is start"
        )

    # 1. Train strictly on all completed historical seasons < target_season
    hist_rows = build_completed_dataset(data, max_pool_adp=max_pool_adp)
    prior_seasons = sorted({r["draft_season"] for r in hist_rows if r["draft_season"] < target_season})
    if not prior_seasons:
        raise ValueError(f"no completed historical draft seasons before {target_season}")
    train_all = [r for r in hist_rows if r["draft_season"] in prior_seasons]

    # Tune lambdas strictly on prior historical CV folds (zero leakage from target_season)
    if board_lambda is None:
        cv_seasons = prior_seasons[2:][-cv_folds:] if len(prior_seasons) >= 3 else prior_seasons[1:]
        cv_scored = []
        for s in cv_seasons:
            earlier = [r for r in train_all if r["draft_season"] < s]
            cands = [r.copy() for r in train_all if r["draft_season"] == s]
            if len(earlier) < 60 or not cands:
                continue
            sc, _ = fit_and_score_full_pool(earlier, cands, max_train_adp=max_train_adp)
            for i, row in enumerate(cands):
                for st, vals in sc.items():
                    row[f"{st}_score"] = float(vals[i])
            cv_scored.extend(cands)
        lambdas = tune_global_lambdas(cv_scored) if cv_scored else {s: (0.0 if s == "market" else 0.50) for s in STRATEGIES}
    else:
        lambdas = {s: (0.0 if s == "market" else board_lambda) for s in STRATEGIES}

    # 2. Load prior season (e.g. 2025-26) + optional in-progress target season (e.g. 2026-27)
    seasons, get_season, get_summaries, get_leaderboard = load_season_cache(data)
    prior_res = get_season(prior_year)
    if prior_res is None:
        raise ValueError(f"missing completed prior season {season_label(prior_year)}")

    early_data = read_in_progress_season(target_folder)
    next_res_for_matching = (None, early_data["players"]) if early_data is not None else None

    played_before = set(prior_res[1].keys())
    history_available = True
    try:
        for y in range(FIRST_ARCHIVE_YEAR, prior_year + 2):
            played_before.update(players_with_appearances(data, y))
    except Exception:
        history_available = False

    excluded_keys = [name_key(p) for p in exclude_players.split(",") if p.strip()] if exclude_players else []
    cands, counts, removed = build_unified_season_rows(
        data=data,
        prior_year=prior_year,
        prior_folder=seasons[prior_year],
        target_season=target_season,
        adp_path=adp_path,
        prior_res=prior_res,
        prior_sum=get_summaries(prior_year),
        prev2_sum=get_summaries(prior_year - 1) if (prior_year - 1) in seasons else None,
        prior_lb_tuple=get_leaderboard(prior_year),
        next_res=next_res_for_matching,
        next_sum=None,
        next_lb_tuple=None,
        played_before=played_before,
        history_available=history_available,
        max_pool_adp=max_pool_adp,
        live_mode=(early_data is None),
        excluded_keys=excluded_keys,
    )
    resolve_early_returning_absences(data, prior_year, cands, played_before)

    # 3. Fit models on all completed seasons through prior_seasons[-1] and score target_season
    scores, meta = fit_and_score_full_pool(train_all, cands, max_train_adp=max_train_adp)
    for i, row in enumerate(cands):
        for st, vals in scores.items():
            row[f"{st}_score"] = float(vals[i])

    # 4. Attach early-season actuals (scaled availability-adjusted FPPG) if 2026-27 games exist
    early_lb = None
    if early_data is not None:
        early_lb = compute_scaled_adj_leaderboard(early_data["players"], early_data["avg_team_games"])
        early_players = early_data["players"]
        waiver = early_lb["waiver_fppg"]
        for row in cands:
            aid = row["athlete_id"]
            p = early_players.get(aid)
            g = p.total_games if p else 0
            tot = p.total_points if p else 0
            raw_fppg = early_lb["raw_by_id"].get(aid, 0.0)
            adj_fppg = early_lb["adj_by_id"].get(aid, waiver)
            if g > 0 and adj_fppg in early_lb["adj_rank_by_score"]:
                e_rank = early_lb["adj_rank_by_score"][adj_fppg]
            else:
                e_rank = sum(1 for v in early_lb["adj_lb"] if v > waiver) + 1
            slot_adj = adp_slot_points(early_lb["adj_lb"], row["adp"])
            row.update({
                "early_games": g,
                "early_total_fpts": tot,
                "early_raw_fppg": round(raw_fppg, 2),
                "early_adj_fppg": round(adj_fppg, 2),
                "early_slot_adj_fppg": round(slot_adj, 2),
                "early_adj_vs_adp": round(adj_fppg - slot_adj, 2),
                "early_rank": e_rank,
                "early_slots_gained": round(row["adp"] - e_rank, 1),
                "early_beat_adp": e_rank < row["adp"],
            })

    # 5. Build primary draft board and export full audit CSV
    lam_primary = lambdas[primary_strategy]
    ordered = build_global_board(cands, primary_strategy, lam_primary)
    peer_center = median(adj for _, adj, rec in ordered if rec["market_rank"] <= BOARD_SIZE)

    board_rows = []
    for rank, (proj, adj, rec) in enumerate(ordered, start=1):
        out_r = {
            "board_rank": rank,
            "in_top_150": rank <= BOARD_SIZE,
            "player": rec["player"],
            "player_type": rec["player_type"],
            "positions": ",".join(rec["adp_positions"]) if rec.get("adp_positions") else "",
            "adp": rec["adp"],
            "market_rank": rec["market_rank"],
            "slot_move": rec["market_rank"] - rank,
            "proj_adj_fppg": round(proj, 2),
            "mkt_expected_adj_fppg": round(rec["mkt_expected_adj_fppg"], 2),
            "edge_vs_peers": round(adj - peer_center, 2),
            "edge_vs_adp": round(adj, 2),
            "market_collapse_flag": rec["market_collapse_excess"] > 0,
            "prior_games": rec["prior_games"],
            "prior_fppg": round(rec["prior_fppg"], 2) if rec["prior_fppg"] is not None else None,
            "draft_pick": rec["draft_pick"],
            "rookie_age": rec["rookie_age"],
            "team_top8_avg_age": rec["team_top8_avg_age"],
            "team_prior_win_pct": rec["team_prior_win_pct"],
        }
        if early_data is not None:
            out_r.update({
                "early_games": rec["early_games"],
                "early_total_fpts": rec["early_total_fpts"],
                "early_raw_fppg": rec["early_raw_fppg"],
                "early_adj_fppg": rec["early_adj_fppg"],
                "early_rank": rec["early_rank"],
                "early_adj_vs_adp": rec["early_adj_vs_adp"],
                "early_beat_adp": rec["early_beat_adp"],
            })
        board_rows.append(out_r)

    board_path = data / f"draft_board_{target_season}_early_audit.csv"
    write_csv(board_path, board_rows)

    wh, wd = meta["vet_weights"]
    print(f"\n================ {target_season} DRAFT BOARD & TOO-EARLY COMPARISON ================")
    print(f"Trained strictly on {len(prior_seasons)} prior completed drafts: {', '.join(prior_seasons)}")
    print(f"Pool: {len(cands)} players ({counts['returning_vet']} vets, {counts['rookie']} rookies, {counts['returning_absence']} returning/unmatched)")
    print(f"Primary strategy: {primary_strategy} (lam={lam_primary:.2f}) | Vet weights (huber/decomp): {wh:.2f}/{wd:.2f} | Rookie bias: {meta['rookie_flat_bias']:+.2f}")
    if removed:
        print(f"Manually excluded: {', '.join(removed)}")

    if early_data is None:
        print(f"\nNo completed regular-season games found yet in {target_folder}/games.csv — printing pre-season draft board only.")
    else:
        print(
            f"Actual {target_season} progress: {early_data['reg_games_count']} regular-season games through {early_data['latest_date']} "
            f"(~{early_data['avg_team_games']:.1f} games/team = {early_data['season_pct']:.1f}% of season; "
            f"scaled AdjFPPG ref={early_lb['ref_games']:.1f}g, waiver={early_lb['waiver_fppg']:.2f})"
        )

        # Strategy-level Global Top-150 Board scorecard so far
        print(f"\n=== TOO-EARLY Global Top-150 Board Comparison ({target_season} Through {early_data['latest_date']}) ===")
        for strat in STRATEGIES:
            lam = lambdas[strat]
            ord_s = build_global_board(cands, strat, lam)
            top150, outside = ord_s[:BOARD_SIZE], ord_s[BOARD_SIZE:]
            cap = sum(1 for _, _, x in top150 if x["early_rank"] <= BOARD_SIZE)
            avg_adj = mean(x["early_adj_fppg"] for _, _, x in top150)
            mae = mean(abs(rk - x["early_rank"]) for rk, (_, _, x) in enumerate(top150, start=1))
            rk_cnt = sum(1 for _, _, x in top150 if x["player_type"] == "rookie")
            z0 = sum(1 for _, _, x in top150 if x["early_games"] == 0)
            added = [x for _, _, x in top150 if x["market_rank"] > BOARD_SIZE]
            dropped = [x for _, _, x in outside if x["market_rank"] <= BOARD_SIZE]
            swap_str = (
                f"Swaps({len(added):2d}): +{100*sum(x['early_rank']<=BOARD_SIZE for x in added)/len(added):4.1f}% vs "
                f"-{100*sum(x['early_rank']<=BOARD_SIZE for x in dropped)/len(dropped):4.1f}% Top150 "
                f"({mean(x['early_adj_fppg'] for x in added) - mean(x['early_adj_fppg'] for x in dropped):+.2f} AdjFPPG)"
                if added and dropped else "Swaps( 0): market baseline"
            )
            print(
                f"  {strat:20s} (lam={lam:.2f}) | Top150: {cap:3d}/150 ({100*cap/150:5.2f}%) | "
                f"AvgAdj: {avg_adj:5.2f} | MAE: {mae:5.2f} | Rk:{rk_cnt:2d} 0g:{z0:2d} | {swap_str}"
            )

    # Helper to print player tables cleanly (with or without early actuals)
    def print_player_table(title, rows_sub):
        if not rows_sub:
            return
        print(f"\n=== {title} ===")
        if early_data is not None:
            print(f"  {'Rk':>3s} {'ADP':>5s} {'Mv':>4s} | {'Player':23s} {'Typ':>4s} | {'Edge':>6s} | {'Gm':>2s} {'FPPG':>5s} {'Adj':>5s} {'ActRk':>5s} {'VsADP':>6s} {'Hit':>3s}")
            print("  " + "-" * 89)
            for r in rows_sub:
                tag = {"returning_vet": "VET", "rookie": "ROOK", "returning_absence": "RET"}[r["player_type"]]
                hit = "YES" if r["early_beat_adp"] else "no"
                print(
                    f"  {r['board_rank']:3d} {r['adp']:5.1f} {r['slot_move']:+4d} | {r['player'][:23]:23s} {tag:>4s} | "
                    f"{r['edge_vs_peers']:+6.2f} | {r['early_games']:2d} {r['early_raw_fppg']:5.1f} {r['early_adj_fppg']:5.1f} "
                    f"{r['early_rank']:5d} {r['early_adj_vs_adp']:+6.2f} {hit:>3s}"
                )
            hit_pct = 100.0 * sum(1 for r in rows_sub if r["early_beat_adp"]) / len(rows_sub)
            avg_v = mean(r["early_adj_vs_adp"] for r in rows_sub)
            print(f"  --> Group Summary ({len(rows_sub)} players): {hit_pct:.1f}% beating ADP so far | Avg Scaled AdjFPPG vs ADP: {avg_v:+.2f}")
        else:
            print(f"  {'Rk':>3s} {'ADP':>5s} {'Mv':>4s} | {'Player':23s} {'Typ':>4s} | {'Proj':>5s} {'Edge':>6s}")
            print("  " + "-" * 62)
            for r in rows_sub:
                tag = {"returning_vet": "VET", "rookie": "ROOK", "returning_absence": "RET"}[r["player_type"]]
                print(
                    f"  {r['board_rank']:3d} {r['adp']:5.1f} {r['slot_move']:+4d} | {r['player'][:23]:23s} {tag:>4s} | "
                    f"{r['proj_adj_fppg']:5.2f} {r['edge_vs_peers']:+6.2f}"
                )

    top150_rows = board_rows[:BOARD_SIZE]
    targets = sorted(
        [r for r in top150_rows if r["edge_vs_peers"] > 0 and not r["market_collapse_flag"]],
        key=lambda r: (-r["edge_vs_peers"], -r["slot_move"]),
    )[:15]
    fades = sorted(
        [r for r in board_rows if r["market_rank"] <= BOARD_SIZE and r["edge_vs_peers"] < 0],
        key=lambda r: (r["edge_vs_peers"], r["slot_move"]),
    )[:15]
    rookies_sub = sorted([r for r in board_rows if r["player_type"] == "rookie"], key=lambda r: r["adp"])[:15]
    flagged = [r for r in board_rows if r["market_collapse_flag"] and r["market_rank"] <= 175]

    if flagged:
        print_player_table(f"Auto-Detected Market-Collapse / Injury-Risk Flags ({target_season})", flagged[:10])
    print_player_table(f"Top 15 Pre-Season Model Targets ({primary_strategy}) vs. Early {target_season} Actuals", targets)
    print_player_table(f"Top 15 Pre-Season Model Fades ({primary_strategy}) vs. Early {target_season} Actuals", fades)
    if rookies_sub:
        print_player_table(f"Incoming {target_season} Rookies: Model View vs. Early Actuals", rookies_sub)

    if early_data is not None and not skip_sim:
        sim_rows = run_early_espn_sim(
            data, target_season, cands, lambdas,
            trials_per_slot=trials_per_slot, adp_var=adp_var, seed=seed,
        )
        if sim_rows:
            write_csv(data / f"sim_early_{target_season}.csv", sim_rows)

    print(f"\nSaved full {target_season} board & early audit to {board_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"), help="download folder (default: data)")
    parser.add_argument("--season", default="2026-27", help="target in-progress or upcoming season (default: 2026-27)")
    parser.add_argument("--strategy", choices=STRATEGIES[1:], default="unified_equal", help="primary strategy to highlight (default: unified_equal)")
    parser.add_argument("--board-lambda", type=float, default=None, help="override global residual lambda")
    parser.add_argument("--cv-folds", type=int, default=6, help="prior CV folds for lambda tuning (default: 6)")
    parser.add_argument("--trials-per-slot", type=int, default=12, help="Monte Carlo drafts per slot for early ESPN sim (default: 12)")
    parser.add_argument("--adp-var", type=float, default=2.0, help="villain R2-13 Uniform ADP noise bound (default: 2.0)")
    parser.add_argument("--skip-sim", action="store_true", help="skip the early 12-team ESPN draft simulation")
    parser.add_argument("--exclude-players", default="", help="comma-separated known injured/out players to exclude")
    parser.add_argument("--seed", type=int, default=None, help="simulation random seed")
    args = parser.parse_args()
    try:
        run_comparison(
            data=args.data,
            target_season=args.season,
            primary_strategy=args.strategy,
            board_lambda=args.board_lambda,
            cv_folds=args.cv_folds,
            trials_per_slot=args.trials_per_slot,
            adp_var=args.adp_var,
            skip_sim=args.skip_sim,
            exclude_players=args.exclude_players,
            seed=args.seed,
        )
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(1, f"Early-season comparison failed: {error}\n")


if __name__ == "__main__":
    main()
