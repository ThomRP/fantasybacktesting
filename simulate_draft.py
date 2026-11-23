"""12-team ESPN snake draft simulator with 4-C cap, 10-starter/3-bench daily lineup scoring, and R1 straight ADP."""

import argparse
import csv
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from adp_names import clean_adp_player_name
from backtest_common import SCORING, name_key, player_lines, read_csv, regular_season_game, write_csv

MODEL_NAMES = (
    "market",
    "logistic_base",
    "huber_clean",
    "decomposed_clean",
    "ensemble_equal",
    "ensemble_stacked",
)
RESIDUAL_CLIP = 8.0
VALID_POS = ("PG", "SG", "SF", "PF", "C")
SLOT_NAMES_10 = ("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL", "UTIL", "UTIL")
SLOT_MASK_MAP = {
    "PG": (1 << 0) | (1 << 5),
    "SG": (1 << 1) | (1 << 5),
    "SF": (1 << 2) | (1 << 6),
    "PF": (1 << 3) | (1 << 6),
    "C":  (1 << 4),
}
UTIL_BITS = (1 << 7) | (1 << 8) | (1 << 9)


def _pos_to_masks(pos_tuple):
    m7 = 0
    for p in pos_tuple:
        m7 |= SLOT_MASK_MAP.get(p, 0)
    if m7 == 0:
        m7 = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 5) | (1 << 6)
    return m7, m7 | UTIL_BITS


def _dfs_match(u, match_slot, masks, vis):
    m = masks[u]
    while m:
        lsb = m & -m
        s = lsb.bit_length() - 1
        m ^= lsb
        if not (vis[0] & lsb):
            vis[0] |= lsb
            prev = match_slot[s]
            if prev == -1 or _dfs_match(prev, match_slot, masks, vis):
                match_slot[s] = u
                return True
    return False


def max_matching_7(roster_indices, masks7):
    """Exact maximum matching size of roster players against the 7 constrained ESPN slots (PG,SG,SF,PF,C,G,F)."""
    match_slot = [-1] * 7
    matched = 0
    for u in roster_indices:
        if _dfs_match(u, match_slot, masks7, [0]):
            matched += 1
            if matched == 7:
                break
    return matched


def assign_lineup_slots(roster_indices, masks10, prio):
    """Assign each of the 13 drafted players to a legal ESPN slot (10 starters + 3 BE) for display."""
    ordered = sorted(roster_indices, key=lambda idx: -prio[idx])
    match_slot = [-1] * 10
    filled = 0
    for idx in ordered:
        if filled < 10 and _dfs_match(idx, match_slot, masks10, [0]):
            filled += 1
    slot_of = {idx: "BE" for idx in roster_indices}
    for s_idx, p_idx in enumerate(match_slot):
        if p_idx != -1:
            slot_of[p_idx] = SLOT_NAMES_10[s_idx]
    return slot_of


def parse_adp_positions(adp_path):
    """Parse FantasyPros multi-position eligibility (in listed order) from adp.csv."""
    pos_map = {}
    if not adp_path.is_file():
        return pos_map
    with adp_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in (reader.fieldnames or [])}
        raw_cols = [cols[k] for k in ("player", "playername") if k in cols]
        for row in reader:
            raw_text = ""
            for c in raw_cols:
                val = row.get(c) or ""
                if re.search(r"\([^()]+\s+-\s+[^()]+\)", val):
                    raw_text = val
                    break
            if not raw_text and raw_cols:
                raw_text = row.get(raw_cols[0]) or ""
            cleaned = clean_adp_player_name(raw_text)
            if not cleaned:
                continue
            key = name_key(cleaned)
            if key in pos_map:
                continue
            m = re.search(r"\([^()]+\s+-\s+([^()]+)\)", raw_text)
            if m:
                tokens = []
                for tok in m.group(1).split(","):
                    t = tok.strip().upper()
                    if t in VALID_POS and t not in tokens:
                        tokens.append(t)
                    elif t == "G":
                        for gpos in ("PG", "SG"):
                            if gpos not in tokens:
                                tokens.append(gpos)
                    elif t == "F":
                        for fpos in ("SF", "PF"):
                            if fpos not in tokens:
                                tokens.append(fpos)
                if tokens:
                    pos_map[key] = tuple(tokens)
    return pos_map


def load_season_daily_logs(season_folder, needed_ids):
    """Load regular-season daily fantasy points and fallback box-score positions for a season."""
    schedule = list(read_csv(season_folder / "games.csv"))
    game_dates = {g["game_id"]: g["game_date"] for g in schedule if regular_season_game(g)}
    dates_sorted = sorted(set(game_dates.values()))
    date_to_idx = {d: i for i, d in enumerate(dates_sorted)}

    daily = defaultdict(dict)
    pos_counts = defaultdict(Counter)
    for gid, aid, _, fpts, pos, _, _ in player_lines(season_folder, game_dates):
        if aid not in needed_ids:
            continue
        di = date_to_idx[game_dates[gid]]
        daily[aid][di] = daily[aid].get(di, 0) + fpts
        if pos:
            pos_counts[aid][pos.strip().upper()] += 1

    fallback_pos = {}
    for aid, ctr in pos_counts.items():
        top = ctr.most_common(1)[0][0]
        if top in VALID_POS:
            fallback_pos[aid] = (top,)
        elif top == "G":
            fallback_pos[aid] = ("PG", "SG")
        elif top == "F":
            fallback_pos[aid] = ("SF", "PF")
    return len(dates_sorted), daily, fallback_pos


def load_saved_lambdas(data):
    lambdas = {"market": 0.0}
    path = data / "backtest_final_global_board_summary.csv"
    if not path.is_file():
        return {m: (0.0 if m == "market" else 0.65) for m in MODEL_NAMES}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            model = row.get("model")
            if model in MODEL_NAMES and row.get("lambda"):
                lambdas[model] = float(row["lambda"])
    for m in MODEL_NAMES:
        lambdas.setdefault(m, 0.0 if m == "market" else 0.65)
    return lambdas


def load_season_data(data, center_rule="primary"):
    """Load saved CV/Holdout player pools + positions from adp.csv + daily game logs from player_box_scores.csv."""
    raw_by_season = {}
    for split, fname in (("cv", "backtest_final_cv_results.csv"), ("holdout", "backtest_final_results.csv")):
        path = data / fname
        if not path.is_file():
            continue
        by_season = defaultdict(list)
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                games = int(float(row["next_games"]))
                raw_fppg = float(row["next_raw_fppg"])
                total_fpts = int(round(raw_fppg * games)) if games > 0 else 0
                rec = {
                    "split": split,
                    "draft_season": row["draft_season"],
                    "player": row["player"],
                    "athlete_id": row["athlete_id"],
                    "adp": float(row["adp"]),
                    "market_rank": int(float(row["market_rank"])),
                    "mkt_expected_adj_fppg": float(row["mkt_expected_adj_fppg"]),
                    "next_games": games,
                    "next_raw_fppg": raw_fppg,
                    "total_fpts": total_fpts,
                }
                for m in MODEL_NAMES:
                    if m != "market":
                        rec[f"{m}_score"] = float(row[f"{m}_score"])
                by_season[rec["draft_season"]].append(rec)
        for season, players in by_season.items():
            players.sort(key=lambda r: (r["adp"], r["market_rank"], r["athlete_id"]))
            raw_by_season[season] = (split, players)

    if not raw_by_season:
        raise ValueError(
            f"no saved results found in {data}. Run backtest_final_board.py first to generate "
            "backtest_final_cv_results.csv and backtest_final_results.csv"
        )

    season_bundles = {}
    for season in sorted(raw_by_season):
        split, players = raw_by_season[season]
        s_folder = data / season
        adp_pos = parse_adp_positions(s_folder / "adp.csv")
        needed_ids = {p["athlete_id"] for p in players}

        has_daily = (s_folder / "games.csv").is_file() and (s_folder / "player_box_scores.csv").is_file()
        if has_daily:
            n_days, daily, fallback_pos = load_season_daily_logs(s_folder, needed_ids)
        else:
            n_days, daily, fallback_pos = 0, {}, {}

        n_players = len(players)
        played_mat = np.zeros((n_players, max(n_days, 1)), dtype=np.uint8)
        fpts_mat = np.zeros((n_players, max(n_days, 1)), dtype=np.int16)

        masks7, masks10, is_center, prio, tot_fpts = [], [], [], [], []
        for idx, p in enumerate(players):
            aid = p["athlete_id"]
            ptup = adp_pos.get(name_key(p["player"])) or fallback_pos.get(aid) or ("SG", "SF")
            m7, m10 = _pos_to_masks(ptup)
            p["positions"] = ptup
            p["pos_str"] = ",".join(ptup)
            c_flag = (ptup[0] == "C") if center_rule == "primary" else ("C" in ptup)
            p["counts_as_center"] = c_flag

            masks7.append(m7)
            masks10.append(m10)
            is_center.append(c_flag)
            prio.append(p["next_raw_fppg"] if p["next_games"] > 0 else -1.0)
            if has_daily:
                day_map = daily.get(aid, {})
                for d, pts in day_map.items():
                    played_mat[idx, d] = 1
                    fpts_mat[idx, d] = pts
                p["total_fpts"] = int(sum(day_map.values()))
            tot_fpts.append(p["total_fpts"])

        season_bundles[season] = {
            "split": split,
            "players": players,
            "has_daily": has_daily,
            "n_days": n_days,
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
    return season_bundles


def score_roster_espn(roster, bundle):
    """Score a 13-man roster day-by-day under standard ESPN active slots (PG,SG,SF,PF,C,G,F,3xUTIL) + 3 BE."""
    raw_total = sum(bundle["tot_fpts"][i] for i in roster)
    if not bundle["has_daily"]:
        return raw_total, raw_total, 0, 0

    key = tuple(sorted(roster))
    cached = bundle["roster_cache"].get(key)
    if cached is not None:
        return cached

    prio = bundle["prio"]
    ordered = sorted(roster, key=lambda idx: -prio[idx])
    counts = bundle["played_mat"][ordered].sum(axis=0)
    crowded_days = np.flatnonzero(counts >= 5).tolist()

    played_lists = bundle["played_lists"]
    fpts_lists = bundle["fpts_lists"]
    masks10 = bundle["masks10"]

    benched_pts = 0
    benched_g = 0
    for d in crowded_days:
        free_bits = 0x3FF
        match_slot = None
        filled = 0
        for idx in ordered:
            if not played_lists[idx][d]:
                continue
            if filled == 10:
                benched_g += 1
                benched_pts += fpts_lists[idx][d]
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
                    if _dfs_match(j, match_slot, masks10, [0]):
                        filled += 1
            if filled < 10 and _dfs_match(idx, match_slot, masks10, [0]):
                filled += 1
            else:
                benched_g += 1
                benched_pts += fpts_lists[idx][d]

    res = (raw_total - benched_pts, raw_total, benched_g, benched_pts)
    bundle["roster_cache"][key] = res
    return res


def build_hero_order(players, model, lam):
    if model == "market":
        return list(range(len(players)))
    scored = []
    for idx, p in enumerate(players):
        edge = max(-RESIDUAL_CLIP, min(RESIDUAL_CLIP, p[f"{model}_score"]))
        proj = p["mkt_expected_adj_fppg"] + lam * edge
        scored.append((proj, -p["adp"], idx))
    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    return [idx for _, _, idx in scored]


def build_opponent_noisy_orders(players, n_teams, adp_var, noise_type, rng):
    orders = []
    n_players = len(players)
    for _ in range(n_teams):
        if noise_type == "uniform":
            noisy = [(players[i]["adp"] + rng.uniform(-adp_var, adp_var), i) for i in range(n_players)]
        else:
            noisy = [(players[i]["adp"] + rng.gauss(0.0, adp_var), i) for i in range(n_players)]
        noisy.sort(key=lambda t: (t[0], t[1]))
        orders.append([i for _, i in noisy])
    return orders


def build_pick_order(n_teams, n_rounds, draft_type="snake"):
    order = []
    for rnd in range(n_rounds):
        teams = range(n_teams - 1, -1, -1) if (draft_type == "snake" and rnd % 2 == 1) else range(n_teams)
        for team in teams:
            order.append((rnd + 1, team))
    return order


def run_single_draft(bundle, hero_slot, hero_order, straight_adp_order, opp_noisy_orders, pick_order, n_teams, n_rounds, max_centers, straight_rounds):
    """Simulate one full draft enforcing ESPN 4-C limit and completable 10-starter + 3-bench rosters."""
    players = bundle["players"]
    masks7 = bundle["masks7"]
    is_center = bundle["is_center"]

    taken = [False] * len(players)
    rosters = [[] for _ in range(n_teams)]
    c_counts = [0] * n_teams
    match_sizes = [0] * n_teams
    hero_picks = []

    for pick_num, (rnd, team) in enumerate(pick_order, start=1):
        if team == hero_slot:
            board = hero_order
        elif rnd <= straight_rounds:
            board = straight_adp_order
        else:
            board = opp_noisy_orders[team]

        roster = rosters[team]
        rem_after = n_rounds - len(roster) - 1
        cur_c = c_counts[team]
        cur_m = match_sizes[team]

        choice = None
        new_m = cur_m
        # Pass 1: highest-ranked available player who respects BOTH 4-C cap AND completable 7 constrained starter slots
        for idx in board:
            if taken[idx]:
                continue
            if is_center[idx] and cur_c >= max_centers:
                continue
            if cur_m + rem_after >= 7:
                # Even without increasing matching size, remaining picks can fill all 7 constrained slots
                choice = idx
                new_m = max_matching_7(roster + [idx], masks7) if cur_m < 7 else 7
                break
            cand_m = max_matching_7(roster + [idx], masks7)
            if cand_m + rem_after >= 7:
                choice = idx
                new_m = cand_m
                break

        # Fallback if pool somehow exhausted legal combinations (should never trigger on >=180-player pools)
        if choice is None:
            for idx in board:
                if not taken[idx] and not (is_center[idx] and cur_c >= max_centers):
                    choice = idx
                    new_m = max_matching_7(roster + [idx], masks7)
                    break
        if choice is None:
            choice = next(idx for idx in board if not taken[idx])
            new_m = max_matching_7(roster + [choice], masks7)

        taken[choice] = True
        roster.append(choice)
        if is_center[choice]:
            c_counts[team] += 1
        match_sizes[team] = new_m
        if team == hero_slot:
            hero_picks.append((rnd, pick_num, choice))

    # Score all 12 teams under exact daily ESPN 10-starter + 3-bench rules
    active_fpts, raw_fpts, benched_g, benched_pts = [], [], [], []
    for t in range(n_teams):
        af, rf, bg, bp = score_roster_espn(rosters[t], bundle)
        active_fpts.append(af)
        raw_fpts.append(rf)
        benched_g.append(bg)
        benched_pts.append(bp)

    hero_af = active_fpts[hero_slot]
    field_af = [active_fpts[t] for t in range(n_teams) if t != hero_slot]
    field_af_avg = sum(field_af) / len(field_af)

    hero_rf = raw_fpts[hero_slot]
    field_rf_avg = sum(raw_fpts[t] for t in range(n_teams) if t != hero_slot) / (n_teams - 1)

    better = sum(1 for x in field_af if x > hero_af)
    tied = sum(1 for x in field_af if x == hero_af)
    rank = 1.0 + better + 0.5 * tied
    won = 1.0 / (1 + tied) if better == 0 else 0.0
    top4 = 1.0 if rank <= 4.0 else 0.0
    beat_field = 1.0 if hero_af > field_af_avg else (0.5 if hero_af == field_af_avg else 0.0)

    hero_0g = sum(1 for i in rosters[hero_slot] if players[i]["next_games"] == 0)
    field_0g = sum(1 for t in range(n_teams) if t != hero_slot for i in rosters[t] if players[i]["next_games"] == 0) / (n_teams - 1)
    hero_bg = benched_g[hero_slot]
    field_bg = sum(benched_g[t] for t in range(n_teams) if t != hero_slot) / (n_teams - 1)

    stats = {
        "hero_active_fpts": hero_af,
        "field_active_fpts": field_af_avg,
        "active_diff": hero_af - field_af_avg,
        "raw_diff": hero_rf - field_rf_avg,
        "rank": rank,
        "won": won,
        "top4": top4,
        "beat_field": beat_field,
        "hero_benched_g": hero_bg,
        "field_benched_g": field_bg,
        "hero_benched_pts": benched_pts[hero_slot],
        "hero_0g": hero_0g,
        "field_0g": field_0g,
        "hero_centers": c_counts[hero_slot],
        "hero_legal_lineup": match_sizes[hero_slot] == 7,
    }
    return stats, hero_picks, rosters[hero_slot]


def aggregate_records(records, label_fields):
    hero_af = mean(r["hero_active_fpts"] for r in records)
    field_af = mean(r["field_active_fpts"] for r in records)
    diff = hero_af - field_af
    return {
        **label_fields,
        "drafts": len(records),
        "hero_espn_fpts": round(hero_af, 1),
        "field_espn_fpts": round(field_af, 1),
        "espn_diff_vs_field": round(diff, 1),
        "espn_lift_pct": round(100.0 * diff / field_af, 2) if field_af else None,
        "raw_13man_diff": round(mean(r["raw_diff"] for r in records), 1),
        "avg_league_rank": round(mean(r["rank"] for r in records), 2),
        "win_rate_pct": round(100.0 * mean(r["won"] for r in records), 1),
        "top4_rate_pct": round(100.0 * mean(r["top4"] for r in records), 1),
        "beat_field_pct": round(100.0 * mean(r["beat_field"] for r in records), 1),
        "hero_benched_g": round(mean(r["hero_benched_g"] for r in records), 1),
        "field_benched_g": round(mean(r["field_benched_g"] for r in records), 1),
        "hero_avg_0g": round(mean(r["hero_0g"] for r in records), 2),
        "field_avg_0g": round(mean(r["field_0g"] for r in records), 2),
        "hero_avg_centers": round(mean(r["hero_centers"] for r in records), 2),
        "legal_lineup_pct": round(100.0 * mean(1.0 if r["hero_legal_lineup"] else 0.0 for r in records), 1),
    }


def run_simulation(
    data,
    n_teams=12,
    n_rounds=13,
    trials_per_slot=12,
    adp_var=2.0,
    noise_type="uniform",
    straight_rounds=1,
    max_centers=4,
    center_rule="primary",
    draft_type="snake",
    lam_override=None,
    only_season=None,
    sample_model="ensemble_stacked",
    seed=None,
):
    if seed is None:
        seed = random.SystemRandom().randint(1, 999_999)
    rng = random.Random(seed)

    bundles = load_season_data(data, center_rule=center_rule)
    lambdas = load_saved_lambdas(data)
    if lam_override is not None:
        for m in MODEL_NAMES:
            if m != "market":
                lambdas[m] = lam_override

    seasons = sorted(bundles)
    if only_season:
        if only_season not in bundles:
            raise ValueError(f"season {only_season!r} not found in saved results ({', '.join(seasons)})")
        seasons = [only_season]

    total_picks = n_teams * n_rounds
    pick_order = build_pick_order(n_teams, n_rounds, draft_type=draft_type)
    usable_seasons = [s for s in seasons if len(bundles[s]["players"]) >= total_picks]
    if not usable_seasons:
        raise ValueError(f"no season has at least {total_picks} players")

    sample_slot = rng.randrange(n_teams)
    sample_trial = rng.randrange(trials_per_slot)
    sample_season = usable_seasons[-1]
    sample_draft_info = None

    records_by_split_model = defaultdict(list)
    records_by_season_model = defaultdict(list)

    for season in usable_seasons:
        bundle = bundles[season]
        split = bundle["split"]
        players = bundle["players"]
        straight_order = list(range(len(players)))
        hero_orders = {m: build_hero_order(players, m, lambdas[m]) for m in MODEL_NAMES}

        for slot in range(n_teams):
            for trial in range(trials_per_slot):
                opp_noisy = build_opponent_noisy_orders(players, n_teams, adp_var, noise_type, rng)
                for m in MODEL_NAMES:
                    stats, picks, roster = run_single_draft(
                        bundle, slot, hero_orders[m], straight_order, opp_noisy,
                        pick_order, n_teams, n_rounds, max_centers, straight_rounds,
                    )
                    records_by_split_model[(split, m)].append(stats)
                    records_by_split_model[("all", m)].append(stats)
                    records_by_season_model[(season, split, m)].append(stats)
                    if season == sample_season and slot == sample_slot and trial == sample_trial and m == sample_model:
                        sample_draft_info = (season, slot + 1, bundle, hero_orders[m], picks, roster, stats)

    summary_rows, season_rows = [], []
    for split in ("cv", "holdout", "all"):
        for m in MODEL_NAMES:
            recs = records_by_split_model.get((split, m))
            if recs:
                summary_rows.append(aggregate_records(recs, {"split": split, "model": m, "lambda": lambdas[m]}))
    for season in usable_seasons:
        split = bundles[season]["split"]
        for m in MODEL_NAMES:
            season_rows.append(aggregate_records(records_by_season_model[(season, split, m)], {"season": season, "split": split, "model": m, "lambda": lambdas[m]}))

    write_csv(data / "sim_espn_summary.csv", summary_rows)
    write_csv(data / "sim_espn_by_season.csv", season_rows)

    noise_desc = f"Uniform[-{adp_var:g}, +{adp_var:g}]" if noise_type == "uniform" else f"Normal(0, sd={adp_var:g})"
    daily_seasons = sum(1 for s in usable_seasons if bundles[s]["has_daily"])
    print(f"=== {n_teams}-Team x {n_rounds}-Round ESPN {draft_type.title()} Draft Simulator ===")
    print(f"Roster Rules: PG, SG, SF, PF, C, G, F, 3xUTIL + 3 Bench | Max {max_centers} Centers ({center_rule}-pos rule)")
    print(f"Villains: Round 1-{straight_rounds} straight ADP (0 variance) | Rounds {straight_rounds + 1}-{n_rounds} ADP + {noise_desc}")
    print(f"Run: {trials_per_slot} trials x {n_teams} slots = {trials_per_slot * n_teams} drafts/season | Seed: {seed} (pass --seed {seed} to reproduce)")
    print(f"Seasons ({len(usable_seasons)}, {daily_seasons} with exact day-by-day active lineup scoring): {', '.join(usable_seasons)}")

    for split_label, title in (("cv", "CV SEASONS"), ("holdout", "HOLDOUT SEASONS"), ("all", "ALL SEASONS COMBINED")):
        rows = [r for r in summary_rows if r["split"] == split_label]
        if not rows:
            continue
        n_s = len([s for s in usable_seasons if split_label == "all" or bundles[s]["split"] == split_label])
        print(f"\n=== Hero vs. 11-Team Field — {title} ({n_s} seasons, {rows[0]['drafts']} drafts/model) ===")
        print(
            f"  {'Model':16s} {'Lam':>4s} | {'Hero ESPN':>9s} {'Field ESPN':>10s} {'Diff':>8s} {'Lift%':>6s} | "
            f"{'RawDiff':>7s} | {'Rank':>5s} {'Win%':>5s} {'Top4%':>6s} {'BeatFld':>7s} | {'BnG(H/F)':>9s} {'0g(H/F)':>9s}"
        )
        print("  " + "-" * 121)
        for r in rows:
            print(
                f"  {r['model']:16s} {r['lambda']:4.2f} | "
                f"{r['hero_espn_fpts']:9.1f} {r['field_espn_fpts']:10.1f} {r['espn_diff_vs_field']:+8.1f} {r['espn_lift_pct']:+5.2f}% | "
                f"{r['raw_13man_diff']:+7.1f} | "
                f"{r['avg_league_rank']:5.2f} {r['win_rate_pct']:4.1f}% {r['top4_rate_pct']:5.1f}% {r['beat_field_pct']:6.1f}% | "
                f"{r['hero_benched_g']:4.1f}/{r['field_benched_g']:4.1f} {r['hero_avg_0g']:4.2f}/{r['field_avg_0g']:4.2f}"
            )

    print(f"\n=== Season-by-Season ESPN Active Lineup FPTS Diff ({sample_model} vs. market control) ===")
    print(f"  {'Season':7s} {'Split':7s} | {'Market Diff':>11s} {'Mkt Rank':>8s} | {sample_model + ' Diff':>20s} {'Rank':>6s} {'Win%':>6s} {'Top4%':>6s} {'BeatFld':>7s}")
    print("  " + "-" * 92)
    by_sm = {(r["season"], r["model"]): r for r in season_rows}
    for s in usable_seasons:
        mkt = by_sm[(s, "market")]
        mod = by_sm[(s, sample_model)]
        print(
            f"  {s:7s} {mod['split']:7s} | {mkt['espn_diff_vs_field']:+11.1f} {mkt['avg_league_rank']:8.2f} | "
            f"{mod['espn_diff_vs_field']:+20.1f} {mod['avg_league_rank']:6.2f} {mod['win_rate_pct']:5.1f}% "
            f"{mod['top4_rate_pct']:5.1f}% {mod['beat_field_pct']:6.1f}%"
        )

    if sample_draft_info:
        season, slot_num, bundle, hero_order, picks, roster, stats = sample_draft_info
        players = bundle["players"]
        rank_of = {idx: r for r, idx in enumerate(hero_order, start=1)}
        slot_of = assign_lineup_slots(roster, bundle["masks10"], bundle["prio"])
        print(f"\n=== Sample {season} ESPN Roster — Hero ({sample_model}) at Random Slot #{slot_num}/{n_teams} ===")
        print(
            f"  ESPN Active FPTS: {stats['hero_active_fpts']:,} vs Field Avg {stats['field_active_fpts']:,.1f} "
            f"({stats['active_diff']:+,.1f}) | Finish: #{stats['rank']:.1f}/{n_teams} | "
            f"Primary C: {stats['hero_centers']}/{max_centers} | Benched: {stats['hero_benched_g']}g ({stats['hero_benched_pts']:,} FPTS)"
        )
        print(f"  {'Rd':>2s} {'Pk':>3s} {'Slot':>4s} | {'Player':23s} {'Pos':>8s} | {'ADP':>5s} {'HeroRk':>6s} | {'Gm':>3s} {'FPPG':>5s} {'Total':>7s}")
        print("  " + "-" * 81)
        for rnd, pick_num, idx in picks:
            p = players[idx]
            print(
                f"  {rnd:2d} {pick_num:3d} {slot_of[idx]:>4s} | {p['player'][:23]:23s} {p['pos_str'][:8]:>8s} | "
                f"{p['adp']:5.1f} {rank_of[idx]:6d} | {p['next_games']:3d} {p['next_raw_fppg']:5.1f} {p['total_fpts']:7,d}"
            )

    print(f"\nSaved ESPN simulation reports to {data}/sim_espn_*.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"), help="folder containing season folders and backtest_final_*results.csv (default: data)")
    parser.add_argument("--teams", type=int, default=12, help="number of teams (default: 12)")
    parser.add_argument("--rounds", type=int, default=13, help="draft rounds per team (default: 13)")
    parser.add_argument("--trials-per-slot", type=int, default=12, help="Monte Carlo drafts per Hero draft slot (default: 12)")
    parser.add_argument("--adp-var", type=float, default=2.0, help="villain ADP variance bound/std after straight rounds (default: 2.0)")
    parser.add_argument("--noise-type", choices=("uniform", "normal"), default="uniform", help="villain ADP noise distribution (default: uniform [-2, +2])")
    parser.add_argument("--straight-rounds", type=int, default=1, help="initial rounds where villains draft straight ADP with 0 noise (default: 1)")
    parser.add_argument("--max-centers", type=int, default=4, help="ESPN max centers per roster (default: 4)")
    parser.add_argument("--center-rule", choices=("primary", "any"), default="primary", help="count primary C only (ESPN default) or any C eligibility")
    parser.add_argument("--draft-type", choices=("snake", "linear"), default="snake", help="draft order type (default: snake)")
    parser.add_argument("--lam", type=float, default=None, help="override global board lambda for all models")
    parser.add_argument("--season", default=None, help="simulate only one specific season (e.g. 2025-26)")
    parser.add_argument("--sample-model", choices=MODEL_NAMES[1:], default="ensemble_stacked", help="model highlighted in season table & sample roster")
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: random each run)")
    args = parser.parse_args()
    if args.teams < 2 or args.rounds < 10 or args.trials_per_slot < 1 or args.adp_var < 0 or args.straight_rounds < 0 or args.max_centers < 1:
        parser.error("use valid positive parameters (rounds >= 10 for standard ESPN 10-starter lineup)")
    try:
        run_simulation(
            data=args.data,
            n_teams=args.teams,
            n_rounds=args.rounds,
            trials_per_slot=args.trials_per_slot,
            adp_var=args.adp_var,
            noise_type=args.noise_type,
            straight_rounds=args.straight_rounds,
            max_centers=args.max_centers,
            center_rule=args.center_rule,
            draft_type=args.draft_type,
            lam_override=args.lam,
            only_season=args.season,
            sample_model=args.sample_model,
            seed=args.seed,
        )
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(1, f"ESPN draft simulation failed: {error}\n")


if __name__ == "__main__":
    main()
