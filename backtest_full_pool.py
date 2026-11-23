"""Unified full-pool (veterans + rookies) Top-150 backtest, ESPN draft simulator, and live board generator."""

import argparse
import csv
import math
import random
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import mean, median

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from adp_names import clean_adp_player_name
from backtest_common import (
    SCORING,
    adp_slot_points,
    load_birth_dates,
    load_player_profiles,
    name_key,
    parse_minutes,
    player_box_scores,
    player_lines,
    read_csv,
    read_season,
    regular_season_game,
    write_csv,
)
from backtest_rookies import FIRST_ARCHIVE_YEAR, players_with_appearances


BOARD_SIZE = 150
REF_GAMES = 65.0
WAIVER_FACTOR = 0.85
COLLAPSE_THRESHOLD = 8.0
COLLAPSE_MIN_SLOT_DROP = 20.0
RESIDUAL_CLIP = 8.0
RIDGE_ALPHAS = np.logspace(-1, 3.5, 25)
LAMBDA_GRID = (0.35, 0.50, 0.65, 0.80)

# Clean 18-feature returning-veteran set (with raw-FPPG + slot-drop injury-collapse hinge)
VET_CLEAN_FEATURES = (
    "log_adp",
    "mkt_expected_adj_fppg",
    "prior_minus_mkt_capped",
    "market_collapse_excess",
    "age_at_prior_break",
    "age_under_24",
    "age_over_30",
    "prior_games",
    "prior_mpg",
    "prior_fppm",
    "prior_start_pct",
    "yoy_fppg_delta",
    "yoy_mpg_delta",
    "two_yr_weighted_fppg",
    "two_yr_games_avg",
    "no_prev2_season",
    "adp_platform_std",
    "espn_vs_avg",
)

VET_MPG_FEATURES = (
    "log_adp", "mkt_expected_adj_fppg", "prior_minus_mkt_capped", "market_collapse_excess",
    "age_at_prior_break", "age_under_24", "age_over_30", "prior_games", "prior_mpg",
    "prior_start_pct", "yoy_mpg_delta", "two_yr_games_avg", "no_prev2_season",
    "adp_platform_std", "espn_vs_avg",
)

VET_FPPM_FEATURES = (
    "log_adp", "mkt_expected_adj_fppg", "prior_minus_mkt_capped", "market_collapse_excess",
    "age_at_prior_break", "age_under_24", "age_over_30", "prior_fppm", "prior_mpg",
    "yoy_fppg_delta", "two_yr_weighted_fppg", "no_prev2_season",
    "adp_platform_std", "espn_vs_avg",
)

VET_AVAIL_FEATURES = (
    "log_adp", "mkt_expected_adj_fppg", "prior_minus_mkt_capped", "market_collapse_excess",
    "age_at_prior_break", "age_under_24", "age_over_30", "prior_games", "two_yr_games_avg",
    "prior_mpg", "prior_start_pct", "no_prev2_season", "adp_platform_std", "espn_vs_avg",
)

# Lean 11-feature rookie set validated by backtest_rookie_debut.py (dropped noisy same_pos_min_share)
ROOKIE_FEATURES = (
    "log_draft_pick",
    "log_adp",
    "mkt_expected_adj_fppg",
    "adp_minus_pick_expected",
    "is_redshirt",
    "rookie_age",
    "team_top8_avg_age",
    "team_prior_win_pct",
    "team_vet30_min_share",
    "team_prior_top150_count",
    "espn_vs_avg",
)

STRATEGIES = (
    "market",
    "vet_only_rk_neutral",
    "unified_flat_rk",
    "unified_equal",
    "unified_stacked",
)

# Standard ESPN roster constants (PG, SG, SF, PF, C, G, F, 3xUTIL + 3 Bench, max 4 C)
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


def _safe_float(raw):
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def _safe_int(raw):
    val = _safe_float(raw)
    return int(round(val)) if val is not None else None


def season_label(start_year):
    return f"{start_year}-{(start_year + 1) % 100:02d}"


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
    match_slot = [-1] * 7
    matched = 0
    for u in roster_indices:
        if _dfs_match(u, match_slot, masks7, [0]):
            matched += 1
            if matched == 7:
                break
    return matched


def assign_lineup_slots(roster_indices, masks10, prio):
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


def compute_season_adj_leaderboard(players):
    """Availability-adjusted FPPG leaderboard (ref=65g, waiver=0.85x 150th) + raw rank lookup."""
    qual_raw = sorted(
        (p.total_points / p.total_games for p in players.values() if p.total_games >= 20),
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
            eff_g = min(float(g), REF_GAMES)
            adj_by_id[aid] = (eff_g * raw + (REF_GAMES - eff_g) * waiver_fppg) / REF_GAMES

    adj_lb = sorted(adj_by_id.values(), reverse=True)
    adj_rank_by_score = {}
    for place, score in enumerate(adj_lb, start=1):
        adj_rank_by_score.setdefault(score, place)

    raw_lb = sorted((v for v in raw_by_id.values() if v > 0), reverse=True)
    raw_rank_by_score = {}
    for place, score in enumerate(raw_lb, start=1):
        raw_rank_by_score.setdefault(score, place)

    return waiver_fppg, adj_by_id, raw_by_id, adj_lb, adj_rank_by_score, raw_rank_by_score


def read_adp_entries(path):
    """Read all ADP entries with multi-position eligibility, platform spreads, and team tag."""
    entries, seen = [], set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in (reader.fieldnames or [])}
        name_col = cols.get("playername") or cols.get("player")
        avg_col = next((cols[k] for k in ("avg", "average", "adp", "averagedraftposition", "consensusadp") if k in cols), None)
        if not name_col or not avg_col:
            return entries
        raw_cols = [cols[k] for k in ("player", "playername") if k in cols]
        for row in reader:
            raw_text = next(
                (row.get(c) or "" for c in raw_cols if re.search(r"\([^()]+\s+-\s+[^()]+\)", row.get(c) or "")),
                row.get(name_col) or "",
            )
            cleaned = clean_adp_player_name(raw_text)
            adp = _safe_float(row.get(avg_col))
            if not cleaned or adp is None or adp <= 0:
                continue
            key = name_key(cleaned)
            if key in seen:
                continue
            seen.add(key)

            espn = _safe_float(row.get(cols["espn"])) if "espn" in cols else None
            platforms = [_safe_float(row.get(cols[p])) for p in ("yahoo", "espn", "cbs") if p in cols]
            valid = [v for v in platforms if v is not None and v > 0]
            std = float(np.std(valid, ddof=1)) if len(valid) >= 2 else None
            espn_vs_avg = (adp - espn) if (espn is not None and espn > 0) else None

            adp_team_abbr = None
            pos_tokens = []
            m = re.search(r"\(([^()\s-]+)\s+-\s+([^()]+)\)", raw_text)
            if m:
                adp_team_abbr = m.group(1).strip().upper()
                for tok in m.group(2).split(","):
                    t = tok.strip().upper()
                    if t in VALID_POS and t not in pos_tokens:
                        pos_tokens.append(t)
                    elif t == "G":
                        for gp in ("PG", "SG"):
                            if gp not in pos_tokens:
                                pos_tokens.append(gp)
                    elif t == "F":
                        for fp in ("SF", "PF"):
                            if fp not in pos_tokens:
                                pos_tokens.append(fp)

            entries.append({
                "key": key,
                "player": cleaned,
                "adp": adp,
                "adp_platform_std": std,
                "espn_vs_avg": espn_vs_avg,
                "adp_team_abbr": adp_team_abbr,
                "adp_positions": tuple(pos_tokens) if pos_tokens else None,
            })
    entries.sort(key=lambda r: (r["adp"], r["player"].casefold()))
    return entries


def parse_draft_info_from_profiles(profiles):
    """Extract draft pick, DOB, experience, team_id, position, and names from ESPN player_core profiles."""
    if not profiles:
        return {}
    rd2_sel = [
        s for p in profiles.values()
        if _safe_int(p.get("draft_round")) == 2 and (s := _safe_int(p.get("draft_selection"))) is not None and s > 0
    ]
    selection_is_overall = any(s > 30 for s in rd2_sel)

    parsed = {}
    for aid, p in profiles.items():
        dy = _safe_int(p.get("draft_year"))
        rd = _safe_int(p.get("draft_round"))
        sel = _safe_int(p.get("draft_selection"))
        overall = None
        for col in ("draft_overall", "overall_pick", "draft_pick"):
            v = _safe_int(p.get(col))
            if v is not None and 1 <= v <= 100:
                overall = v
                break
        if overall is None and sel is not None and sel > 0:
            overall = sel if (selection_is_overall or rd is None or rd <= 1) else 30 * (rd - 1) + sel
        if overall is None:
            txt = str(p.get("draft_display_text") or p.get("draft_text") or "")
            m_rp = re.search(r"(?:rd|round)\s*(\d+)[^0-9]+(?:pk|pick)\s*(\d+)", txt, flags=re.I)
            if m_rp:
                rn, pn = int(m_rp.group(1)), int(m_rp.group(2))
                overall = pn if (pn > 30 or rn <= 1) else 30 * (rn - 1) + pn

        dob = None
        raw_dob = p.get("date_of_birth") or ""
        if len(raw_dob) >= 10:
            try:
                dob = date.fromisoformat(raw_dob[:10])
            except ValueError:
                dob = None

        exp = _safe_int(p.get("experience_years") or p.get("experience"))
        tid = str(p.get("team_id") or p.get("draft_team_id") or "").strip() or None
        pos = (p.get("position_abbreviation") or p.get("position") or "").strip().upper() or None
        names = {
            (p.get(c) or "").strip()
            for c in ("display_name", "full_name", "athlete_display_name", "name")
            if (p.get(c) or "").strip()
        }
        parsed[aid] = {
            "draft_year": dy,
            "overall_pick": overall,
            "dob": dob,
            "experience": exp,
            "team_id": tid,
            "position": pos,
            "names": names,
        }
    return parsed


def compute_prior_team_context(prior_folder, all_star_date, prior_births, prior_adj_by_id, prior_adj_rank_by_score):
    """Compute strictly Year-0 team win %, top-8 rotation age, 30+ vet minute share, and top-150 count."""
    schedule = list(read_csv(prior_folder / "games.csv"))
    reg_games = [g for g in schedule if regular_season_game(g)]
    game_dates = {g["game_id"]: g["game_date"] for g in reg_games}

    wins, gp, abbr_to_tid = Counter(), Counter(), {}
    for g in reg_games:
        hid, aid = str(g["home_id"]), str(g["away_id"])
        gp[hid] += 1
        gp[aid] += 1
        if g.get("home_abbreviation"):
            abbr_to_tid[g["home_abbreviation"].strip().upper()] = hid
        if g.get("away_abbreviation"):
            abbr_to_tid[g["away_abbreviation"].strip().upper()] = aid
        hw = str(g.get("home_winner") or "").strip().lower()
        aw = str(g.get("away_winner") or "").strip().lower()
        if hw == "true":
            wins[hid] += 1
        elif aw == "true":
            wins[aid] += 1
        else:
            hs, as_ = _safe_float(g.get("home_score")), _safe_float(g.get("away_score"))
            if hs is not None and as_ is not None:
                if hs > as_:
                    wins[hid] += 1
                elif as_ > hs:
                    wins[aid] += 1

    team_player_min = defaultdict(lambda: defaultdict(float))
    for gid, aid, _, stats, _, _, tid in player_box_scores(prior_folder, game_dates):
        mins = parse_minutes(stats.get("minutes"))
        if mins is not None and mins > 0 and tid:
            team_player_min[str(tid)][aid] += mins

    team_ctx = {}
    for tid, p_mins in team_player_min.items():
        if sum(p_mins.values()) <= 0:
            continue
        ranked_rot = sorted(p_mins.items(), key=lambda kv: -kv[1])
        top8_ids = {aid for aid, _ in ranked_rot[:8]}
        age_min = top8_age = top8_min = vet30_min = 0.0
        top150_cnt = 0
        for aid, mins in ranked_rot:
            score = prior_adj_by_id.get(aid)
            if score is not None and prior_adj_rank_by_score.get(score, 999) <= 150 and mins >= 500:
                top150_cnt += 1
            dob = prior_births.get(aid)
            if dob is not None and all_star_date is not None:
                age = (all_star_date - dob).days / 365.2425
                age_min += mins
                if age >= 30.0:
                    vet30_min += mins
                if aid in top8_ids:
                    top8_age += age * mins
                    top8_min += mins
        team_ctx[tid] = {
            "win_pct": wins[tid] / gp[tid] if gp[tid] > 0 else None,
            "top8_avg_age": top8_age / top8_min if top8_min > 0 else None,
            "vet30_min_share": vet30_min / age_min if age_min > 0 else None,
            "top150_count": top150_cnt,
        }
    return team_ctx, abbr_to_tid


def player_season_summary(player):
    games = player.total_games
    if not games:
        return None
    fppg = player.total_points / games
    tot_min = player.box_score_totals.get("minutes")
    mpg, fppm = (tot_min / games, player.total_points / tot_min) if (tot_min is not None and tot_min > 0) else (None, None)
    start_pct = (100.0 * player.starts / games) if not player.games_without_start_status else None
    return {"games": games, "fppg": fppg, "mpg": mpg, "fppm": fppm, "start_pct": start_pct}


def load_season_daily_logs(season_folder, needed_ids):
    """Load daily regular-season fantasy points, first-appearance teams, and box-score fallback positions."""
    schedule = list(read_csv(season_folder / "games.csv"))
    game_dates = {g["game_id"]: g["game_date"] for g in schedule if regular_season_game(g)}
    dates_sorted = sorted(set(game_dates.values()))
    date_to_idx = {d: i for i, d in enumerate(dates_sorted)}

    daily = defaultdict(dict)
    first_seen_team = {}
    pos_counts = defaultdict(Counter)
    for gid, aid, _, fpts, pos, _, tid in player_lines(season_folder, game_dates):
        gdate = game_dates[gid]
        if tid and (aid not in first_seen_team or gdate < first_seen_team[aid][0]):
            first_seen_team[aid] = (gdate, str(tid))
        if aid in needed_ids:
            di = date_to_idx[gdate]
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
    return len(dates_sorted), daily, {aid: tid for aid, (_, tid) in first_seen_team.items()}, fallback_pos


def load_season_cache(data):
    seasons = {int(p.name[:4]): p for p in data.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}", p.name)}
    raw_cache, sum_cache, lb_cache = {}, {}, {}

    def get_season(year):
        if year not in raw_cache:
            folder = seasons.get(year)
            if folder is None or not (folder / "games.csv").is_file() or not (folder / "player_box_scores.csv").is_file():
                raw_cache[year] = None
            else:
                raw_cache[year] = read_season(folder, split_at_all_star=True, include_box_scores=True)
        return raw_cache[year]

    def get_summaries(year):
        if year not in sum_cache:
            res = get_season(year)
            sum_cache[year] = (
                {aid: s for aid, p in res[1].items() if (s := player_season_summary(p)) is not None}
                if res is not None else None
            )
        return sum_cache[year]

    def get_leaderboard(year):
        if year not in lb_cache:
            res = get_season(year)
            lb_cache[year] = compute_season_adj_leaderboard(res[1]) if res is not None else None
        return lb_cache[year]

    return seasons, get_season, get_summaries, get_leaderboard


def build_unified_season_rows(
    *,
    data,
    prior_year,
    prior_folder,
    target_season,
    adp_path,
    prior_res,
    prior_sum,
    prev2_sum,
    prior_lb_tuple,
    next_res,
    next_sum,
    next_lb_tuple,
    played_before,
    history_available,
    max_pool_adp,
    live_mode=False,
    excluded_keys=(),
):
    """Build unified full-pool rows (returning_vet + rookie + returning_absence) for one draft season."""
    all_star_date, prior_players = prior_res
    next_players = next_res[1] if next_res is not None else {}

    prior_waiver, prior_adj_by_id, prior_raw_by_id, prior_adj_lb, prior_adj_rank_by_score, prior_raw_rank_by_score = prior_lb_tuple
    try:
        prior_births = load_birth_dates(data, prior_year + 1)
    except Exception:
        prior_births = {}

    try:
        next_profiles = load_player_profiles(data, prior_year + 2)
    except Exception:
        next_profiles = {}
    parsed_profiles = parse_draft_info_from_profiles(next_profiles)

    team_ctx, abbr_to_tid = compute_prior_team_context(
        prior_folder, all_star_date, prior_births, prior_adj_by_id, prior_adj_rank_by_score,
    )

    debut_teams = {}
    if not live_mode and (data / target_season / "games.csv").is_file() and (data / target_season / "player_box_scores.csv").is_file():
        _, _, debut_teams, _ = load_season_daily_logs(data / target_season, set())

    ids_prior = defaultdict(set)
    for aid, p in prior_players.items():
        for n in p.names:
            ids_prior[name_key(n)].add(aid)
    ids_next_played = defaultdict(set)
    for aid, p in next_players.items():
        for n in p.names:
            ids_next_played[name_key(n)].add(aid)
    ids_next_profile = defaultdict(set)
    for aid, prof in parsed_profiles.items():
        for n in prof["names"]:
            ids_next_profile[name_key(n)].add(aid)

    adp_entries = read_adp_entries(adp_path)
    draft_date_ref = date(prior_year + 1, 10, 15)
    excluded_set = set(excluded_keys)
    seen_ids = set()
    rows, removed = [], []
    counts = Counter()

    for entry in adp_entries:
        if len(rows) >= int(max_pool_adp) and entry["adp"] > max_pool_adp:
            continue
        if entry["key"] in excluded_set:
            removed.append(entry["player"])
            continue

        key = entry["key"]
        adp = entry["adp"]
        mkt_exp_adj = adp_slot_points(prior_adj_lb, adp)

        # Classify player type & resolve athlete_id
        aid = None
        player_type = "returning_absence"

        p_ids = ids_prior[key]
        if len(p_ids) == 1 and next(iter(p_ids)) not in seen_ids:
            (aid,) = p_ids
            if aid in prior_sum:
                player_type = "returning_vet"
        else:
            # Non-Year-0 player: check next_players first, then player_core profiles
            cand_ids = ids_next_played.get(key, set())
            if len(cand_ids) == 1:
                (cand_aid,) = cand_ids
                if cand_aid not in seen_ids:
                    aid = cand_aid
            else:
                prof_cands = [c for c in ids_next_profile.get(key, set()) if c not in seen_ids]
                if len(prof_cands) > 1:
                    recent = [c for c in prof_cands if parsed_profiles[c]["draft_year"] in (prior_year + 1, prior_year)]
                    if len(recent) == 1:
                        prof_cands = recent
                if len(prof_cands) == 1:
                    aid = prof_cands[0]

            if aid is not None and aid not in played_before:
                prof = parsed_profiles.get(aid, {})
                dy = prof.get("draft_year")
                exp = prof.get("experience")
                # Guard against returning veterans if career history wasn't complete
                is_true_rookie = True
                if not history_available:
                    if (dy is not None and dy < prior_year - 1) or (exp is not None and exp > 1):
                        is_true_rookie = False
                elif aid not in next_players and dy is not None and dy < prior_year - 1:
                    is_true_rookie = False
                if is_true_rookie:
                    player_type = "rookie"
            elif live_mode and aid is None:
                # Pre-season live fallback when next-year player_core isn't published yet
                player_type = "rookie"

        if aid is not None:
            seen_ids.add(aid)
        counts[player_type] += 1
        market_rank = len(rows) + 1

        row = {
            "prior_season": season_label(prior_year),
            "draft_season": target_season,
            "player": entry["player"],
            "athlete_id": aid or f"unmatched_{key}",
            "player_type": player_type,
            "adp": adp,
            "market_rank": market_rank,
            "log_adp": math.log(max(adp, 1.0)),
            "mkt_expected_adj_fppg": mkt_exp_adj,
            "prior_waiver_fppg": prior_waiver,
            "adp_platform_std": entry["adp_platform_std"],
            "espn_vs_avg": entry["espn_vs_avg"],
            "adp_positions": entry["adp_positions"],
            # Shared feature defaults (populated by player_type below)
            "age_at_prior_break": None, "age_under_24": None, "age_over_30": None,
            "prior_games": None, "prior_fppg": None, "prior_adj_fppg": None,
            "prior_raw_rank": None, "prior_mpg": None, "prior_fppm": None, "prior_start_pct": None,
            "prior_minus_mkt_capped": None, "market_collapse_excess": 0.0,
            "yoy_fppg_delta": None, "yoy_mpg_delta": None, "two_yr_weighted_fppg": None,
            "two_yr_games_avg": None, "no_prev2_season": None,
            "draft_pick": None, "log_draft_pick": None, "is_redshirt": None, "rookie_age": None,
            "team_top8_avg_age": None, "team_prior_win_pct": None,
            "team_vet30_min_share": None, "team_prior_top150_count": None,
        }

        if player_type == "returning_vet":
            pr = prior_sum[aid]
            p2 = prev2_sum.get(aid) if prev2_sum is not None else None
            dob = prior_births.get(aid)
            age = (
                all_star_date.year - dob.year - ((all_star_date.month, all_star_date.day) < (dob.month, dob.day))
                if (dob is not None and all_star_date is not None) else None
            )
            # Raw prior FPPG vs ADP-implied AdjFPPG, gated by >=20-slot drop from prior raw rank:
            # catches BOTH playoff Achilles tears (Tatum/Haliburton/Lillard) AND mid-season tears (Dejounte Murray)
            # while never flagging healthy top-5 stars (Giannis)
            prior_raw_rank = prior_raw_rank_by_score.get(prior_raw_by_id.get(aid, 0.0), 500)
            raw_gap = pr["fppg"] - mkt_exp_adj
            slot_drop = adp - prior_raw_rank
            is_collapse = slot_drop >= COLLAPSE_MIN_SLOT_DROP and raw_gap > COLLAPSE_THRESHOLD
            row.update({
                "age_at_prior_break": age,
                "age_under_24": max(0.0, 24.0 - age) if age is not None else None,
                "age_over_30": max(0.0, age - 30.0) if age is not None else None,
                "prior_games": pr["games"],
                "prior_fppg": pr["fppg"],
                "prior_adj_fppg": prior_adj_by_id.get(aid, prior_waiver),
                "prior_raw_rank": prior_raw_rank,
                "prior_mpg": pr["mpg"],
                "prior_fppm": pr["fppm"],
                "prior_start_pct": pr["start_pct"],
                "prior_minus_mkt_capped": max(-10.0, min(COLLAPSE_THRESHOLD, raw_gap)),
                "market_collapse_excess": (raw_gap - COLLAPSE_THRESHOLD) if is_collapse else 0.0,
                "yoy_fppg_delta": (pr["fppg"] - p2["fppg"]) if p2 else None,
                "yoy_mpg_delta": (pr["mpg"] - p2["mpg"]) if (p2 and pr["mpg"] is not None and p2["mpg"] is not None) else None,
                "two_yr_weighted_fppg": (0.65 * pr["fppg"] + 0.35 * p2["fppg"]) if p2 else pr["fppg"],
                "two_yr_games_avg": (0.5 * (pr["games"] + p2["games"])) if p2 else float(pr["games"]),
                "no_prev2_season": 0.0 if p2 else (1.0 if prev2_sum is not None else None),
            })

        elif player_type == "rookie":
            prof = parsed_profiles.get(aid, {}) if aid is not None else {}
            if live_mode:
                tid = prof.get("team_id") or (abbr_to_tid.get(entry["adp_team_abbr"]) if entry["adp_team_abbr"] else None)
            else:
                tid = debut_teams.get(aid) or prof.get("team_id") or (abbr_to_tid.get(entry["adp_team_abbr"]) if entry["adp_team_abbr"] else None)
            tctx = team_ctx.get(str(tid)) if tid is not None else None

            overall_pick = prof.get("overall_pick")
            eff_pick = float(overall_pick) if overall_pick is not None else (65.0 if prof else None)
            dy = prof.get("draft_year")
            dob = prof.get("dob")
            row.update({
                "draft_pick": overall_pick,
                "log_draft_pick": math.log(eff_pick) if eff_pick is not None else None,
                "is_redshirt": (1.0 if dy < prior_year + 1 else 0.0) if dy is not None else None,
                "rookie_age": round((draft_date_ref - dob).days / 365.2425, 2) if dob is not None else None,
                "team_top8_avg_age": round(tctx["top8_avg_age"], 2) if (tctx and tctx["top8_avg_age"] is not None) else None,
                "team_prior_win_pct": round(tctx["win_pct"], 4) if (tctx and tctx["win_pct"] is not None) else None,
                "team_vet30_min_share": round(tctx["vet30_min_share"], 4) if (tctx and tctx["vet30_min_share"] is not None) else None,
                "team_prior_top150_count": tctx["top150_count"] if tctx else None,
            })

        # Outcomes (for completed historical seasons)
        if next_lb_tuple is not None:
            next_waiver, next_adj_by_id, next_raw_by_id, next_adj_lb, next_adj_rank_by_score, _ = next_lb_tuple
            after = next_players.get(aid) if aid is not None else None
            next_games = after.total_games if after else 0
            raw_next_fppg = next_raw_by_id.get(aid, 0.0) if aid is not None else 0.0
            next_adj_fppg = next_adj_by_id.get(aid, next_waiver) if aid is not None else next_waiver
            if next_games > 0 and next_adj_fppg in next_adj_rank_by_score:
                next_rank = next_adj_rank_by_score[next_adj_fppg]
            else:
                next_rank = sum(1 for v in next_adj_lb if v > next_waiver) + 1
            slot_adj = adp_slot_points(next_adj_lb, adp)
            ns = next_sum.get(aid) if (next_sum is not None and aid is not None) else None
            row.update({
                "next_games": next_games,
                "next_avail_frac": min(float(next_games), REF_GAMES) / REF_GAMES,
                "next_raw_fppg": round(raw_next_fppg, 3),
                "next_adj_fppg": round(next_adj_fppg, 3),
                "total_fpts": after.total_points if after else 0,
                "fppg_above_adp": round(next_adj_fppg - slot_adj, 3),
                "next_rank": next_rank,
                "adp_slots_gained": round(adp - next_rank, 3),
                "beat_adp": next_rank < adp,
                "next_mpg": ns["mpg"] if ns else None,
                "next_fppm": ns["fppm"] if ns else None,
            })

        rows.append(row)

    return rows, counts, removed


def build_completed_dataset(data, max_pool_adp=225.0):
    seasons, get_season, get_summaries, get_leaderboard = load_season_cache(data)
    played_before = set()
    archive_through = FIRST_ARCHIVE_YEAR - 1
    history_available = True
    all_rows = []

    for year in sorted(seasons):
        following = seasons.get(year + 1)
        if following is None or not (following / "adp.csv").is_file():
            continue
        prior_res, next_res = get_season(year), get_season(year + 1)
        if prior_res is None or next_res is None:
            continue

        if history_available and year >= FIRST_ARCHIVE_YEAR:
            try:
                for ending_year in range(archive_through + 1, year + 2):
                    played_before.update(players_with_appearances(data, ending_year))
                archive_through = year + 1
            except Exception as err:
                print(f"Note: ESPN career appearance history unavailable ({err}); using prior-season + draft-year fallback")
                history_available = False

        played_before.update(prior_res[1].keys())

        rows, counts, _ = build_unified_season_rows(
            data=data,
            prior_year=year,
            prior_folder=seasons[year],
            target_season=following.name,
            adp_path=following / "adp.csv",
            prior_res=prior_res,
            prior_sum=get_summaries(year),
            prev2_sum=get_summaries(year - 1) if (year - 1) in seasons else None,
            prior_lb_tuple=get_leaderboard(year),
            next_res=next_res,
            next_sum=get_summaries(year + 1),
            next_lb_tuple=get_leaderboard(year + 1),
            played_before=played_before,
            history_available=history_available,
            max_pool_adp=max_pool_adp,
            live_mode=False,
        )
        print(
            f"{seasons[year].name} -> {following.name}: {len(rows)} full-pool ADP players "
            f"({counts['returning_vet']} vets, {counts['rookie']} rookies, {counts['returning_absence']} returning/unmatched)"
        )
        all_rows.extend(rows)

    return all_rows


def to_matrix(rows, features):
    return np.array(
        [[np.nan if row.get(f) is None else float(row[f]) for f in features] for row in rows],
        dtype=float,
    )


def active_features(training, features):
    return [f for f in features if len({row[f] for row in training if row.get(f) is not None}) > 1]


def _make_spline_ridge(feat_list):
    spline_targets = {"log_adp", "mkt_expected_adj_fppg", "age_at_prior_break"}
    s_cols = [i for i, f in enumerate(feat_list) if f in spline_targets]
    o_cols = [i for i in range(len(feat_list)) if i not in s_cols]
    transformers = []
    if s_cols:
        transformers.append(("splines", make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            SplineTransformer(n_knots=4, degree=2, include_bias=False),
            StandardScaler(),
        ), s_cols))
    if o_cols:
        transformers.append(("linear", make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
        ), o_cols))
    return make_pipeline(ColumnTransformer(transformers), RidgeCV(alphas=RIDGE_ALPHAS))


def _fit_predict_vet_huber(tr_vets, ev_vets):
    feats = active_features(tr_vets, VET_CLEAN_FEATURES)
    pipe = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        HuberRegressor(alpha=25.0, epsilon=1.35, max_iter=1000),
    )
    pipe.fit(to_matrix(tr_vets, feats), [float(r["fppg_above_adp"]) for r in tr_vets])
    preds = pipe.predict(to_matrix(ev_vets, feats)) if ev_vets else np.array([], dtype=float)
    return preds, dict(zip(feats, map(float, pipe.named_steps["huberregressor"].coef_)))


def _fit_predict_vet_decomposed(tr_vets, ev_vets):
    if not ev_vets:
        return np.array([], dtype=float)
    rate_rows = [r for r in tr_vets if r.get("next_mpg") is not None and r.get("next_fppm") is not None]
    mpg_f = active_features(rate_rows, VET_MPG_FEATURES)
    fppm_f = active_features(rate_rows, VET_FPPM_FEATURES)
    avail_f = active_features(tr_vets, VET_AVAIL_FEATURES)

    X_mpg, y_mpg = to_matrix(rate_rows, mpg_f), np.array([float(r["next_mpg"]) for r in rate_rows], dtype=float)
    X_fppm, y_fppm = to_matrix(rate_rows, fppm_f), np.array([float(r["next_fppm"]) for r in rate_rows], dtype=float)
    X_av, y_av = to_matrix(tr_vets, avail_f), np.array([float(r["next_avail_frac"]) for r in tr_vets], dtype=float)

    groups = np.array([r["draft_season"] for r in tr_vets])
    n_s = len(set(groups))
    oof_mpg = np.zeros(len(tr_vets), dtype=float)
    oof_fppm = np.zeros(len(tr_vets), dtype=float)
    oof_av = np.zeros(len(tr_vets), dtype=float)

    if n_s >= 2:
        for tr_i, va_i in GroupKFold(n_splits=min(n_s, 5)).split(tr_vets, y_av, groups):
            tr_a = [tr_vets[i] for i in tr_i]
            va_a = [tr_vets[i] for i in va_i]
            tr_r = [r for r in tr_a if r.get("next_mpg") is not None and r.get("next_fppm") is not None]
            m_m = _make_spline_ridge(mpg_f).fit(to_matrix(tr_r, mpg_f), [float(r["next_mpg"]) for r in tr_r])
            f_m = _make_spline_ridge(fppm_f).fit(to_matrix(tr_r, fppm_f), [float(r["next_fppm"]) for r in tr_r])
            a_m = _make_spline_ridge(avail_f).fit(to_matrix(tr_a, avail_f), [float(r["next_avail_frac"]) for r in tr_a])
            oof_mpg[va_i] = np.clip(m_m.predict(to_matrix(va_a, mpg_f)), 5.0, 42.0)
            oof_fppm[va_i] = np.clip(f_m.predict(to_matrix(va_a, fppm_f)), 0.3, 2.5)
            oof_av[va_i] = np.clip(a_m.predict(to_matrix(va_a, avail_f)), 0.0, 1.0)
    else:
        oof_mpg = np.clip(_make_spline_ridge(mpg_f).fit(X_mpg, y_mpg).predict(to_matrix(tr_vets, mpg_f)), 5.0, 42.0)
        oof_fppm = np.clip(_make_spline_ridge(fppm_f).fit(X_fppm, y_fppm).predict(to_matrix(tr_vets, fppm_f)), 0.3, 2.5)
        oof_av = np.clip(_make_spline_ridge(avail_f).fit(X_av, y_av).predict(X_av), 0.0, 1.0)

    tr_w = np.array([float(r["prior_waiver_fppg"]) for r in tr_vets])
    tr_struct = oof_av * (oof_mpg * oof_fppm) + (1.0 - oof_av) * tr_w

    ev_mpg = np.clip(_make_spline_ridge(mpg_f).fit(X_mpg, y_mpg).predict(to_matrix(ev_vets, mpg_f)), 5.0, 42.0)
    ev_fppm = np.clip(_make_spline_ridge(fppm_f).fit(X_fppm, y_fppm).predict(to_matrix(ev_vets, fppm_f)), 0.3, 2.5)
    ev_av = np.clip(_make_spline_ridge(avail_f).fit(X_av, y_av).predict(to_matrix(ev_vets, avail_f)), 0.0, 1.0)
    ev_w = np.array([float(r["prior_waiver_fppg"]) for r in ev_vets])
    ev_struct = ev_av * (ev_mpg * ev_fppm) + (1.0 - ev_av) * ev_w

    tr_mkt = np.array([float(r["mkt_expected_adj_fppg"]) for r in tr_vets])
    ev_mkt = np.array([float(r["mkt_expected_adj_fppg"]) for r in ev_vets])
    tr_la = np.array([float(r["log_adp"]) for r in tr_vets])
    ev_la = np.array([float(r["log_adp"]) for r in ev_vets])
    tr_c = np.array([float(r["market_collapse_excess"]) for r in tr_vets])
    ev_c = np.array([float(r["market_collapse_excess"]) for r in ev_vets])

    meta_tr = np.column_stack([tr_struct - tr_mkt, (tr_struct - tr_mkt) * tr_la, tr_la, tr_c])
    meta_ev = np.column_stack([ev_struct - ev_mkt, (ev_struct - ev_mkt) * ev_la, ev_la, ev_c])
    meta = make_pipeline(StandardScaler(), HuberRegressor(alpha=10.0, max_iter=1000))
    meta.fit(meta_tr, [float(r["fppg_above_adp"]) for r in tr_vets])
    return meta.predict(meta_ev)


def _huber_loss_1d(err, delta=3.0):
    a = np.abs(err)
    return float(np.mean(np.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))))


def _fit_two_model_weights(oof_h, oof_d, y_true, min_weight=0.15, shrink_l2=0.15):
    best_w, best_obj = 0.5, float("inf")
    for w in np.linspace(min_weight, 1.0 - min_weight, 29):
        pred = w * oof_h + (1.0 - w) * oof_d
        obj = _huber_loss_1d(y_true - pred) + shrink_l2 * ((w - 0.5) ** 2)
        if obj < best_obj:
            best_obj, best_w = obj, float(w)
    return best_w, 1.0 - best_w


def _fit_predict_rookie_huber(tr_rk, ev_rk):
    """Fit lean rookie_full_huber strictly on earlier seasons' rookies; return (full_preds, flat_bias, coefs)."""
    if not tr_rk:
        return np.zeros(len(ev_rk), dtype=float), 0.0, {}
    y_res = np.array([float(r["fppg_above_adp"]) for r in tr_rk], dtype=float)
    flat_bias = float(np.median(y_res))
    if not ev_rk:
        return np.array([], dtype=float), flat_bias, {}

    # Compute zero-leakage Rookie Hype Tax (adp_minus_pick_expected) strictly on tr_rk
    tr_work = [r.copy() for r in tr_rk]
    ev_work = [r.copy() for r in ev_rk]
    tr_valid = [r for r in tr_work if r.get("log_draft_pick") is not None]
    if len(tr_valid) >= 8:
        pick_mod = HuberRegressor(alpha=5.0, max_iter=500).fit(
            np.array([[float(r["log_draft_pick"])] for r in tr_valid], dtype=float),
            np.array([float(r["next_adj_fppg"]) for r in tr_valid], dtype=float),
        )
        fallback = float(np.median([float(r["next_adj_fppg"]) for r in tr_valid]))
        for grp in (tr_work, ev_work):
            for r in grp:
                pe = float(pick_mod.predict([[float(r["log_draft_pick"])]])[0]) if r.get("log_draft_pick") is not None else fallback
                r["adp_minus_pick_expected"] = float(r["mkt_expected_adj_fppg"]) - pe
    else:
        for grp in (tr_work, ev_work):
            for r in grp:
                r["adp_minus_pick_expected"] = None

    feats = active_features(tr_work, ROOKIE_FEATURES)
    if len(tr_work) < 12 or not feats:
        return np.full(len(ev_rk), flat_bias, dtype=float), flat_bias, {}

    pipe = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        HuberRegressor(alpha=30.0, epsilon=1.35, max_iter=1000),
    )
    pipe.fit(to_matrix(tr_work, feats), y_res)
    preds = pipe.predict(to_matrix(ev_work, feats))
    coefs = dict(zip(feats, map(float, pipe.named_steps["huberregressor"].coef_)))
    return preds, flat_bias, coefs


def fit_and_score_full_pool(training_all, evaluated_all, max_train_adp=175.0, max_rookie_train_adp=200.0):
    """Fit veteran 2-model ensemble and rookie_full_huber strictly on training_all and score evaluated_all."""
    tr_vets = [r for r in training_all if r["player_type"] == "returning_vet" and r["adp"] <= max_train_adp]
    tr_rk = [r for r in training_all if r["player_type"] == "rookie" and r["adp"] <= max_rookie_train_adp]

    ev_vet_idx = [i for i, r in enumerate(evaluated_all) if r["player_type"] == "returning_vet"]
    ev_rk_idx = [i for i, r in enumerate(evaluated_all) if r["player_type"] == "rookie"]
    ev_abs_idx = [i for i, r in enumerate(evaluated_all) if r["player_type"] == "returning_absence"]

    ev_vets = [evaluated_all[i] for i in ev_vet_idx]
    ev_rk = [evaluated_all[i] for i in ev_rk_idx]

    # 1. Inner OOF for 2-model veteran stacking weights
    y_vet = np.array([float(r["fppg_above_adp"]) for r in tr_vets], dtype=float)
    groups = np.array([r["draft_season"] for r in tr_vets])
    n_s = len(set(groups))
    oof_h = np.zeros(len(tr_vets), dtype=float)
    oof_d = np.zeros(len(tr_vets), dtype=float)
    if n_s >= 2:
        for tr_i, va_i in GroupKFold(n_splits=min(n_s, 5)).split(tr_vets, y_vet, groups):
            tr_sub = [tr_vets[i] for i in tr_i]
            va_sub = [tr_vets[i] for i in va_i]
            oof_h[va_i], _ = _fit_predict_vet_huber(tr_sub, va_sub)
            oof_d[va_i] = _fit_predict_vet_decomposed(tr_sub, va_sub)
    else:
        oof_h, _ = _fit_predict_vet_huber(tr_vets, tr_vets)
        oof_d = _fit_predict_vet_decomposed(tr_vets, tr_vets)

    w_h, w_d = _fit_two_model_weights(oof_h, oof_d, y_vet)
    ev_h, vet_coefs = _fit_predict_vet_huber(tr_vets, ev_vets)
    ev_d = _fit_predict_vet_decomposed(tr_vets, ev_vets)
    ev_vet_equal = 0.5 * ev_h + 0.5 * ev_d
    ev_vet_stacked = w_h * ev_h + w_d * ev_d

    # Neutral peer baseline so returning_absence players (and vet_only_rk_neutral rookies) sit cleanly at market peer level
    vet_peer_center = float(np.median(ev_vet_stacked)) if len(ev_vet_stacked) else 0.0

    # 2. Rookie model
    ev_rk_huber, rk_flat_bias, rk_coefs = _fit_predict_rookie_huber(tr_rk, ev_rk)

    n = len(evaluated_all)
    scores = {s: np.zeros(n, dtype=float) for s in STRATEGIES}

    for pos, idx in enumerate(ev_vet_idx):
        scores["vet_only_rk_neutral"][idx] = ev_vet_stacked[pos]
        scores["unified_flat_rk"][idx] = ev_vet_stacked[pos]
        scores["unified_equal"][idx] = ev_vet_equal[pos]
        scores["unified_stacked"][idx] = ev_vet_stacked[pos]

    for pos, idx in enumerate(ev_rk_idx):
        scores["vet_only_rk_neutral"][idx] = vet_peer_center
        scores["unified_flat_rk"][idx] = rk_flat_bias
        scores["unified_equal"][idx] = ev_rk_huber[pos]
        scores["unified_stacked"][idx] = ev_rk_huber[pos]

    for idx in ev_abs_idx:
        scores["vet_only_rk_neutral"][idx] = vet_peer_center
        scores["unified_flat_rk"][idx] = vet_peer_center
        scores["unified_equal"][idx] = vet_peer_center
        scores["unified_stacked"][idx] = vet_peer_center

    meta = {
        "vet_weights": (w_h, w_d),
        "vet_coefs": vet_coefs,
        "rookie_coefs": rk_coefs,
        "rookie_flat_bias": rk_flat_bias,
        "vet_peer_center": vet_peer_center,
    }
    return scores, meta


def build_global_board(season_rows, strategy, lam):
    scored = []
    for item in season_rows:
        adj = 0.0
        if strategy != "market":
            adj = lam * max(-RESIDUAL_CLIP, min(RESIDUAL_CLIP, float(item[f"{strategy}_score"])))
        scored.append((item["mkt_expected_adj_fppg"] + adj, adj, item))
    scored.sort(key=lambda t: (-t[0], t[2]["market_rank"]))
    return scored


def tune_global_lambdas(cv_rows):
    by_season = defaultdict(list)
    for r in cv_rows:
        by_season[r["draft_season"]].append(r)
    seasons = sorted(s for s, grp in by_season.items() if len(grp) >= BOARD_SIZE)
    best_lambdas = {"market": 0.0}
    for strat in STRATEGIES:
        if strat == "market":
            continue
        best_lam, best_score = 0.65, (-1e9, -1e9)
        for lam in LAMBDA_GRID:
            fppgs, hits = [], 0
            for s in seasons:
                for _, _, item in build_global_board(by_season[s], strat, lam)[:BOARD_SIZE]:
                    fppgs.append(item["next_adj_fppg"])
                    if item["next_rank"] <= BOARD_SIZE:
                        hits += 1
            cand = (mean(fppgs) if fppgs else -1e9, hits)
            if cand > best_score:
                best_score, best_lam = cand, lam
        best_lambdas[strat] = best_lam
    return best_lambdas


def paired_bootstrap_ci(diffs, n_boot=2000, seed=42):
    valid = [d for d in diffs if d is not None and math.isfinite(d)]
    if len(valid) < 2:
        return (valid[0] if valid else None), None, None, None
    rng = random.Random(seed)
    n = len(valid)
    boots = [mean(rng.choices(valid, k=n)) for _ in range(n_boot)]
    boots.sort()
    p_val = sum(b <= 0 for b in boots) / n_boot
    return mean(valid), boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)], p_val


def evaluate_global_board(scored_rows, split_label, strategy, lam):
    by_season = defaultdict(list)
    for r in scored_rows:
        by_season[r["draft_season"]].append(r)
    seasons = sorted(s for s, grp in by_season.items() if len(grp) >= BOARD_SIZE)
    market_fppg = {
        s: mean(x["next_adj_fppg"] for x in sorted(by_season[s], key=lambda r: r["market_rank"])[:BOARD_SIZE])
        for s in seasons
    }

    board_items, added, dropped, season_lifts = [], [], [], []
    rk_on_board = 0
    for s in seasons:
        ordered = build_global_board(by_season[s], strategy, lam)
        top150, outside = ordered[:BOARD_SIZE], ordered[BOARD_SIZE:]
        season_lifts.append(mean(item["next_adj_fppg"] for _, _, item in top150) - market_fppg[s])
        for rank, (_, _, item) in enumerate(top150, start=1):
            board_items.append((rank, item))
            if item["player_type"] == "rookie":
                rk_on_board += 1
            if item["market_rank"] > BOARD_SIZE:
                added.append(item)
        for _, _, item in outside:
            if item["market_rank"] <= BOARD_SIZE:
                dropped.append(item)

    _, _, _, p_val = paired_bootstrap_ci(season_lifts) if strategy != "market" else (None, None, None, None)
    return {
        "split": split_label,
        "strategy": strategy,
        "lambda": lam,
        "seasons": len(seasons),
        "board_slots": len(board_items),
        "rookies_on_board": rk_on_board,
        "zero_game_busts": sum(x["next_games"] == 0 for _, x in board_items),
        "actual_top_150_captured": sum(x["next_rank"] <= BOARD_SIZE for _, x in board_items),
        "top_150_capture_pct": round(100 * sum(x["next_rank"] <= BOARD_SIZE for _, x in board_items) / len(board_items), 2),
        "board_mean_adj_fppg": round(mean(x["next_adj_fppg"] for _, x in board_items), 3),
        "mean_abs_rank_error": round(mean(abs(rank - x["next_rank"]) for rank, x in board_items), 2),
        "swaps_count": len(added),
        "added_top150_pct": round(100 * sum(x["next_rank"] <= BOARD_SIZE for x in added) / len(added), 1) if added else None,
        "dropped_top150_pct": round(100 * sum(x["next_rank"] <= BOARD_SIZE for x in dropped) / len(dropped), 1) if dropped else None,
        "added_minus_dropped_fppg": round(mean(x["next_adj_fppg"] for x in added) - mean(x["next_adj_fppg"] for x in dropped), 2) if (added and dropped) else None,
        "vs_market_p_value": round(p_val, 4) if p_val is not None else None,
    }


def score_roster_espn(roster, bundle):
    """Day-by-day exact ESPN 10-starter (PG,SG,SF,PF,C,G,F,3xUTIL) + 3 Bench active lineup scorer."""
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
    played_lists, fpts_lists, masks10 = bundle["played_lists"], bundle["fpts_lists"], bundle["masks10"]

    benched_pts = benched_g = 0
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


def run_espn_simulation(data, scored_rows, lambdas, *, n_teams=12, n_rounds=13, trials_per_slot=12, adp_var=2.0, max_centers=4, seed=None):
    """Run 12-team x 13-round ESPN snake draft simulation across the FULL pool (vets + rookies + returning)."""
    if seed is None:
        seed = random.SystemRandom().randint(1, 999_999)
    rng = random.Random(seed)

    by_season = defaultdict(list)
    for r in scored_rows:
        by_season[r["draft_season"]].append(r)

    total_picks = n_teams * n_rounds
    seasons = sorted(s for s, grp in by_season.items() if len(grp) >= total_picks)
    if not seasons:
        raise ValueError(f"no season has at least {total_picks} full-pool players for simulation")

    # Build daily scoring bundles
    bundles = {}
    for s in seasons:
        players = sorted(by_season[s], key=lambda r: (r["adp"], r["market_rank"], r["athlete_id"]))
        s_folder = data / s
        needed_ids = {p["athlete_id"] for p in players}
        has_daily = (s_folder / "games.csv").is_file() and (s_folder / "player_box_scores.csv").is_file()
        if has_daily:
            n_days, daily, _, fallback_pos = load_season_daily_logs(s_folder, needed_ids)
        else:
            n_days, daily, fallback_pos = 0, {}, {}

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
            prio.append(p["next_raw_fppg"] if p["next_games"] > 0 else -1.0)
            if has_daily:
                day_map = daily.get(aid, {})
                for d, pts in day_map.items():
                    played_mat[idx, d] = 1
                    fpts_mat[idx, d] = pts
                p["total_fpts"] = int(sum(day_map.values()))
            tot_fpts.append(p["total_fpts"])

        bundles[s] = {
            "split": players[0]["split"],
            "players": players,
            "has_daily": has_daily,
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
    sample_slot = rng.randrange(n_teams)
    sample_trial = rng.randrange(trials_per_slot)
    sample_season = seasons[-1]
    sample_info = None

    records_split = defaultdict(list)
    records_season = defaultdict(list)

    for s in seasons:
        b = bundles[s]
        split = b["split"]
        players = b["players"]
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
                    masks7, is_c = b["masks7"], b["is_center"]

                    for pick_num, (rnd, team) in enumerate(pick_order, start=1):
                        board = hero_orders[strat] if team == slot else (straight_order if rnd == 1 else opp_noisy[team])
                        roster = rosters[team]
                        rem_after = n_rounds - len(roster) - 1
                        cur_c, cur_m = c_cnt[team], m_sz[team]
                        choice, new_m = None, cur_m
                        for idx in board:
                            if taken[idx] or (is_c[idx] and cur_c >= max_centers):
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
                                (idx for idx in board if not taken[idx] and not (is_c[idx] and cur_c >= max_centers)),
                                next(idx for idx in board if not taken[idx]),
                            )
                            new_m = max_matching_7(roster + [choice], masks7)
                        taken[choice] = True
                        roster.append(choice)
                        if is_c[choice]:
                            c_cnt[team] += 1
                        m_sz[team] = new_m
                        if team == slot:
                            hero_picks.append((rnd, pick_num, choice))

                    afs, rfs, bgs, bps = [], [], [], []
                    for t in range(n_teams):
                        af, rf, bg, bp = score_roster_espn(rosters[t], b)
                        afs.append(af); rfs.append(rf); bgs.append(bg); bps.append(bp)

                    hero_af = afs[slot]
                    field_afs = [afs[t] for t in range(n_teams) if t != slot]
                    field_af_avg = sum(field_afs) / len(field_afs)
                    better = sum(1 for x in field_afs if x > hero_af)
                    tied = sum(1 for x in field_afs if x == hero_af)
                    rank = 1.0 + better + 0.5 * tied
                    rec = {
                        "hero_af": hero_af,
                        "field_af": field_af_avg,
                        "diff": hero_af - field_af_avg,
                        "raw_diff": rfs[slot] - sum(rfs[t] for t in range(n_teams) if t != slot) / (n_teams - 1),
                        "rank": rank,
                        "won": (1.0 / (1 + tied)) if better == 0 else 0.0,
                        "top4": 1.0 if rank <= 4.0 else 0.0,
                        "beat_field": 1.0 if hero_af > field_af_avg else (0.5 if hero_af == field_af_avg else 0.0),
                        "hero_rk": sum(1 for i in rosters[slot] if players[i]["player_type"] == "rookie"),
                        "field_rk": sum(1 for t in range(n_teams) if t != slot for i in rosters[t] if players[i]["player_type"] == "rookie") / (n_teams - 1),
                        "hero_0g": sum(1 for i in rosters[slot] if players[i]["next_games"] == 0),
                        "field_0g": sum(1 for t in range(n_teams) if t != slot for i in rosters[t] if players[i]["next_games"] == 0) / (n_teams - 1),
                        "hero_bg": bgs[slot],
                        "hero_bp": bps[slot],
                        "hero_c": c_cnt[slot],
                    }
                    records_split[(split, strat)].append(rec)
                    records_split[("all", strat)].append(rec)
                    records_season[(s, split, strat)].append(rec)
                    if s == sample_season and slot == sample_slot and trial == sample_trial and strat == "unified_stacked":
                        sample_info = (s, slot + 1, b, hero_orders[strat], hero_picks, rosters[slot], rec)

    def agg(recs, fields):
        h_avg = mean(r["hero_af"] for r in recs)
        f_avg = mean(r["field_af"] for r in recs)
        d = h_avg - f_avg
        return {
            **fields,
            "drafts": len(recs),
            "hero_espn_fpts": round(h_avg, 1),
            "field_espn_fpts": round(f_avg, 1),
            "espn_diff": round(d, 1),
            "espn_lift_pct": round(100.0 * d / f_avg, 2) if f_avg else None,
            "raw_diff": round(mean(r["raw_diff"] for r in recs), 1),
            "avg_rank": round(mean(r["rank"] for r in recs), 2),
            "win_pct": round(100.0 * mean(r["won"] for r in recs), 1),
            "top4_pct": round(100.0 * mean(r["top4"] for r in recs), 1),
            "beat_field_pct": round(100.0 * mean(r["beat_field"] for r in recs), 1),
            "hero_rk": round(mean(r["hero_rk"] for r in recs), 2),
            "field_rk": round(mean(r["field_rk"] for r in recs), 2),
            "hero_0g": round(mean(r["hero_0g"] for r in recs), 2),
            "field_0g": round(mean(r["field_0g"] for r in recs), 2),
        }

    sim_summary, sim_seasons = [], []
    for split in ("cv", "holdout", "all"):
        for strat in STRATEGIES:
            if (split, strat) in records_split:
                sim_summary.append(agg(records_split[(split, strat)], {"split": split, "strategy": strat, "lambda": lambdas[strat]}))
    for s in seasons:
        sp = bundles[s]["split"]
        for strat in STRATEGIES:
            sim_seasons.append(agg(records_season[(s, sp, strat)], {"season": s, "split": sp, "strategy": strat, "lambda": lambdas[strat]}))

    write_csv(data / "sim_full_pool_summary.csv", sim_summary)
    write_csv(data / "sim_full_pool_by_season.csv", sim_seasons)

    print(f"\n================ FULL-POOL ESPN SNAKE DRAFT SIMULATOR (Vets + Rookies Together) ================")
    print(f"Rules: {n_teams}T x {n_rounds}R | PG,SG,SF,PF,C,G,F,3xUTIL + 3 BE | Max {max_centers} Primary C | R1 straight ADP, R2-13 Uniform[-{adp_var:g},+{adp_var:g}]")
    print(f"Trials: {trials_per_slot} x {n_teams} slots = {trials_per_slot * n_teams} drafts/season | Seed: {seed}")

    for split_label, title in (("cv", "CV SEASONS"), ("holdout", "HOLDOUT SEASONS"), ("all", "ALL SEASONS COMBINED")):
        rows = [r for r in sim_summary if r["split"] == split_label]
        if not rows:
            continue
        print(f"\n=== Full-Pool ESPN Simulation — {title} ({rows[0]['drafts']} drafts/strategy) ===")
        print(
            f"  {'Strategy':20s} {'Lam':>4s} | {'Hero ESPN':>9s} {'Field ESPN':>10s} {'Diff':>8s} {'Lift%':>6s} | "
            f"{'Rank':>5s} {'Win%':>5s} {'Top4%':>6s} {'BeatFld':>7s} | {'Rk(H/F)':>9s} {'0g(H/F)':>9s}"
        )
        print("  " + "-" * 116)
        for r in rows:
            print(
                f"  {r['strategy']:20s} {r['lambda']:4.2f} | "
                f"{r['hero_espn_fpts']:9.1f} {r['field_espn_fpts']:10.1f} {r['espn_diff']:+8.1f} {r['espn_lift_pct']:+5.2f}% | "
                f"{r['avg_rank']:5.2f} {r['win_pct']:4.1f}% {r['top4_pct']:5.1f}% {r['beat_field_pct']:6.1f}% | "
                f"{r['hero_rk']:4.2f}/{r['field_rk']:4.2f} {r['hero_0g']:4.2f}/{r['field_0g']:4.2f}"
            )

    print("\n=== Season-by-Season Full-Pool ESPN FPTS Diff (unified_stacked vs. market) ===")
    print(f"  {'Season':7s} {'Split':7s} | {'Market Diff':>11s} {'MktRk':>5s} | {'unified_stacked Diff':>20s} {'Rank':>5s} {'Win%':>6s} {'Top4%':>6s} {'BeatFld':>7s} {'Rk(H/F)':>9s}")
    print("  " + "-" * 100)
    by_sm = {(r["season"], r["strategy"]): r for r in sim_seasons}
    for s in seasons:
        m = by_sm[(s, "market")]
        u = by_sm[(s, "unified_stacked")]
        print(
            f"  {s:7s} {u['split']:7s} | {m['espn_diff']:+11.1f} {m['avg_rank']:5.2f} | "
            f"{u['espn_diff']:+20.1f} {u['avg_rank']:5.2f} {u['win_pct']:5.1f}% {u['top4_pct']:5.1f}% "
            f"{u['beat_field_pct']:6.1f}% {u['hero_rk']:4.2f}/{u['field_rk']:4.2f}"
        )

    if sample_info:
        s, slot_num, b, hero_order, picks, roster, rec = sample_info
        players = b["players"]
        rank_of = {idx: r for r, idx in enumerate(hero_order, start=1)}
        slot_of = assign_lineup_slots(roster, b["masks10"], b["prio"])
        print(f"\n=== Sample {s} Full-Pool ESPN Roster — Hero (unified_stacked) at Random Slot #{slot_num}/{n_teams} ===")
        print(
            f"  ESPN Active FPTS: {rec['hero_af']:,} vs Field Avg {rec['field_af']:,.1f} ({rec['diff']:+,.1f}) | "
            f"Finish: #{rec['rank']:.1f}/{n_teams} | Rookies: {rec['hero_rk']} | Primary C: {rec['hero_c']}/{max_centers} | Benched: {rec['hero_bg']}g"
        )
        print(f"  {'Rd':>2s} {'Pk':>3s} {'Slot':>4s} | {'Player':23s} {'Type':>4s} {'Pos':>8s} | {'ADP':>5s} {'HeroRk':>6s} | {'Gm':>3s} {'FPPG':>5s} {'Total':>7s}")
        print("  " + "-" * 86)
        tag_map = {"returning_vet": "VET", "rookie": "ROOK", "returning_absence": "RET"}
        for rnd, pick_num, idx in picks:
            p = players[idx]
            print(
                f"  {rnd:2d} {pick_num:3d} {slot_of[idx]:>4s} | {p['player'][:23]:23s} {tag_map[p['player_type']]:>4s} {p['pos_str'][:8]:>8s} | "
                f"{p['adp']:5.1f} {rank_of[idx]:6d} | {p['next_games']:3d} {p['next_raw_fppg']:5.1f} {p['total_fpts']:7,d}"
            )


def generate_live_full_board(data, target_season, strategy, board_lambda, max_train_adp, max_pool_adp, cv_folds, exclude_players):
    """Train strictly on completed drafts < target_season and export a full-pool Top-150 CSV (vets + rookies)."""
    if not re.fullmatch(r"\d{4}-\d{2}", target_season):
        raise ValueError(f"--generate-board expects YYYY-YY format (got {target_season!r})")
    target_year = int(target_season[:4])
    prior_year = target_year - 1
    adp_path = data / target_season / "adp.csv"
    if not adp_path.is_file():
        raise ValueError(f"missing {adp_path}")

    hist_rows = build_completed_dataset(data, max_pool_adp=max_pool_adp)
    prior_seasons = sorted({r["draft_season"] for r in hist_rows if r["draft_season"] < target_season})
    if not prior_seasons:
        raise ValueError(f"no completed draft seasons before {target_season}")
    train_all = [r for r in hist_rows if r["draft_season"] in prior_seasons]

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
        lam = tune_global_lambdas(cv_scored).get(strategy, 0.65) if cv_scored else 0.65
    else:
        lam = board_lambda

    seasons, get_season, get_summaries, get_leaderboard = load_season_cache(data)
    prior_res = get_season(prior_year)
    if prior_res is None:
        raise ValueError(f"missing completed prior season {season_label(prior_year)}")
    next_res = get_season(target_year)

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
        next_res=next_res,
        next_sum=get_summaries(target_year) if next_res is not None else None,
        next_lb_tuple=get_leaderboard(target_year) if next_res is not None else None,
        played_before=played_before,
        history_available=history_available,
        max_pool_adp=max_pool_adp,
        live_mode=(next_res is None),
        excluded_keys=excluded_keys,
    )
    scores, meta = fit_and_score_full_pool(train_all, cands, max_train_adp=max_train_adp)
    for i, row in enumerate(cands):
        for st, vals in scores.items():
            row[f"{st}_score"] = float(vals[i])

    ordered = build_global_board(cands, strategy, lam)
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
            "is_redshirt": rec["is_redshirt"],
            "rookie_age": rec["rookie_age"],
            "team_top8_avg_age": rec["team_top8_avg_age"],
            "team_prior_win_pct": rec["team_prior_win_pct"],
        }
        if "next_adj_fppg" in rec:
            out_r.update({
                "actual_games": rec["next_games"],
                "actual_rank": rec["next_rank"],
                "actual_adj_fppg": rec["next_adj_fppg"],
                "fppg_above_adp": rec["fppg_above_adp"],
            })
        board_rows.append(out_r)

    out_path = data / f"draft_board_full_{target_season}.csv"
    write_csv(out_path, board_rows)

    wh, wd = meta["vet_weights"]
    print(f"\n=== Generated {target_season} FULL-POOL Draft Board ({strategy}, lambda={lam:.2f}) ===")
    print(f"Trained on {len(prior_seasons)} prior drafts: {', '.join(prior_seasons)}")
    print(f"Pool: {len(cands)} players ({counts['returning_vet']} vets, {counts['rookie']} rookies, {counts['returning_absence']} returning/unmatched)")
    print(f"Vet 2-model weights (huber / decomposed): {wh:.2f} / {wd:.2f} | Rookie flat bias: {meta['rookie_flat_bias']:+.2f} AdjFPPG")
    if removed:
        print(f"Manually excluded: {', '.join(removed)}")

    flagged = [r for r in board_rows if r["market_collapse_flag"] and r["market_rank"] <= 175]
    if flagged:
        print("\nAuto-Detected Market-Collapse / Injured Stars (raw prior FPPG > mkt + 8 & >=20 slot drop):")
        for r in flagged[:10]:
            print(f"  #{r['board_rank']:3d} (ADP {r['adp']:5.1f}, {r['slot_move']:+3d}) | {r['player']:24s} | Prior {r['prior_fppg']:5.1f} vs Mkt {r['mkt_expected_adj_fppg']:5.1f} | {r['edge_vs_peers']:+.2f} vs peers")

    rk_board = [r for r in board_rows if r["player_type"] == "rookie"]
    if rk_board:
        print(f"\nIncoming Rookies Ranked on {target_season} Board ({len(rk_board)} total, {sum(r['in_top_150'] for r in rk_board)} in Top 150):")
        for r in rk_board[:12]:
            pk = f"Pk#{r['draft_pick']}" if r["draft_pick"] is not None else "UDFA"
            tag = "" if r["in_top_150"] else " [CUT]"
            print(f"  #{r['board_rank']:3d} (ADP {r['adp']:5.1f}, {r['slot_move']:+3d}) | {r['player']:23s} ({pk:5s}) | Proj {r['proj_adj_fppg']:5.2f} | {r['edge_vs_peers']:+.2f} vs peers{tag}")

    print(f"\nSaved full-pool ranked draft board to {out_path}")


def run_backtest(data, holdout_seasons=2, cv_folds=6, max_train_adp=175.0, max_pool_adp=225.0, skip_sim=False, trials_per_slot=12, adp_var=2.0, seed=None):
    all_rows = build_completed_dataset(data, max_pool_adp=max_pool_adp)
    seasons = sorted({r["draft_season"] for r in all_rows})
    if len(seasons) <= holdout_seasons:
        raise ValueError(f"need more completed draft seasons than --holdout-seasons ({len(seasons)} found)")

    development, holdout = seasons[:-holdout_seasons], seasons[-holdout_seasons:]
    dev_rows = [r for r in all_rows if r["draft_season"] in development]
    ho_rows = [r for r in all_rows if r["draft_season"] in holdout]

    # 1. Expanding-window chronological CV across development seasons
    cv_seasons = development[2:][-cv_folds:] if len(development) >= 3 else development[1:]
    cv_scored = []
    for s in cv_seasons:
        earlier = [r for r in dev_rows if r["draft_season"] < s]
        cands = [r.copy() for r in dev_rows if r["draft_season"] == s]
        if len(earlier) < 60 or not cands:
            continue
        scores, _ = fit_and_score_full_pool(earlier, cands, max_train_adp=max_train_adp)
        for i, row in enumerate(cands):
            row["split"] = "cv"
            row["training_through"] = max(r["draft_season"] for r in earlier)
            for strat, vals in scores.items():
                row[f"{strat}_score"] = round(float(vals[i]), 6)
        cv_scored.extend(cands)

    best_lambdas = tune_global_lambdas(cv_scored)

    # 2. Final holdout trained on all development seasons
    ho_scored = [r.copy() for r in ho_rows]
    ho_scores, ho_meta = fit_and_score_full_pool(dev_rows, ho_scored, max_train_adp=max_train_adp)
    for i, row in enumerate(ho_scored):
        row["split"] = "holdout"
        row["training_through"] = development[-1]
        for strat, vals in ho_scores.items():
            row[f"{strat}_score"] = round(float(vals[i]), 6)

    board_summary = []
    for split_label, split_rows in (("cv", cv_scored), ("holdout", ho_scored)):
        if not split_rows:
            continue
        for strat in STRATEGIES:
            board_summary.append(evaluate_global_board(split_rows, split_label, strat, best_lambdas[strat]))

    # Strip tuple column before writing CSVs
    def csv_clean(rows_in):
        out = []
        for r in rows_in:
            d = dict(r)
            if isinstance(d.get("adp_positions"), tuple):
                d["adp_positions"] = ",".join(d["adp_positions"])
            out.append(d)
        return out

    write_csv(data / "backtest_full_pool_board_summary.csv", board_summary)
    if cv_scored:
        write_csv(data / "backtest_full_pool_cv_results.csv", csv_clean(cv_scored))
    write_csv(data / "backtest_full_pool_results.csv", csv_clean(ho_scored))

    wh, wd = ho_meta["vet_weights"]
    print(f"\nDevelopment drafts: {', '.join(development)}")
    print(f"Holdout drafts:     {', '.join(holdout)}")
    print(f"Vet 2-model weights (huber / decomposed): {wh:.2f} / {wd:.2f} | Rookie median bias: {ho_meta['rookie_flat_bias']:+.2f}")
    cap_w = ho_meta["vet_coefs"].get("prior_minus_mkt_capped")
    col_w = ho_meta["vet_coefs"].get("market_collapse_excess")
    if cap_w is not None and col_w is not None:
        print(f"Injury-collapse hinge (raw prior FPPG, >=20 slot drop) -> capped: {cap_w:+.2f} | collapse_excess: {col_w:+.2f}")

    for split_label in ("cv", "holdout"):
        b_rows = [r for r in board_summary if r["split"] == split_label]
        if not b_rows:
            continue
        print(f"\n=== FULL-POOL GLOBAL Top-150 Board ({split_label.upper()}, {b_rows[0]['seasons']} seasons, Vets + Rookies) ===")
        for r in b_rows:
            swap_str = (
                f"Swaps({r['swaps_count']:2d}): +{r['added_top150_pct']:4.1f}% vs -{r['dropped_top150_pct']:4.1f}% Top150 ({r['added_minus_dropped_fppg']:+.2f})"
                if r["swaps_count"] and r["added_minus_dropped_fppg"] is not None
                else "Swaps( 0): market baseline"
            )
            pval = f" p={r['vs_market_p_value']:.3f}" if r["vs_market_p_value"] is not None else ""
            print(
                f"  {r['strategy']:20s} (lam={r['lambda']:.2f}) | Top150: {r['actual_top_150_captured']:3d}/{r['board_slots']} "
                f"({r['top_150_capture_pct']:5.2f}%) | AvgAdj: {r['board_mean_adj_fppg']:5.2f}{pval} | "
                f"MAE: {r['mean_abs_rank_error']:5.2f} | Rk:{r['rookies_on_board']:2d} 0g:{r['zero_game_busts']:2d} | {swap_str}"
            )

    if not skip_sim:
        run_espn_simulation(
            data, cv_scored + ho_scored, best_lambdas,
            trials_per_slot=trials_per_slot, adp_var=adp_var, seed=seed,
        )

    print(f"\nSaved full-pool outputs under {data}/backtest_full_pool_*.csv and {data}/sim_full_pool_*.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"), help="download folder (default: data)")
    parser.add_argument("--holdout-seasons", type=int, default=2, help="latest draft seasons held out (default: 2)")
    parser.add_argument("--cv-folds", type=int, default=6, help="max chronological CV folds (default: 6)")
    parser.add_argument("--max-train-adp", type=float, default=175.0, help="max vet ADP used for fitting (default: 175)")
    parser.add_argument("--max-pool-adp", type=float, default=225.0, help="max ADP allowed into full pool (default: 225)")
    parser.add_argument("--skip-sim", action="store_true", help="skip the 12-team ESPN draft simulation")
    parser.add_argument("--trials-per-slot", type=int, default=12, help="Monte Carlo drafts per Hero slot (default: 12)")
    parser.add_argument("--adp-var", type=float, default=2.0, help="villain R2-13 Uniform ADP noise bound (default: 2.0)")
    parser.add_argument("--seed", type=int, default=None, help="simulation random seed (default: random)")
    parser.add_argument("--generate-board", metavar="YYYY-YY", help="export full-pool draft board CSV for a season")
    parser.add_argument("--board-strategy", choices=STRATEGIES[1:], default="unified_stacked", help="strategy for --generate-board")
    parser.add_argument("--board-lambda", type=float, default=None, help="override global lambda for --generate-board")
    parser.add_argument("--exclude-players", default="", help="comma-separated injured/out players to exclude in --generate-board")
    args = parser.parse_args()
    if args.holdout_seasons < 1 or args.cv_folds < 1 or args.trials_per_slot < 1 or args.adp_var < 0:
        parser.error("use valid positive parameters")
    try:
        if args.generate_board:
            generate_live_full_board(
                args.data, args.generate_board, args.board_strategy, args.board_lambda,
                args.max_train_adp, args.max_pool_adp, args.cv_folds, args.exclude_players,
            )
        else:
            run_backtest(
                args.data, args.holdout_seasons, args.cv_folds, args.max_train_adp,
                args.max_pool_adp, args.skip_sim, args.trials_per_slot, args.adp_var, args.seed,
            )
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(1, f"Full-pool pipeline failed: {error}\n")


if __name__ == "__main__":
    main()
