"""Availability-adjusted, returning-player-only 2-model ensemble backtest & live Top-150 board generator."""

import argparse
import csv
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LogisticRegression, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

from adp_names import clean_adp_player_name
from backtest_common import (
    adp_slot_points,
    load_birth_dates,
    name_key,
    read_adp,
    read_season,
    write_csv,
)


BOARD_SIZE = 150
BANDS = ((1, 25), (26, 50), (51, 75), (76, 100), (101, 125), (126, 175))
TARGETS_PER_BAND = 5
REF_GAMES = 65.0
WAIVER_FACTOR = 0.85
COLLAPSE_THRESHOLD = 8.0
COLLAPSE_MIN_SLOT_DROP = 20.0
RIDGE_ALPHAS = np.logspace(-1, 3.5, 25)
LAMBDA_GRID = (0.35, 0.50, 0.65, 0.80)
RESIDUAL_CLIP = 8.0

# Standalone benchmark features (not included in the final 2-model ensemble)
BASE_FEATURES = (
    "adp",
    "age_at_prior_break",
    "prior_games",
    "prior_fppg",
    "prior_post_break_fppg_change",
    "prior_start_pct",
)

# Clean feature set with same-scale AdjFPPG gap and slot-gated market-collapse hinge
CLEAN_FEATURES = (
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

MPG_FEATURES = (
    "log_adp",
    "mkt_expected_adj_fppg",
    "prior_minus_mkt_capped",
    "market_collapse_excess",
    "age_at_prior_break",
    "age_under_24",
    "age_over_30",
    "prior_games",
    "prior_mpg",
    "prior_start_pct",
    "yoy_mpg_delta",
    "two_yr_games_avg",
    "no_prev2_season",
    "adp_platform_std",
    "espn_vs_avg",
)

FPPM_FEATURES = (
    "log_adp",
    "mkt_expected_adj_fppg",
    "prior_minus_mkt_capped",
    "market_collapse_excess",
    "age_at_prior_break",
    "age_under_24",
    "age_over_30",
    "prior_fppm",
    "prior_mpg",
    "yoy_fppg_delta",
    "two_yr_weighted_fppg",
    "no_prev2_season",
    "adp_platform_std",
    "espn_vs_avg",
)

AVAIL_FEATURES = (
    "log_adp",
    "mkt_expected_adj_fppg",
    "prior_minus_mkt_capped",
    "market_collapse_excess",
    "age_at_prior_break",
    "age_under_24",
    "age_over_30",
    "prior_games",
    "two_yr_games_avg",
    "prior_mpg",
    "prior_start_pct",
    "no_prev2_season",
    "adp_platform_std",
    "espn_vs_avg",
)

MODEL_NAMES = (
    "logistic_base",
    "huber_clean",
    "decomposed_clean",
    "ensemble_equal",
    "ensemble_stacked",
)


def _safe_float(raw):
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def season_label(start_year):
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def compute_season_adj_leaderboard(players):
    """Compute waiver replacement FPPG and availability-adjusted FPPG leaderboard for a season.

    Each player's AdjFPPG blends their actual FPPG over min(games, 65) games with
    waiver-wire replacement value (0.85 * 150th-ranked player's FPPG) for missed games.
    A 0-game injured player receives waiver value instead of being dropped as unranked.
    """
    qual = sorted(
        (p.total_points / p.total_games for p in players.values() if p.total_games >= 20),
        reverse=True,
    )
    if not qual:
        qual = sorted((p.total_points / p.total_games for p in players.values() if p.total_games > 0), reverse=True)
    repl_fppg = qual[min(149, len(qual) - 1)] if qual else 20.0
    waiver_fppg = WAIVER_FACTOR * repl_fppg

    adj_by_id = {}
    for aid, p in players.items():
        g = p.total_games
        if g <= 0:
            adj_by_id[aid] = waiver_fppg
        else:
            raw = p.total_points / g
            eff_g = min(float(g), REF_GAMES)
            adj_by_id[aid] = (eff_g * raw + (REF_GAMES - eff_g) * waiver_fppg) / REF_GAMES

    leaderboard = sorted(adj_by_id.values(), reverse=True)
    rank_by_score = {}
    for place, score in enumerate(leaderboard, start=1):
        rank_by_score.setdefault(score, place)
    return waiver_fppg, adj_by_id, leaderboard, rank_by_score


def read_adp_platform_stats(path):
    """Extract cross-platform ADP disagreement (Yahoo, ESPN, CBS vs AVG) when available."""
    stats = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = csv.DictReader(source)
        cols = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in (rows.fieldnames or [])}
        name_col = cols.get("playername")
        avg_col = next((cols[k] for k in ("avg", "average", "adp", "averagedraftposition") if k in cols), None)
        if not name_col or not avg_col:
            return stats
        for row in rows:
            cleaned = clean_adp_player_name(row.get(name_col) or "")
            avg = _safe_float(row.get(avg_col))
            if not cleaned or avg is None or avg <= 0:
                continue
            key = name_key(cleaned)
            if key in stats:
                continue
            platforms = [_safe_float(row.get(cols[p])) for p in ("yahoo", "espn", "cbs") if p in cols]
            valid = [v for v in platforms if v is not None and v > 0]
            espn = _safe_float(row.get(cols["espn"])) if "espn" in cols else None
            std = float(np.std(valid, ddof=1)) if len(valid) >= 2 else None
            espn_vs_avg = (avg - espn) if (espn is not None and espn > 0) else None
            stats[key] = {"adp_platform_std": std, "espn_vs_avg": espn_vs_avg}
    return stats


def player_season_summary(player):
    games = player.total_games
    if not games:
        return None
    fppg = player.total_points / games
    tot_min = player.box_score_totals.get("minutes")
    if tot_min is not None and tot_min > 0:
        mpg = tot_min / games
        fppm = player.total_points / tot_min
    else:
        mpg, fppm = None, None
    start_pct = (100.0 * player.starts / games) if not player.games_without_start_status else None
    post_chg = (
        player.post_points / player.post_games - player.pre_points / player.pre_games
        if player.pre_games and player.post_games
        else None
    )
    return {
        "games": games,
        "fppg": fppg,
        "mpg": mpg,
        "fppm": fppm,
        "start_pct": start_pct,
        "post_fppg_chg": post_chg,
    }


def load_season_cache(data):
    seasons = {int(p.name[:4]): p for p in data.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}", p.name)}
    raw_cache, summary_cache, lb_cache = {}, {}, {}

    def get_season(year):
        if year not in raw_cache:
            folder = seasons.get(year)
            if folder is None or not (folder / "games.csv").is_file() or not (folder / "player_box_scores.csv").is_file():
                raw_cache[year] = None
            else:
                raw_cache[year] = read_season(folder, split_at_all_star=True, include_box_scores=True)
        return raw_cache[year]

    def get_summaries(year):
        if year not in summary_cache:
            res = get_season(year)
            summary_cache[year] = (
                {aid: s for aid, p in res[1].items() if (s := player_season_summary(p)) is not None}
                if res is not None else None
            )
        return summary_cache[year]

    def get_leaderboard(year):
        if year not in lb_cache:
            res = get_season(year)
            lb_cache[year] = compute_season_adj_leaderboard(res[1]) if res is not None else None
        return lb_cache[year]

    return seasons, get_season, get_summaries, get_leaderboard


def match_returning_adp(adp_path, prior_players, max_pool_adp=225.0):
    """Return ordered list of returning players (present in prior_players) from adp.csv.

    Rookies and players who did not appear in the prior regular season are omitted completely.
    """
    ids_by_name = defaultdict(set)
    for aid, p in prior_players.items():
        for n in p.names:
            ids_by_name[name_key(n)].add(aid)

    market = {}
    for name, adp in sorted(read_adp(adp_path), key=lambda item: (item[1], item[0].casefold())):
        market.setdefault(name_key(name), (name, adp))

    matched, seen_ids = [], set()
    skipped_rookies = 0
    for key, (name, adp) in market.items():
        ids = ids_by_name[key]
        if len(ids) != 1:
            skipped_rookies += 1
            continue
        (aid,) = ids
        if aid in seen_ids:
            continue
        seen_ids.add(aid)
        if len(matched) >= int(max_pool_adp) and adp > max_pool_adp:
            continue
        matched.append((len(matched) + 1, name, aid, adp))
    return matched, len(market), skipped_rookies


def build_candidate_row(
    *,
    prior_name,
    draft_name,
    player_name,
    athlete_id,
    adp,
    market_rank,
    prior_summary,
    prev2_summary,
    has_prev2_season,
    prior_lb_tuple,
    all_star_date,
    birth_date,
    platform_stats,
    next_player,
    next_summary,
    next_lb_tuple,
):
    """Build a returning player's feature + availability-adjusted outcome row."""
    age = None
    if birth_date is not None and all_star_date is not None:
        age = all_star_date.year - birth_date.year - (
            (all_star_date.month, all_star_date.day) < (birth_date.month, birth_date.day)
        )

    prior_waiver_fppg, prior_adj_by_id, prior_adj_leaderboard, prior_rank_by_score = prior_lb_tuple
    prior_adj_fppg = prior_adj_by_id.get(athlete_id, prior_waiver_fppg)
    prior_adj_rank = prior_rank_by_score.get(prior_adj_fppg, len(prior_adj_leaderboard))

    # Same-scale gap (prior AdjFPPG vs ADP-implied AdjFPPG) + slot-drop gate to prevent top-5 false alarms
    mkt_exp_adj = adp_slot_points(prior_adj_leaderboard, adp)
    adj_gap = prior_adj_fppg - mkt_exp_adj
    slot_drop = adp - prior_adj_rank
    is_market_collapse = slot_drop >= COLLAPSE_MIN_SLOT_DROP and adj_gap > COLLAPSE_THRESHOLD
    capped_gap = max(-10.0, min(COLLAPSE_THRESHOLD, adj_gap))
    collapse_excess = (adj_gap - COLLAPSE_THRESHOLD) if is_market_collapse else 0.0
    pstats = platform_stats.get(name_key(player_name), {})

    if prev2_summary is not None:
        yoy_fppg = prior_summary["fppg"] - prev2_summary["fppg"]
        yoy_mpg = (
            prior_summary["mpg"] - prev2_summary["mpg"]
            if prior_summary["mpg"] is not None and prev2_summary["mpg"] is not None
            else None
        )
        two_yr_fppg = 0.65 * prior_summary["fppg"] + 0.35 * prev2_summary["fppg"]
        two_yr_games = 0.5 * (prior_summary["games"] + prev2_summary["games"])
        no_prev2 = 0.0
    else:
        yoy_fppg, yoy_mpg = None, None
        two_yr_fppg, two_yr_games = prior_summary["fppg"], float(prior_summary["games"])
        no_prev2 = 1.0 if has_prev2_season else None

    outcome = {}
    if next_lb_tuple is not None:
        next_waiver, next_adj_by_id, next_adj_lb, next_rank_by_score = next_lb_tuple
        next_games = next_player.total_games if next_player else 0
        raw_next_fppg = (next_player.total_points / next_games) if next_games > 0 else 0.0
        next_adj_fppg = next_adj_by_id.get(athlete_id, next_waiver)
        if next_games > 0 and next_adj_fppg in next_rank_by_score:
            next_adj_rank = next_rank_by_score[next_adj_fppg]
        else:
            next_adj_rank = sum(1 for v in next_adj_lb if v > next_waiver) + 1
        slot_adj_fppg = adp_slot_points(next_adj_lb, adp)
        adj_above_adp = next_adj_fppg - slot_adj_fppg
        outcome = {
            "next_games": next_games,
            "next_avail_frac": min(float(next_games), REF_GAMES) / REF_GAMES,
            "next_raw_fppg": round(raw_next_fppg, 3),
            "next_adj_fppg": round(next_adj_fppg, 3),
            "adp_slot_adj_fppg": round(slot_adj_fppg, 3),
            "fppg_above_adp": round(adj_above_adp, 3),
            "next_rank": next_adj_rank,
            "adp_slots_gained": round(adp - next_adj_rank, 3),
            "beat_adp": next_adj_rank < adp,
            "next_fppg": round(next_adj_fppg, 3),
            "next_mpg": next_summary["mpg"] if next_summary else None,
            "next_fppm": next_summary["fppm"] if next_summary else None,
        }

    return {
        "prior_season": prior_name,
        "draft_season": draft_name,
        "player": player_name,
        "athlete_id": athlete_id,
        "adp": adp,
        "market_rank": market_rank,
        "prior_waiver_fppg": prior_waiver_fppg,
        "prior_adj_fppg": prior_adj_fppg,
        "prior_adj_rank": prior_adj_rank,
        # Base logistic benchmark features
        "age_at_prior_break": age,
        "prior_games": prior_summary["games"],
        "prior_fppg": prior_summary["fppg"],
        "prior_post_break_fppg_change": prior_summary["post_fppg_chg"],
        "prior_start_pct": prior_summary["start_pct"],
        # Clean availability-aware features
        "log_adp": math.log(max(adp, 1.0)),
        "mkt_expected_adj_fppg": mkt_exp_adj,
        "prior_adj_minus_mkt": adj_gap,
        "prior_minus_mkt_capped": capped_gap,
        "market_collapse_excess": collapse_excess,
        "age_under_24": max(0.0, 24.0 - age) if age is not None else None,
        "age_over_30": max(0.0, age - 30.0) if age is not None else None,
        "prior_mpg": prior_summary["mpg"],
        "prior_fppm": prior_summary["fppm"],
        "yoy_fppg_delta": yoy_fppg,
        "yoy_mpg_delta": yoy_mpg,
        "two_yr_weighted_fppg": two_yr_fppg,
        "two_yr_games_avg": two_yr_games,
        "no_prev2_season": no_prev2,
        "adp_platform_std": pstats.get("adp_platform_std"),
        "espn_vs_avg": pstats.get("espn_vs_avg"),
        **outcome,
    }


def build_completed_dataset(data, max_pool_adp=225.0):
    """Build returning-player-only dataset across all completed season pairs (0-game busts included)."""
    seasons, get_season, get_summaries, get_leaderboard = load_season_cache(data)
    rows, pools_by_season = [], {}

    for year in sorted(seasons):
        following = seasons.get(year + 1)
        if following is None or not (following / "adp.csv").is_file():
            continue
        prior_res, next_res = get_season(year), get_season(year + 1)
        if prior_res is None or next_res is None:
            continue

        all_star_date, prior_players = prior_res
        _, next_players = next_res
        prior_sum = get_summaries(year)
        next_sum = get_summaries(year + 1)
        prev2_sum = get_summaries(year - 1) if (year - 1) in seasons else None
        prior_lb_tuple = get_leaderboard(year)
        next_lb_tuple = get_leaderboard(year + 1)
        platform_stats = read_adp_platform_stats(following / "adp.csv")

        try:
            births = load_birth_dates(data, year + 1)
        except Exception as err:
            print(f"{seasons[year].name}: birth dates unavailable ({err})")
            births = {}

        matched, total_adp, skipped = match_returning_adp(following / "adp.csv", prior_players, max_pool_adp=max_pool_adp)
        print(
            f"{seasons[year].name} -> {following.name}: {len(matched)}/{total_adp} returning ADP players "
            f"({skipped} rookies/unmatched omitted)"
        )

        season_rows = []
        for market_rank, name, aid, adp in matched:
            pr = prior_sum.get(aid)
            if pr is None:
                continue
            row = build_candidate_row(
                prior_name=seasons[year].name,
                draft_name=following.name,
                player_name=name,
                athlete_id=aid,
                adp=adp,
                market_rank=market_rank,
                prior_summary=pr,
                prev2_summary=prev2_sum.get(aid) if prev2_sum is not None else None,
                has_prev2_season=prev2_sum is not None,
                prior_lb_tuple=prior_lb_tuple,
                all_star_date=all_star_date,
                birth_date=births.get(aid),
                platform_stats=platform_stats,
                next_player=next_players.get(aid),
                next_summary=next_sum.get(aid),
                next_lb_tuple=next_lb_tuple,
            )
            rows.append(row)
            season_rows.append(row)
        if len(season_rows) >= BOARD_SIZE:
            pools_by_season[following.name] = season_rows

    return rows, pools_by_season


def build_live_candidates(data, target_season, max_pool_adp=225.0, excluded_keys=()):
    """Build returning-player-only candidates for a target draft season (works pre-season with only adp.csv)."""
    if not re.fullmatch(r"\d{4}-\d{2}", target_season):
        raise ValueError(f"--generate-board expects YYYY-YY format (got {target_season!r})")
    target_year = int(target_season[:4])
    prior_year = target_year - 1
    target_folder = data / target_season
    adp_path = target_folder / "adp.csv"
    if not adp_path.is_file():
        raise ValueError(f"missing {adp_path} for --generate-board {target_season}")

    seasons, get_season, get_summaries, get_leaderboard = load_season_cache(data)
    prior_res = get_season(prior_year)
    if prior_res is None:
        raise ValueError(f"missing completed prior season {season_label(prior_year)}")

    all_star_date, prior_players = prior_res
    prior_sum = get_summaries(prior_year)
    prev2_sum = get_summaries(prior_year - 1) if (prior_year - 1) in seasons else None
    prior_lb_tuple = get_leaderboard(prior_year)

    next_res = get_season(target_year)
    next_players = next_res[1] if next_res is not None else {}
    next_sum = get_summaries(target_year) if next_res is not None else {}
    next_lb_tuple = get_leaderboard(target_year) if next_res is not None else None

    platform_stats = read_adp_platform_stats(adp_path)
    try:
        births = load_birth_dates(data, prior_year + 1)
    except Exception as err:
        print(f"{season_label(prior_year)}: birth dates unavailable ({err})")
        births = {}

    matched, total_adp, skipped = match_returning_adp(adp_path, prior_players, max_pool_adp=max_pool_adp)
    excluded_set = set(excluded_keys)
    candidates, removed = [], []
    rank = 0
    for _, name, aid, adp in matched:
        if name_key(name) in excluded_set:
            removed.append(name)
            continue
        pr = prior_sum.get(aid)
        if pr is None:
            continue
        rank += 1
        candidates.append(build_candidate_row(
            prior_name=season_label(prior_year),
            draft_name=target_season,
            player_name=name,
            athlete_id=aid,
            adp=adp,
            market_rank=rank,
            prior_summary=pr,
            prev2_summary=prev2_sum.get(aid) if prev2_sum is not None else None,
            has_prev2_season=prev2_sum is not None,
            prior_lb_tuple=prior_lb_tuple,
            all_star_date=all_star_date,
            birth_date=births.get(aid),
            platform_stats=platform_stats,
            next_player=next_players.get(aid) if next_lb_tuple is not None else None,
            next_summary=next_sum.get(aid) if next_lb_tuple is not None else None,
            next_lb_tuple=next_lb_tuple,
        ))
    if len(candidates) < BOARD_SIZE:
        raise ValueError(f"{target_season}: only {len(candidates)} returning players found, need at least {BOARD_SIZE}")
    return candidates, total_adp, skipped, removed


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


def _fit_predict_decomposed(training, evaluated):
    """3-part structural decomposition: E[AdjFPPG] = E[avail] * (E[MPG] * E[FPPM]) + (1 - E[avail]) * waiver."""
    rate_rows = [r for r in training if r.get("next_mpg") is not None and r.get("next_fppm") is not None]
    mpg_feats = active_features(rate_rows, MPG_FEATURES)
    fppm_feats = active_features(rate_rows, FPPM_FEATURES)
    avail_feats = active_features(training, AVAIL_FEATURES)

    X_mpg_tr = to_matrix(rate_rows, mpg_feats)
    y_mpg_tr = np.array([float(r["next_mpg"]) for r in rate_rows], dtype=float)
    X_fppm_tr = to_matrix(rate_rows, fppm_feats)
    y_fppm_tr = np.array([float(r["next_fppm"]) for r in rate_rows], dtype=float)
    X_av_tr = to_matrix(training, avail_feats)
    y_av_tr = np.array([float(r["next_avail_frac"]) for r in training], dtype=float)

    groups_all = np.array([r["draft_season"] for r in training])
    unique_seasons = sorted(set(groups_all))

    oof_mpg = np.zeros(len(training), dtype=float)
    oof_fppm = np.zeros(len(training), dtype=float)
    oof_avail = np.zeros(len(training), dtype=float)

    if len(unique_seasons) >= 2:
        gkf = GroupKFold(n_splits=min(len(unique_seasons), 5))
        for tr_idx, va_idx in gkf.split(training, y_av_tr, groups_all):
            tr_all = [training[i] for i in tr_idx]
            va_all = [training[i] for i in va_idx]
            tr_rate = [r for r in tr_all if r.get("next_mpg") is not None and r.get("next_fppm") is not None]
            m_mod = _make_spline_ridge(mpg_feats).fit(to_matrix(tr_rate, mpg_feats), [float(r["next_mpg"]) for r in tr_rate])
            f_mod = _make_spline_ridge(fppm_feats).fit(to_matrix(tr_rate, fppm_feats), [float(r["next_fppm"]) for r in tr_rate])
            a_mod = _make_spline_ridge(avail_feats).fit(to_matrix(tr_all, avail_feats), [float(r["next_avail_frac"]) for r in tr_all])
            oof_mpg[va_idx] = np.clip(m_mod.predict(to_matrix(va_all, mpg_feats)), 5.0, 42.0)
            oof_fppm[va_idx] = np.clip(f_mod.predict(to_matrix(va_all, fppm_feats)), 0.3, 2.5)
            oof_avail[va_idx] = np.clip(a_mod.predict(to_matrix(va_all, avail_feats)), 0.0, 1.0)
    else:
        m_mod = _make_spline_ridge(mpg_feats).fit(X_mpg_tr, y_mpg_tr)
        f_mod = _make_spline_ridge(fppm_feats).fit(X_fppm_tr, y_fppm_tr)
        a_mod = _make_spline_ridge(avail_feats).fit(X_av_tr, y_av_tr)
        oof_mpg = np.clip(m_mod.predict(to_matrix(training, mpg_feats)), 5.0, 42.0)
        oof_fppm = np.clip(f_mod.predict(to_matrix(training, fppm_feats)), 0.3, 2.5)
        oof_avail = np.clip(a_mod.predict(X_av_tr), 0.0, 1.0)

    tr_waiver = np.array([float(r["prior_waiver_fppg"]) for r in training])
    tr_struct_adj = oof_avail * (oof_mpg * oof_fppm) + (1.0 - oof_avail) * tr_waiver

    mpg_full = _make_spline_ridge(mpg_feats).fit(X_mpg_tr, y_mpg_tr)
    fppm_full = _make_spline_ridge(fppm_feats).fit(X_fppm_tr, y_fppm_tr)
    avail_full = _make_spline_ridge(avail_feats).fit(X_av_tr, y_av_tr)

    ev_mpg = np.clip(mpg_full.predict(to_matrix(evaluated, mpg_feats)), 5.0, 42.0)
    ev_fppm = np.clip(fppm_full.predict(to_matrix(evaluated, fppm_feats)), 0.3, 2.5)
    ev_avail = np.clip(avail_full.predict(to_matrix(evaluated, avail_feats)), 0.0, 1.0)
    ev_waiver = np.array([float(r["prior_waiver_fppg"]) for r in evaluated])
    ev_struct_adj = ev_avail * (ev_mpg * ev_fppm) + (1.0 - ev_avail) * ev_waiver

    tr_mkt = np.array([float(r["mkt_expected_adj_fppg"]) for r in training])
    ev_mkt = np.array([float(r["mkt_expected_adj_fppg"]) for r in evaluated])
    tr_log_adp = np.array([float(r["log_adp"]) for r in training])
    ev_log_adp = np.array([float(r["log_adp"]) for r in evaluated])
    tr_coll = np.array([float(r["market_collapse_excess"]) for r in training])
    ev_coll = np.array([float(r["market_collapse_excess"]) for r in evaluated])

    meta_tr = np.column_stack([
        tr_struct_adj - tr_mkt,
        (tr_struct_adj - tr_mkt) * tr_log_adp,
        tr_log_adp,
        tr_coll,
    ])
    meta_ev = np.column_stack([
        ev_struct_adj - ev_mkt,
        (ev_struct_adj - ev_mkt) * ev_log_adp,
        ev_log_adp,
        ev_coll,
    ])
    meta = make_pipeline(StandardScaler(), HuberRegressor(alpha=10.0, max_iter=1000))
    meta.fit(meta_tr, [float(r["fppg_above_adp"]) for r in training])
    return meta.predict(meta_ev)


def _fit_huber(training, evaluated, feats=CLEAN_FEATURES, alpha=25.0):
    use_feats = active_features(training, feats)
    pipe = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        HuberRegressor(alpha=alpha, epsilon=1.35, max_iter=1000),
    )
    pipe.fit(to_matrix(training, use_feats), [float(r["fppg_above_adp"]) for r in training])
    return pipe.predict(to_matrix(evaluated, use_feats)), dict(zip(use_feats, map(float, pipe.named_steps["huberregressor"].coef_)))


def _fit_logistic_logits(training, evaluated):
    base_feats = active_features(training, BASE_FEATURES)
    y_cls = np.array([1 if r["beat_adp"] else 0 for r in training], dtype=int)
    pipe = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=1000),
    )
    pipe.fit(to_matrix(training, base_feats), y_cls)
    probs = np.clip(pipe.predict_proba(to_matrix(evaluated, base_feats))[:, 1], 1e-4, 1.0 - 1e-4)
    return probs, np.log(probs / (1.0 - probs))


def _huber_loss_1d(err, delta=3.0):
    a = np.abs(err)
    return float(np.mean(np.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))))


def _fit_two_model_weights(oof_huber, oof_decomp, y_true, min_weight=0.15, shrink_l2=0.15):
    """Find convex weight w_huber in [min_weight, 1 - min_weight] minimizing Huber loss + L2 shrinkage to 0.5."""
    best_w, best_obj = 0.5, float("inf")
    for w in np.linspace(min_weight, 1.0 - min_weight, 29):
        pred = w * oof_huber + (1.0 - w) * oof_decomp
        obj = _huber_loss_1d(y_true - pred) + shrink_l2 * ((w - 0.5) ** 2)
        if obj < best_obj:
            best_obj, best_w = obj, float(w)
    return best_w, 1.0 - best_w


def fit_and_score_models(training, evaluated):
    """Fit huber_clean, decomposed_clean, their 2-model ensembles, and standalone logistic_base benchmark."""
    y_res = np.array([float(r["fppg_above_adp"]) for r in training], dtype=float)
    groups = np.array([r["draft_season"] for r in training])
    unique_seasons = sorted(set(groups))

    oof_logit = np.zeros(len(training), dtype=float)
    oof_huber = np.zeros(len(training), dtype=float)
    oof_decomp = np.zeros(len(training), dtype=float)

    if len(unique_seasons) >= 2:
        gkf = GroupKFold(n_splits=min(len(unique_seasons), 5))
        for tr_idx, va_idx in gkf.split(training, y_res, groups):
            tr_sub = [training[i] for i in tr_idx]
            va_sub = [training[i] for i in va_idx]
            if len({r["beat_adp"] for r in tr_sub}) >= 2:
                _, oof_logit[va_idx] = _fit_logistic_logits(tr_sub, va_sub)
            oof_huber[va_idx], _ = _fit_huber(tr_sub, va_sub, CLEAN_FEATURES)
            oof_decomp[va_idx] = _fit_predict_decomposed(tr_sub, va_sub)
    else:
        _, oof_logit = _fit_logistic_logits(training, training)
        oof_huber, _ = _fit_huber(training, training, CLEAN_FEATURES)
        oof_decomp = _fit_predict_decomposed(training, training)

    _, ev_logit = _fit_logistic_logits(training, evaluated)
    ev_huber, huber_coefs = _fit_huber(training, evaluated, CLEAN_FEATURES)
    ev_decomp = _fit_predict_decomposed(training, evaluated)

    # Calibrate standalone logistic_base logits onto FPPG scale strictly for apples-to-apples benchmarking
    cal = HuberRegressor(alpha=5.0, max_iter=500).fit(oof_logit.reshape(-1, 1), y_res)
    if cal.coef_[0] < 0.05:
        cal.coef_[0] = 0.05
    ev_log_fppg = cal.predict(ev_logit.reshape(-1, 1))

    # Clean 2-model ensembles (huber_clean + decomposed_clean ONLY; logistic_base excluded)
    ev_ens_equal = 0.5 * ev_huber + 0.5 * ev_decomp
    w_huber, w_decomp = _fit_two_model_weights(oof_huber, oof_decomp, y_res)
    ev_ens_stacked = w_huber * ev_huber + w_decomp * ev_decomp

    return {
        "logistic_base": ev_log_fppg,
        "huber_clean": ev_huber,
        "decomposed_clean": ev_decomp,
        "ensemble_equal": ev_ens_equal,
        "ensemble_stacked": ev_ens_stacked,
    }, {"stack_weights": (w_huber, w_decomp), "huber_coefs": huber_coefs}


def select_top_pct_indices(rows, score_col, top_pct):
    selected = set()
    for season in sorted({r["draft_season"] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r["draft_season"] == season]
        indices.sort(key=lambda i: (-rows[i][score_col], rows[i]["adp"], rows[i]["athlete_id"]))
        selected.update(indices[:math.ceil(len(indices) * top_pct / 100)])
    return selected


def attach_selection_flags(rows, top_pct):
    for model in MODEL_NAMES:
        selected = select_top_pct_indices(rows, f"{model}_score", top_pct)
        for i, row in enumerate(rows):
            row[f"{model}_selected"] = i in selected


def price_matched_lift(pool, picks):
    """Compare picks vs drawing the same season/25-slot ADP mix from pool (all players scored, 0-game included)."""
    by_band = defaultdict(list)
    for r in pool:
        by_band[(r["draft_season"], math.ceil(r["adp"] / 25))].append(r)
    counts = defaultdict(int)
    for r in picks:
        counts[(r["draft_season"], math.ceil(r["adp"] / 25))] += 1

    exp_hits = exp_margin = 0.0
    for key, k in counts.items():
        grp = by_band[key]
        frac = k / len(grp)
        exp_hits += frac * sum(1 for r in grp if r["beat_adp"])
        exp_margin += frac * sum(r["fppg_above_adp"] for r in grp)
    n = len(picks)
    if not n:
        return None, None
    hit_lift = 100.0 * (sum(1 for r in picks if r["beat_adp"]) - exp_hits) / n
    fppg_lift = mean(r["fppg_above_adp"] for r in picks) - exp_margin / n
    return round(fppg_lift, 2), round(hit_lift, 2)


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


def evaluate_band_picks(rows, split_label, top_pct):
    summary = []
    for lo, hi in ((0, 25), (25, 50), (50, 75), (75, 100), (100, 125), (125, 150)):
        label = f"{lo + 1}-{hi}"
        pool = [r for r in rows if lo < r["adp"] <= hi]
        if not pool:
            continue
        seasons = sorted({r["draft_season"] for r in pool})
        for model in ("market_pool", *MODEL_NAMES):
            picks = (
                pool if model == "market_pool"
                else [pool[i] for i in sorted(select_top_pct_indices(pool, f"{model}_score", top_pct))]
            )
            fppg_lift, hit_lift = price_matched_lift(pool, picks)
            season_lifts = []
            if model != "market_pool":
                for s in seasons:
                    sl, _ = price_matched_lift(
                        [r for r in pool if r["draft_season"] == s],
                        [r for r in picks if r["draft_season"] == s],
                    )
                    if sl is not None:
                        season_lifts.append(sl)
            _, _, _, p_val = paired_bootstrap_ci(season_lifts)
            summary.append({
                "split": split_label,
                "adp_band": label,
                "model": model,
                "seasons": len(seasons),
                "selected_players": len(picks),
                "zero_game_busts": sum(1 for r in picks if r["next_games"] == 0),
                "hit_rate_pct": round(100 * sum(1 for r in picks if r["beat_adp"]) / len(picks), 2) if picks else None,
                "avg_adj_fppg_above_adp": round(mean(r["fppg_above_adp"] for r in picks), 3) if picks else None,
                "price_matched_fppg_lift": fppg_lift,
                "price_matched_hit_rate_lift_pp": hit_lift,
                "season_lift_p_value": round(p_val, 4) if p_val is not None else None,
            })
    return summary


def evaluate_banded_boards(scored_rows, split_label):
    by_season = defaultdict(list)
    for r in scored_rows:
        by_season[r["draft_season"]].append(r)
    seasons = sorted(s for s, grp in by_season.items() if len(grp) >= BOARD_SIZE)
    summary = []

    for model in ("market", *MODEL_NAMES):
        placements, season_lifts = [], []
        for season in seasons:
            pool = sorted(by_season[season], key=lambda r: (r["market_rank"], r["adp"]))
            s_targets, m_targets = [], []
            for first, last in BANDS:
                band = pool[first - 1:last]
                m_targets.extend(band[:TARGETS_PER_BAND])
                ordered = band if model == "market" else sorted(band, key=lambda r: (-r[f"{model}_score"], r["market_rank"]))
                for rank, item in enumerate(ordered, start=first):
                    if rank <= BOARD_SIZE:
                        is_t = (rank - first) < TARGETS_PER_BAND
                        placements.append((rank, item, is_t))
                        if is_t:
                            s_targets.append(item)
            if s_targets and m_targets:
                season_lifts.append(mean(x["fppg_above_adp"] for x in s_targets) - mean(x["fppg_above_adp"] for x in m_targets))

        targets = [x for _, x, t in placements if t]
        _, _, _, p_val = paired_bootstrap_ci(season_lifts) if model != "market" else (None, None, None, None)
        summary.append({
            "split": split_label,
            "model": model,
            "seasons": len(seasons),
            "board_slots": len(placements),
            "actual_top_150_captured": sum(x["next_rank"] <= BOARD_SIZE for _, x, _ in placements),
            "top_150_capture_pct": round(100 * sum(x["next_rank"] <= BOARD_SIZE for _, x, _ in placements) / len(placements), 2) if placements else None,
            "mean_abs_rank_error": round(mean(abs(rank - x["next_rank"]) for rank, x, _ in placements), 2) if placements else None,
            "targets_zero_game_busts": sum(x["next_games"] == 0 for x in targets),
            "targets_hit_rate_pct": round(100 * sum(x["beat_adp"] for x in targets) / len(targets), 2) if targets else None,
            "targets_avg_adj_fppg_above_adp": round(mean(x["fppg_above_adp"] for x in targets), 3) if targets else None,
            "vs_market_p_value": round(p_val, 4) if p_val is not None else None,
        })
    return summary


def build_global_board(season_rows, model, lam):
    scored = []
    for item in season_rows:
        adj = 0.0
        if model != "market":
            adj = lam * max(-RESIDUAL_CLIP, min(RESIDUAL_CLIP, float(item[f"{model}_score"])))
        scored.append((item["mkt_expected_adj_fppg"] + adj, adj, item))
    scored.sort(key=lambda t: (-t[0], t[2]["market_rank"]))
    return scored


def tune_global_lambdas(cv_rows):
    by_season = defaultdict(list)
    for r in cv_rows:
        by_season[r["draft_season"]].append(r)
    seasons = sorted(s for s, grp in by_season.items() if len(grp) >= BOARD_SIZE)
    best_lambdas = {"market": 0.0}
    for model in MODEL_NAMES:
        best_lam, best_score = 0.50, (-1e9, -1e9)
        for lam in LAMBDA_GRID:
            fppgs, hits = [], 0
            for s in seasons:
                for _, _, item in build_global_board(by_season[s], model, lam)[:BOARD_SIZE]:
                    fppgs.append(item["next_adj_fppg"])
                    if item["next_rank"] <= BOARD_SIZE:
                        hits += 1
            cand = (mean(fppgs) if fppgs else -1e9, hits)
            if cand > best_score:
                best_score, best_lam = cand, lam
        best_lambdas[model] = best_lam
    return best_lambdas


def evaluate_global_board_single(scored_rows, split_label, model, lam):
    by_season = defaultdict(list)
    for r in scored_rows:
        by_season[r["draft_season"]].append(r)
    seasons = sorted(s for s, grp in by_season.items() if len(grp) >= BOARD_SIZE)
    market_fppg = {
        s: mean(x["next_adj_fppg"] for x in sorted(by_season[s], key=lambda r: r["market_rank"])[:BOARD_SIZE])
        for s in seasons
    }

    board_items, added, dropped, risers, fallers, season_lifts, rankings = [], [], [], [], [], [], []
    for s in seasons:
        ordered = build_global_board(by_season[s], model, lam)
        top150, outside = ordered[:BOARD_SIZE], ordered[BOARD_SIZE:]
        peer_baseline = median(adj for _, adj, item in ordered if item["market_rank"] <= BOARD_SIZE)
        season_lifts.append(mean(item["next_adj_fppg"] for _, _, item in top150) - market_fppg[s])

        for rank, (proj, adj, item) in enumerate(top150, start=1):
            board_items.append((rank, item))
            delta = item["market_rank"] - rank
            rel_edge = adj - peer_baseline
            if item["market_rank"] > BOARD_SIZE:
                added.append(item)
            if delta >= 10 and rel_edge > 0:
                risers.append(item)
            elif delta <= -10 and rel_edge < 0:
                fallers.append(item)
            rankings.append({
                "split": split_label, "draft_season": s, "model": model, "lambda": lam,
                "board_rank": rank, "market_rank": item["market_rank"], "slot_move": delta,
                "board_proj_adj_fppg": round(proj, 3), "edge_vs_adp": round(adj, 3),
                "edge_vs_peers": round(rel_edge, 3),
                "player": item["player"], "adp": item["adp"], "next_games": item["next_games"],
                "next_rank": item["next_rank"], "next_adj_fppg": item["next_adj_fppg"],
                "beat_adp": item["beat_adp"], "fppg_above_adp": item["fppg_above_adp"],
            })
        for rank, (_, adj, item) in enumerate(outside, start=BOARD_SIZE + 1):
            if item["market_rank"] <= BOARD_SIZE:
                dropped.append(item)
                if (item["market_rank"] - rank) <= -10 and (adj - peer_baseline) < 0:
                    fallers.append(item)

    _, _, _, p_val = paired_bootstrap_ci(season_lifts) if model != "market" else (None, None, None, None)
    summary_row = {
        "split": split_label,
        "model": model,
        "lambda": lam,
        "seasons": len(seasons),
        "board_slots": len(board_items),
        "zero_game_busts_on_board": sum(x["next_games"] == 0 for _, x in board_items),
        "actual_top_150_captured": sum(x["next_rank"] <= BOARD_SIZE for _, x in board_items),
        "top_150_capture_pct": round(100 * sum(x["next_rank"] <= BOARD_SIZE for _, x in board_items) / len(board_items), 2) if board_items else None,
        "board_mean_adj_fppg": round(mean(x["next_adj_fppg"] for _, x in board_items), 3) if board_items else None,
        "mean_abs_rank_error": round(mean(abs(rank - x["next_rank"]) for rank, x in board_items), 2) if board_items else None,
        "swaps_count": len(added),
        "added_top150_pct": round(100 * sum(x["next_rank"] <= BOARD_SIZE for x in added) / len(added), 1) if added else None,
        "dropped_top150_pct": round(100 * sum(x["next_rank"] <= BOARD_SIZE for x in dropped) / len(dropped), 1) if dropped else None,
        "added_minus_dropped_fppg": round(mean(x["next_adj_fppg"] for x in added) - mean(x["next_adj_fppg"] for x in dropped), 2) if (added and dropped) else None,
        "risers_10plus": len(risers),
        "risers_avg_fppg_vs_adp": round(mean(x["fppg_above_adp"] for x in risers), 2) if risers else None,
        "fallers_10plus": len(fallers),
        "fallers_avg_fppg_vs_adp": round(mean(x["fppg_above_adp"] for x in fallers), 2) if fallers else None,
        "vs_market_board_fppg_p": round(p_val, 4) if p_val is not None else None,
    }
    return summary_row, rankings


def run_cv_folds(train_pool, eval_pool, cv_seasons):
    cv_scored = []
    for season in cv_seasons:
        earlier = [r for r in train_pool if r["draft_season"] < season]
        candidates = [r.copy() for r in eval_pool if r["draft_season"] == season]
        if len(earlier) < 40 or len({r["beat_adp"] for r in earlier}) < 2 or not candidates:
            continue
        model_scores, _ = fit_and_score_models(earlier, candidates)
        for i, row in enumerate(candidates):
            row["split"] = "cv"
            row["training_through"] = max(r["draft_season"] for r in earlier)
            for name, vals in model_scores.items():
                row[f"{name}_score"] = round(float(vals[i]), 6)
        cv_scored.extend(candidates)
    return cv_scored


def generate_live_board(data, target_season, board_model, board_lambda, max_train_adp, max_pool_adp, cv_folds, exclude_players):
    """Train on all prior completed seasons and export a returning-player-only Top-150 draft board CSV."""
    hist_rows, _ = build_completed_dataset(data, max_pool_adp=max_pool_adp)
    prior_seasons = sorted({r["draft_season"] for r in hist_rows if r["draft_season"] < target_season})
    if not prior_seasons:
        raise ValueError(f"no completed draft seasons strictly before {target_season}")

    train_pool = [r for r in hist_rows if r["draft_season"] in prior_seasons and r["adp"] <= max_train_adp]
    if len({r["beat_adp"] for r in train_pool}) < 2:
        raise ValueError(f"training drafts before {target_season} need both ADP hits and misses")

    if board_lambda is None:
        if len(prior_seasons) >= 2:
            cv_seasons = prior_seasons[1:][-cv_folds:]
            eval_hist = [r for r in hist_rows if r["draft_season"] in prior_seasons and r["adp"] <= max_pool_adp]
            cv_scored = run_cv_folds(train_pool, eval_hist, cv_seasons)
            lam = tune_global_lambdas(cv_scored).get(board_model, 0.65) if cv_scored else 0.65
        else:
            lam = 0.65
    else:
        lam = board_lambda

    excluded_keys = [name_key(p) for p in exclude_players.split(",") if p.strip()] if exclude_players else []
    candidates, _, skipped_rookies, removed = build_live_candidates(
        data, target_season, max_pool_adp=max_pool_adp, excluded_keys=excluded_keys,
    )
    scores, meta = fit_and_score_models(train_pool, candidates)
    for i, row in enumerate(candidates):
        for name, vals in scores.items():
            row[f"{name}_score"] = float(vals[i])

    ordered = build_global_board(candidates, board_model, lam)
    # Median scaled edge across the market top 150 (centers peer-relative edge without shifting board ranks)
    peer_baseline = median(adj for _, adj, rec in ordered if rec["market_rank"] <= BOARD_SIZE)

    board_rows = []
    for rank, (proj_fppg, adj, rec) in enumerate(ordered, start=1):
        rel_edge = adj - peer_baseline
        row = {
            "board_rank": rank,
            "in_top_150": rank <= BOARD_SIZE,
            "player": rec["player"],
            "adp": rec["adp"],
            "market_rank": rec["market_rank"],
            "slot_move": rec["market_rank"] - rank,
            "proj_adj_fppg": round(proj_fppg, 2),
            "mkt_expected_adj_fppg": round(rec["mkt_expected_adj_fppg"], 2),
            "edge_vs_peers": round(rel_edge, 2),
            "edge_vs_adp": round(adj, 2),
            "market_collapse_flag": rec["market_collapse_excess"] > 0,
            "market_collapse_excess": round(rec["market_collapse_excess"], 2),
            "huber_edge": round(rec["huber_clean_score"], 2),
            "decomposed_edge": round(rec["decomposed_clean_score"], 2),
            "age": rec["age_at_prior_break"],
            "prior_games": rec["prior_games"],
            "prior_adj_fppg": round(rec["prior_adj_fppg"], 2),
            "prior_adj_rank": rec["prior_adj_rank"],
            "prior_fppg": round(rec["prior_fppg"], 2),
            "prior_mpg": round(rec["prior_mpg"], 1) if rec["prior_mpg"] is not None else None,
            "espn_vs_avg": round(rec["espn_vs_avg"], 1) if rec["espn_vs_avg"] is not None else None,
        }
        if "next_adj_fppg" in rec:
            row.update({
                "actual_games": rec["next_games"],
                "actual_adj_rank": rec["next_rank"],
                "actual_adj_fppg": rec["next_adj_fppg"],
                "adj_fppg_above_adp": rec["fppg_above_adp"],
            })
        board_rows.append(row)

    out_path = data / f"draft_board_{target_season}.csv"
    write_csv(out_path, board_rows)

    w_huber, w_decomp = meta["stack_weights"]
    print(f"\n=== Generated {target_season} Returning-Player Draft Board ({board_model}, lambda={lam:.2f}) ===")
    print(f"Trained on {len(prior_seasons)} prior drafts: {', '.join(prior_seasons)}")
    print(f"2-model stacked weights (huber_clean / decomposed_clean): {w_huber:.2f} / {w_decomp:.2f}")
    print(f"Scored {len(candidates)} returning players ({skipped_rookies} rookies/unmatched omitted); peer edge center: {peer_baseline:+.2f}")
    if removed:
        print(f"Manually excluded via --exclude-players: {', '.join(removed)}")
    coll_w = meta["huber_coefs"].get("market_collapse_excess")
    cap_w = meta["huber_coefs"].get("prior_minus_mkt_capped")
    if coll_w is not None and cap_w is not None:
        print(f"Learned injury-trap check -> prior_minus_mkt_capped: {cap_w:+.2f} | market_collapse_excess (>+8 FPPG & >=20 slot drop): {coll_w:+.2f}")

    top150 = board_rows[:BOARD_SIZE]
    flagged = [r for r in board_rows if r["market_collapse_flag"] and r["market_rank"] <= 175]
    targets = sorted(
        [r for r in top150 if r["edge_vs_peers"] > 0 and not r["market_collapse_flag"]],
        key=lambda r: (-r["edge_vs_peers"], -r["slot_move"]),
    )[:10]
    fades = sorted(
        [r for r in board_rows if r["market_rank"] <= BOARD_SIZE and r["edge_vs_peers"] < 0],
        key=lambda r: (r["edge_vs_peers"], r["slot_move"]),
    )[:10]
    entrants = [r for r in top150 if r["market_rank"] > BOARD_SIZE and r["edge_vs_peers"] > 0]

    if flagged:
        print(f"\nAuto-Detected Market-Collapse / Likely-Injured Stars (prior AdjFPPG > mkt + {COLLAPSE_THRESHOLD:.0f} & >={int(COLLAPSE_MIN_SLOT_DROP)} slot drop):")
        for r in flagged[:10]:
            print(
                f"  #{r['board_rank']:3d} (ADP {r['adp']:5.1f}, {r['slot_move']:+3d} slots) | {r['player']:24s} | "
                f"PriorAdj {r['prior_adj_fppg']:5.1f} (#{r['prior_adj_rank']}) vs Mkt {r['mkt_expected_adj_fppg']:5.1f} | "
                f"{r['edge_vs_peers']:+.2f} vs peers ({r['edge_vs_adp']:+.2f} raw)"
            )

    print("\nTop 10 Model Targets (Highest Edge vs Draft Peers):")
    for r in targets:
        print(
            f"  #{r['board_rank']:3d} (ADP {r['adp']:5.1f}, {r['slot_move']:+3d} slots) | {r['player']:24s} | "
            f"Proj {r['proj_adj_fppg']:5.2f} | {r['edge_vs_peers']:+.2f} vs peers ({r['edge_vs_adp']:+.2f} raw)"
        )

    print("\nTop 10 Model Fades (Steepest Negative Edge vs Draft Peers):")
    for r in fades:
        tag = " [CUT]" if not r["in_top_150"] else ""
        print(
            f"  #{r['board_rank']:3d} (ADP {r['adp']:5.1f}, {r['slot_move']:+3d} slots) | {r['player']:24s} | "
            f"Proj {r['proj_adj_fppg']:5.2f} | {r['edge_vs_peers']:+.2f} vs peers ({r['edge_vs_adp']:+.2f} raw){tag}"
        )

    if entrants:
        print(f"\nLate Entrants Promoted into Top 150 ({len(entrants)} players):")
        for r in entrants:
            print(
                f"  #{r['board_rank']:3d} (ADP {r['adp']:5.1f}, {r['slot_move']:+3d} slots) | {r['player']:24s} | "
                f"Proj {r['proj_adj_fppg']:5.2f} | {r['edge_vs_peers']:+.2f} vs peers ({r['edge_vs_adp']:+.2f} raw)"
            )

    print(f"\nSaved full ranked draft board to {out_path}")


def run_backtest(data, holdout_seasons=2, cv_folds=6, top_pct=25.0, max_train_adp=175.0, max_pool_adp=225.0):
    rows, _ = build_completed_dataset(data, max_pool_adp=max_pool_adp)
    seasons = sorted({r["draft_season"] for r in rows})
    if len(seasons) <= holdout_seasons:
        raise ValueError(f"need more completed draft seasons than --holdout-seasons ({len(seasons)} found)")

    development, holdout = seasons[:-holdout_seasons], seasons[-holdout_seasons:]
    train_pool = [r for r in rows if r["draft_season"] in development and r["adp"] <= max_train_adp]
    eval_pool = [r for r in rows if r["adp"] <= max_pool_adp]
    if len({r["beat_adp"] for r in train_pool}) < 2:
        raise ValueError("development seasons need both ADP hits and misses")

    cv_seasons = development[1:][-cv_folds:]
    cv_scored = run_cv_folds(train_pool, eval_pool, cv_seasons)
    if cv_scored:
        attach_selection_flags(cv_scored, top_pct)

    best_lambdas = tune_global_lambdas(cv_scored)

    holdout_scored = [r.copy() for r in eval_pool if r["draft_season"] in holdout]
    holdout_scores, holdout_meta = fit_and_score_models(train_pool, holdout_scored)
    for i, row in enumerate(holdout_scored):
        row["split"] = "holdout"
        row["training_through"] = development[-1]
        for name, vals in holdout_scores.items():
            row[f"{name}_score"] = round(float(vals[i]), 6)
    attach_selection_flags(holdout_scored, top_pct)

    band_summary, banded_summary, global_summary, global_rankings, lambda_sweep = [], [], [], [], []
    for split_label, split_rows in (("cv", cv_scored), ("holdout", holdout_scored)):
        if not split_rows:
            continue
        band_summary.extend(evaluate_band_picks(split_rows, split_label, top_pct))
        banded_summary.extend(evaluate_banded_boards(split_rows, split_label))
        for model in ("market", *MODEL_NAMES):
            g_row, g_rank = evaluate_global_board_single(split_rows, split_label, model, best_lambdas[model])
            global_summary.append(g_row)
            global_rankings.extend(g_rank)
        for lam in (0.0, *LAMBDA_GRID):
            m = "market" if lam == 0.0 else "ensemble_stacked"
            s_row, _ = evaluate_global_board_single(split_rows, split_label, m, lam)
            lambda_sweep.append(s_row)

    write_csv(data / "backtest_final_band_summary.csv", band_summary)
    write_csv(data / "backtest_final_banded_board_summary.csv", banded_summary)
    write_csv(data / "backtest_final_global_board_summary.csv", global_summary)
    write_csv(data / "backtest_final_global_board_rankings.csv", global_rankings)
    write_csv(data / "backtest_final_lambda_sweep.csv", lambda_sweep)
    write_csv(data / "backtest_final_results.csv", holdout_scored)
    if cv_scored:
        write_csv(data / "backtest_final_cv_results.csv", cv_scored)

    w_huber, w_decomp = holdout_meta["stack_weights"]
    print(f"\nTarget: Availability-Adjusted FPPG (ref={int(REF_GAMES)}g, waiver={WAIVER_FACTOR:.2f}x 150th, 0-game busts included; rookies omitted)")
    print(f"Ensemble: 2-model blend (huber_clean + decomposed_clean; logistic_base kept as standalone benchmark only)")
    print(f"Development drafts: {', '.join(development)}")
    print(f"Holdout drafts:     {', '.join(holdout)}")
    print(f"2-model stacked weights (huber_clean / decomposed_clean): {w_huber:.2f} / {w_decomp:.2f}")
    top_coefs = sorted(holdout_meta["huber_coefs"].items(), key=lambda kv: -abs(kv[1]))
    print("Top Huber standardized weights: " + ", ".join(f"{k}={v:+.2f}" for k, v in top_coefs[:8]))

    for split_label in ("cv", "holdout"):
        b_rows = [r for r in banded_summary if r["split"] == split_label]
        if b_rows:
            print(f"\n=== BANDED Top-150 Board ({split_label.upper()}, {b_rows[0]['seasons']} seasons, Returning Players Only) ===")
            for r in b_rows:
                pval = f", p={r['vs_market_p_value']:.3f}" if r["vs_market_p_value"] is not None else ""
                print(
                    f"  {r['model']:16s} | Top150: {r['actual_top_150_captured']:3d}/{r['board_slots']} "
                    f"({r['top_150_capture_pct']:5.2f}%) | MAE: {r['mean_abs_rank_error']:5.2f} | "
                    f"Targets: {r['targets_hit_rate_pct']:5.1f}% beat ADP, "
                    f"{r['targets_avg_adj_fppg_above_adp']:+.3f} AdjFPPG vs ADP (0g busts: {r['targets_zero_game_busts']}){pval}"
                )

        g_rows = [r for r in global_summary if r["split"] == split_label]
        if not g_rows:
            continue
        print(f"\n=== GLOBAL UNBANDED Top-150 Board ({split_label.upper()}, {g_rows[0]['seasons']} seasons, Returning Players Only) ===")
        for r in g_rows:
            swap_str = (
                f"Swaps({r['swaps_count']:2d}): +{r['added_top150_pct']:4.1f}% vs -{r['dropped_top150_pct']:4.1f}% Top150 "
                f"({r['added_minus_dropped_fppg']:+.2f} FPPG)"
                if r["swaps_count"] and r["added_minus_dropped_fppg"] is not None
                else "Swaps( 0): market baseline"
            )
            pval = f" p={r['vs_market_board_fppg_p']:.3f}" if r["vs_market_board_fppg_p"] is not None else ""
            print(
                f"  {r['model']:16s} (lam={r['lambda']:.2f}) | Top150: {r['actual_top_150_captured']:3d}/{r['board_slots']} "
                f"({r['top_150_capture_pct']:5.2f}%) | AvgAdjFPPG: {r['board_mean_adj_fppg']:5.2f}{pval} | "
                f"MAE: {r['mean_abs_rank_error']:5.2f} | 0g:{r['zero_game_busts_on_board']:2d} | {swap_str}"
            )

        print(f"\n=== Global Lambda Sweep for ensemble_stacked ({split_label.upper()}) ===")
        for r in [x for x in lambda_sweep if x["split"] == split_label]:
            r_val = f"{r['risers_avg_fppg_vs_adp']:+.2f}" if r["risers_avg_fppg_vs_adp"] is not None else "  n/a"
            f_val = f"{r['fallers_avg_fppg_vs_adp']:+.2f}" if r["fallers_avg_fppg_vs_adp"] is not None else "  n/a"
            swap_fppg = f"{r['added_minus_dropped_fppg']:+.2f}" if r["added_minus_dropped_fppg"] is not None else "  n/a"
            print(
                f"  lam={r['lambda']:.2f} | Top150: {r['actual_top_150_captured']:3d}/{r['board_slots']} "
                f"| AvgAdjFPPG: {r['board_mean_adj_fppg']:5.2f} | MAE: {r['mean_abs_rank_error']:5.2f} "
                f"| Swaps({r['swaps_count']:2d}): {swap_fppg} FPPG | Risers({r['risers_10plus']:2d}): {r_val} / Fallers({r['fallers_10plus']:2d}): {f_val}"
            )

        print(f"\n=== Price-Matched AdjFPPG Lift by ADP Band ({split_label.upper()}, top {top_pct:g}%) ===")
        bands = sorted({r["adp_band"] for r in band_summary if r["split"] == split_label}, key=lambda s: int(s.split("-")[0]))
        header = f"  {'Band':7s} | " + " | ".join(f"{m:16s}" for m in MODEL_NAMES)
        print(header)
        print("  " + "-" * (len(header) - 2))
        lookup = {(r["adp_band"], r["model"]): r for r in band_summary if r["split"] == split_label}
        for band in bands:
            cells = []
            for m in MODEL_NAMES:
                rec = lookup.get((band, m))
                lift = rec["price_matched_fppg_lift"] if rec else None
                cells.append(f"{lift:+16.2f}" if lift is not None else f"{'n/a':>16s}")
            print(f"  {band:7s} | " + " | ".join(cells))

    print(f"\nSaved outputs under {data}/backtest_final_*.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"), help="download folder (default: data)")
    parser.add_argument("--holdout-seasons", type=int, default=2, help="latest draft seasons held out (default: 2)")
    parser.add_argument("--cv-folds", type=int, default=6, help="max chronological CV folds (default: 6)")
    parser.add_argument("--top-pct", type=float, default=25.0, help="selection pct for band checks (default: 25)")
    parser.add_argument("--max-train-adp", type=float, default=175.0, help="max ADP used to fit models (default: 175)")
    parser.add_argument("--max-pool-adp", type=float, default=225.0, help="max ADP allowed into global Top-150 pool (default: 225)")
    parser.add_argument("--generate-board", metavar="YYYY-YY", help="train on all prior seasons and export data/draft_board_<season>.csv")
    parser.add_argument("--board-model", choices=MODEL_NAMES, default="ensemble_stacked", help="model used for --generate-board (default: ensemble_stacked)")
    parser.add_argument("--board-lambda", type=float, default=None, help="override global residual scale lambda for --generate-board")
    parser.add_argument("--exclude-players", default="", help="comma-separated known injured/out players to exclude from --generate-board")
    args = parser.parse_args()
    if args.holdout_seasons < 1 or args.cv_folds < 1 or not 0 < args.top_pct <= 100:
        parser.error("use valid positive holdout-seasons, cv-folds, and 0 < top-pct <= 100")
    if args.board_lambda is not None and args.board_lambda < 0:
        parser.error("--board-lambda must be nonnegative")
    try:
        if args.generate_board:
            generate_live_board(
                args.data, args.generate_board, args.board_model, args.board_lambda,
                args.max_train_adp, args.max_pool_adp, args.cv_folds, args.exclude_players,
            )
        else:
            run_backtest(
                args.data, args.holdout_seasons, args.cv_folds, args.top_pct,
                args.max_train_adp, args.max_pool_adp,
            )
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(1, f"Final board pipeline failed: {error}\n")


if __name__ == "__main__":
    main()
