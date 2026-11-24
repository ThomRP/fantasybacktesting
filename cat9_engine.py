"""Unified 9-Category Fantasy Basketball Draft Engine (Roto + H2H Each-Cat + H2H Most-Cat).

Empirically validated across 8 NBA seasons (2018-19 through 2025-26):
  - Scoring the exact same 12-team rosters under Roto vs. H2H Each-Cat vs. H2H Most-Cat yields
    r = 0.94-0.985 team-score correlation (Roto #1 finishes Top-3 in H2H Most-Cat 95.2% of the time).
  - Therefore, ONE unified zero-leakage 9-Cat projection model powers all three formats, with
    optional draft-day punting (--punt FT,TO) and Reach-vs.-Wait survival optimization layered on top.

Categories (9): PTS, REB, AST, STL, BLK, 3PM, FG% (volume-weighted), FT% (volume-weighted), TO (inverted).
"""

import argparse
import csv
import math
import random
import re
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest_common import (
    name_key,
    player_box_scores,
    read_csv,
    regular_season_game,
    write_csv,
)
import draft_engine as de


# 11 raw box-score stats tracked daily/weekly per player
STAT_KEYS_11 = (
    "points",
    "rebounds",
    "assists",
    "steals",
    "blocks",
    "three_point_field_goals_made",
    "field_goals_made",
    "field_goals_attempted",
    "free_throws_made",
    "free_throws_attempted",
    "turnovers",
)
CAT_NAMES = ("PTS", "REB", "AST", "STL", "BLK", "3PM", "FG%", "FT%", "TO")
CAT_ALIASES = {
    "PTS": 0, "POINTS": 0,
    "REB": 1, "REBS": 1, "REBOUNDS": 1,
    "AST": 2, "ASTS": 2, "ASSISTS": 2,
    "STL": 3, "STLS": 3, "STEALS": 3,
    "BLK": 4, "BLKS": 4, "BLOCKS": 4,
    "3PM": 5, "3PT": 5, "3PTM": 5, "THREES": 5, "TPM": 5,
    "FG": 6, "FG%": 6, "FGP": 6, "FG_PCT": 6,
    "FT": 7, "FT%": 7, "FTP": 7, "FT_PCT": 7,
    "TO": 8, "TOV": 8, "TOS": 8, "TURNOVERS": 8,
}

# Category-specific Empirical-Bayes year-over-year persistence weights for per-cat Z-shape projection
CAT_YOY_SHRINK = np.array([0.74, 0.74, 0.76, 0.56, 0.68, 0.74, 0.58, 0.72, 0.68], dtype=float)

BOARD_SIZE = 150
REF_GAMES_WAIVER = 65.0
FULL_SEASON_GAMES = 82.0
MAX_RESIDUAL_VET_ADP = 175.0
MAX_RESIDUAL_ROOKIE_ADP = 200.0
COLLAPSE_SPREAD_RATIO = 0.25
COLLAPSE_MIN_SLOT_DROP = 20.0
DEFAULT_LAM = 0.50
DEFAULT_WAIT_LAM = 0.85
DEFAULT_TO_WEIGHT = 0.75

VET_CAT9_RESIDUAL_FEATURES = (
    "log_adp",
    "z_mkt",
    "z_prior_minus_mkt_capped",
    "z_collapse_excess",
    "z_prior_tot",
    "z_prior_raw",
    "z_prior_fg_imp",
    "z_prior_ft_imp",
    "z_prior_to",
    "z_yoy_delta",
    "age_at_prior_break",
    "age_under_24",
    "age_over_30",
    "prior_games",
    "prior_mpg",
    "prior_start_pct",
    "prior_post_break_mpg_chg",
    "prior_tp_pct",
    "age30_x_low_g",
    "young_x_post_mpg",
    "log_adp_x_age30",
    "yoy_mpg_delta",
    "two_yr_games_avg",
    "no_prev2_season",
    "adp_platform_std",
    "espn_vs_avg",
)

VET_CAT9_PG_FEATURES = (
    "log_adp",
    "z_mkt",
    "z_prior_raw",
    "z_prior_tot",
    "z_prior_minus_mkt_capped",
    "z_collapse_excess",
    "z_prior_fg_imp",
    "z_prior_ft_imp",
    "z_prior_to",
    "z_yoy_delta",
    "age_at_prior_break",
    "age_under_24",
    "age_over_30",
    "prior_mpg",
    "prior_start_pct",
    "prior_post_break_mpg_chg",
    "prior_tp_pct",
    "yoy_mpg_delta",
    "no_prev2_season",
    "adp_platform_std",
    "espn_vs_avg",
)

VET_CAT9_AVAIL_FEATURES = (
    "log_adp",
    "z_mkt",
    "z_prior_minus_mkt_capped",
    "z_collapse_excess",
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

ROOKIE_CAT9_FEATURES = (
    "log_draft_pick",
    "log_adp",
    "z_mkt",
    "adp_minus_pick_expected",
    "is_redshirt",
    "rookie_age",
    "team_top8_avg_age",
    "team_vegas_win_pct",
    "team_vet30_min_share",
    "team_prior_top150_count",
    "espn_vs_avg",
)

CAT9_STRATEGIES = (
    "market",
    "residual_unified",
)


# ==============================================================================
# 9-Cat Math & Signed Z-Score Interpolation
# ==============================================================================

def parse_punt_cats(punt_str):
    """Parse comma-separated category names (e.g., 'FT,TO') into sorted index tuple."""
    if not punt_str:
        return ()
    out = set()
    for tok in str(punt_str).split(","):
        key = tok.strip().upper()
        if not key:
            continue
        if key not in CAT_ALIASES:
            valid = ", ".join(CAT_NAMES)
            raise ValueError(f"unknown 9-cat category {tok!r} (valid: {valid})")
        out.add(CAT_ALIASES[key])
    return tuple(sorted(out))


def make_cat_weights(to_weight=DEFAULT_TO_WEIGHT, punt_indices=()):
    w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, float(to_weight)], dtype=float)
    for idx in punt_indices:
        w[idx] = 0.0
    return w


def z_slot_value(lb_desc, adp):
    """Signed linear interpolation on a descending Z-score leaderboard at 1-based fractional ADP.

    Unlike backtest_common.adp_slot_points (which clamps at >= 0.0 for positive FPTS),
    this preserves negative Z-scores past ADP #90.
    """
    if not lb_desc:
        return 0.0
    pos = max(0.0, min(float(len(lb_desc) - 1), float(adp) - 1.0))
    lo = int(math.floor(pos))
    hi = min(len(lb_desc) - 1, lo + 1)
    frac = pos - lo
    return (1.0 - frac) * float(lb_desc[lo]) + frac * float(lb_desc[hi])


def stats11_to_cat9_impact(arr11, lg_fg, lg_ft):
    """Convert raw 11-stat array (..., 11) to 9-cat volume-weighted values (higher is always better)."""
    a = np.asarray(arr11, dtype=float)
    fg_imp = a[..., 6] - float(lg_fg) * a[..., 7]
    ft_imp = a[..., 8] - float(lg_ft) * a[..., 9]
    return np.stack([
        a[..., 0], a[..., 1], a[..., 2], a[..., 3], a[..., 4], a[..., 5],
        fg_imp, ft_imp, -a[..., 10],
    ], axis=-1)


def team_cat9_from_stats11(arr11):
    """Convert team 11-stat totals (..., 11) to actual 9-cat standings metrics (higher is better)."""
    a = np.asarray(arr11, dtype=float)
    fgp = np.where(a[..., 7] > 0.0, a[..., 6] / np.maximum(a[..., 7], 1e-9), 0.0)
    ftp = np.where(a[..., 9] > 0.0, a[..., 8] / np.maximum(a[..., 9], 1e-9), 0.0)
    return np.stack([
        a[..., 0], a[..., 1], a[..., 2], a[..., 3], a[..., 4], a[..., 5],
        fgp, ftp, -a[..., 10],
    ], axis=-1)


def compute_cat9_zscores(p_tot, p_g, top_pool_n=180, min_qual_games=20):
    """Compute standardized per-game (`z_raw`), season-total (`z_tot`), and 65g-waiver (`z_adj`) 9-Cat Z-vectors."""
    qual = [aid for aid, g in p_g.items() if g >= min_qual_games]
    if len(qual) < 30:
        qual = [aid for aid, g in p_g.items() if g > 0]
    if not qual:
        empty9 = np.zeros(9, dtype=float)
        return {
            "z_raw": {}, "z_tot": {}, "z_adj": {},
            "zero_tot_vec": empty9, "waiver_raw_vec": empty9, "waiver_pg_11": np.zeros(11, dtype=float),
            "lg_fg": 0.46, "lg_ft": 0.78, "mu_tot": empty9, "sd_tot": np.ones(9, dtype=float),
            "mu_pg": empty9, "sd_pg": np.ones(9, dtype=float),
        }

    # 1. Per-game Z-scores over top-180 fantasy-relevant qualifiers
    def _fit_pg_pool(ids):
        pg11 = np.array([p_tot[a] / float(p_g[a]) for a in ids], dtype=float)
        lg_fg = float(pg11[:, 6].sum() / max(pg11[:, 7].sum(), 1e-9))
        lg_ft = float(pg11[:, 8].sum() / max(pg11[:, 9].sum(), 1e-9))
        imp = stats11_to_cat9_impact(pg11, lg_fg, lg_ft)
        mu = imp.mean(axis=0)
        sd = np.maximum(imp.std(axis=0), 1e-6)
        return mu, sd, lg_fg, lg_ft

    mu0, sd0, lg_fg0, lg_ft0 = _fit_pg_pool(qual)
    sel_w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25], dtype=float)
    qual_pg_scores = {}
    for a in qual:
        imp = stats11_to_cat9_impact(p_tot[a] / float(p_g[a]), lg_fg0, lg_ft0)
        qual_pg_scores[a] = float(((imp - mu0) / sd0) @ sel_w)
    top_pg_ids = sorted(qual, key=lambda a: -qual_pg_scores[a])[:top_pool_n]
    mu_pg, sd_pg, lg_fg_pg, lg_ft_pg = _fit_pg_pool(top_pg_ids)

    z_raw = {}
    for a, tot in p_tot.items():
        g = p_g.get(a, 0)
        if g > 0:
            imp = stats11_to_cat9_impact(tot / float(g), lg_fg_pg, lg_ft_pg)
            z_raw[a] = (imp - mu_pg) / sd_pg

    # Waiver-wire per-game baseline (ranks 145-175 of top-180 pool)
    w_lo = min(145, max(0, len(top_pg_ids) - 10))
    w_hi = min(175, len(top_pg_ids))
    waiver_ids = top_pg_ids[w_lo:w_hi] if w_hi > w_lo else top_pg_ids[-10:]
    waiver_raw_vec = np.mean([z_raw[a] for a in waiver_ids], axis=0)
    waiver_pg_11 = np.mean([p_tot[a] / float(p_g[a]) for a in waiver_ids], axis=0)

    z_adj = {}
    for a, zr in z_raw.items():
        eff = min(float(p_g[a]), REF_GAMES_WAIVER) / REF_GAMES_WAIVER
        z_adj[a] = eff * zr + (1.0 - eff) * waiver_raw_vec

    # 2. Season-total Z-scores over top-180 season-total qualifiers (directly matches Roto & H2H totals!)
    def _fit_tot_pool(ids):
        t11 = np.array([p_tot[a] for a in ids], dtype=float)
        lg_fg = float(t11[:, 6].sum() / max(t11[:, 7].sum(), 1e-9))
        lg_ft = float(t11[:, 8].sum() / max(t11[:, 9].sum(), 1e-9))
        imp = stats11_to_cat9_impact(t11, lg_fg, lg_ft)
        mu = imp.mean(axis=0)
        sd = np.maximum(imp.std(axis=0), 1e-6)
        return mu, sd, lg_fg, lg_ft

    t_mu0, t_sd0, t_fg0, t_ft0 = _fit_tot_pool(qual)
    qual_tot_scores = {}
    for a in qual:
        imp = stats11_to_cat9_impact(p_tot[a], t_fg0, t_ft0)
        qual_tot_scores[a] = float(((imp - t_mu0) / t_sd0) @ sel_w)
    top_tot_ids = sorted(qual, key=lambda a: -qual_tot_scores[a])[:top_pool_n]
    mu_tot, sd_tot, lg_fg_tot, lg_ft_tot = _fit_tot_pool(top_tot_ids)

    z_tot = {}
    for a, tot in p_tot.items():
        imp = stats11_to_cat9_impact(tot, lg_fg_tot, lg_ft_tot)
        z_tot[a] = (imp - mu_tot) / sd_tot
    zero_tot_vec = (np.zeros(9, dtype=float) - mu_tot) / sd_tot

    return {
        "z_raw": z_raw,
        "z_tot": z_tot,
        "z_adj": z_adj,
        "zero_tot_vec": zero_tot_vec,
        "waiver_raw_vec": waiver_raw_vec,
        "waiver_pg_11": waiver_pg_11,
        "lg_fg": lg_fg_tot,
        "lg_ft": lg_ft_tot,
        "mu_tot": mu_tot,
        "sd_tot": sd_tot,
        "mu_pg": mu_pg,
        "sd_pg": sd_pg,
    }


def load_cat9_season(folder, require_completed=True):
    """Load one season's daily/weekly 11-stat box scores and 9-Cat Z-scores."""
    games_path = folder / "games.csv"
    box_path = folder / "player_box_scores.csv"
    if not games_path.is_file() or not box_path.is_file():
        return None
    schedule = list(read_csv(games_path))
    if require_completed and not any(
        g.get("season_type") == "3" and (g.get("status_type_completed") or "").lower() == "true"
        for g in schedule
    ):
        return None

    game_dates = {}
    for g in schedule:
        if regular_season_game(g):
            game_dates[g["game_id"]] = date.fromisoformat(g["game_date"][:10])
    if not game_dates:
        return None

    dates_sorted = sorted(set(game_dates.values()))
    date_to_idx = {d: i for i, d in enumerate(dates_sorted)}
    wk0_mon = dates_sorted[0] - timedelta(days=dates_sorted[0].weekday())
    day_to_week = np.array([(d - wk0_mon).days // 7 for d in dates_sorted], dtype=np.int32)
    n_days = len(dates_sorted)
    n_wks = int(day_to_week.max()) + 1

    p_tot = defaultdict(lambda: np.zeros(11, dtype=np.float64))
    p_g = defaultdict(int)
    daily_11 = defaultdict(dict)  # aid -> {day_idx: ndarray(11)}
    wk_fga = np.zeros(n_wks, dtype=np.float64)

    for gid, aid, _disp, st, _pos, _starter, _tid in player_box_scores(folder, game_dates):
        d_idx = date_to_idx[game_dates[gid]]
        w_idx = int(day_to_week[d_idx])
        vals = np.array([float(st.get(k) or 0.0) for k in STAT_KEYS_11], dtype=np.float64)
        p_tot[aid] += vals
        p_g[aid] += 1
        wk_fga[w_idx] += vals[7]
        prev = daily_11[aid].get(d_idx)
        daily_11[aid][d_idx] = vals if prev is None else (prev + vals)

    # Valid regular H2H fantasy weeks (>= 40% of median league FGA volume; skips All-Star break/stub days)
    pos_fga = wk_fga[wk_fga > 0]
    med_fga = float(np.median(pos_fga)) if len(pos_fga) else 0.0
    valid_wks = np.where(wk_fga >= 0.40 * med_fga)[0]
    if len(valid_wks) == 0:
        valid_wks = np.arange(n_wks)

    zinfo = compute_cat9_zscores(p_tot, p_g)
    return {
        "season": folder.name,
        "n_days": n_days,
        "n_wks": n_wks,
        "day_to_week": day_to_week,
        "valid_wks": valid_wks,
        "p_tot": dict(p_tot),
        "p_g": dict(p_g),
        "daily_11": dict(daily_11),
        **zinfo,
    }


def load_all_cat9_seasons(data):
    """Load all completed NBA seasons' 9-Cat aggregates alongside draft_engine base datasets."""
    scoring = dict(de.SCORING_PRESETS["default"])
    seasons_dirs, season_data, births_cache, profiles_cache = de.load_all_completed_seasons(data, scoring)
    cat9_by_year = {}
    for year in sorted(season_data):
        obj = load_cat9_season(seasons_dirs[year], require_completed=True)
        if obj is not None:
            cat9_by_year[year] = obj
    return seasons_dirs, season_data, births_cache, profiles_cache, cat9_by_year


def _season_weighted_lb(cat9_s, w9):
    scores = {aid: float(vec @ w9) for aid, vec in cat9_s["z_tot"].items()}
    raw_scores = {aid: float(vec @ w9) for aid, vec in cat9_s["z_raw"].items()}
    lb_tot = sorted(scores.values(), reverse=True)
    lb_raw = sorted(raw_scores.values(), reverse=True)
    raw_rank_map = {}
    for place, val in enumerate(lb_raw, start=1):
        raw_rank_map.setdefault(val, place)
    zero_score = float(cat9_s["zero_tot_vec"] @ w9)
    waiver_raw_score = float(cat9_s["waiver_raw_vec"] @ w9)
    return scores, raw_scores, lb_tot, raw_rank_map, zero_score, waiver_raw_score


def build_cat9_adp_pools(data, seasons_dirs, season_data, births_cache, profiles_cache, cat9_by_year, w9, max_pool_adp=225.0):
    """Build enriched 9-Cat ADP candidate pools for every completed draft season."""
    _, _, base_adp_pools, played_before, history_available = de.build_all_training_datasets(
        data, seasons_dirs, season_data, births_cache, profiles_cache, max_pool_adp=max_pool_adp,
    )

    # Precompute weighted leaderboards per completed season
    lb_cache = {yr: _season_weighted_lb(c9, w9) for yr, c9 in cat9_by_year.items()}

    cat9_pools = {}
    for target_s in sorted(base_adp_pools):
        target_yr = int(target_s[:4])
        prior_yr = target_yr - 1
        prev2_yr = target_yr - 2
        if prior_yr not in cat9_by_year or target_yr not in cat9_by_year:
            continue
        pri_c9 = cat9_by_year[prior_yr]
        nxt_c9 = cat9_by_year[target_yr]
        p2_c9 = cat9_by_year.get(prev2_yr)

        pri_tot_map, pri_raw_map, pri_lb, pri_raw_rk, pri_zero, pri_waiv_raw = lb_cache[prior_yr]
        nxt_tot_map, nxt_raw_map, nxt_lb, nxt_raw_rk, nxt_zero, _ = lb_cache[target_yr]
        p2_tot_map = lb_cache[prev2_yr][0] if prev2_yr in lb_cache else {}

        spread = max(1.0, z_slot_value(pri_lb, 10.0) - z_slot_value(pri_lb, 150.0))
        col_thresh = COLLAPSE_SPREAD_RATIO * spread

        # Load raw Yahoo ADP for stress-test opponent drafting
        yahoo_map = {}
        adp_file = seasons_dirs[target_yr] / "adp.csv"
        if adp_file.is_file():
            for raw_r in read_csv(adp_file):
                yv = de._safe_float(raw_r.get("Yahoo"))
                nm = de.clean_adp_player_name(raw_r.get("player_name") or raw_r.get("Player") or "")
                if yv is not None and nm:
                    yahoo_map[name_key(nm)] = yv

        enriched = []
        for r in base_adp_pools[target_s]:
            row = r.copy()
            aid = row["athlete_id"]
            adp = float(row["adp"])
            z_mkt = z_slot_value(pri_lb, adp)
            nxt_z_mkt = z_slot_value(nxt_lb, adp)

            act_tot_vec = nxt_c9["z_tot"].get(aid, nxt_c9["zero_tot_vec"])
            act_tot = float(act_tot_vec @ w9)
            act_raw_vec = nxt_c9["z_raw"].get(aid)
            act_raw = float(act_raw_vec @ w9) if act_raw_vec is not None else None
            nxt_g = int(nxt_c9["p_g"].get(aid, 0))

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
            row["yahoo_adp"] = yahoo_map.get(name_key(row["player"]), adp)

            if row["player_type"] == "returning_vet" and pri_tot is not None:
                raw_gap = float(pri_tot) - z_mkt
                p_rk = pri_raw_rk.get(pri_raw, 500) if pri_raw is not None else 500
                slot_drop = adp - p_rk
                is_col = slot_drop >= COLLAPSE_MIN_SLOT_DROP and raw_gap > col_thresh
                row["z_prior_minus_mkt_capped"] = max(-1.25 * col_thresh, min(col_thresh, raw_gap))
                row["z_collapse_excess"] = (raw_gap - col_thresh) if is_col else 0.0
            else:
                row["z_prior_minus_mkt_capped"] = None
                row["z_collapse_excess"] = 0.0

            row["z_actual"] = act_tot
            row["z_actual_vec"] = act_tot_vec.copy()
            row["z_actual_raw"] = act_raw
            row["z_above_adp"] = act_tot - nxt_z_mkt
            row["next_avail_82"] = min(float(nxt_g), FULL_SEASON_GAMES) / FULL_SEASON_GAMES
            row["next_games"] = nxt_g
            # Next rank by total 9-Cat Z
            row["z_next_rank"] = sum(1 for v in nxt_lb if v > act_tot) + 1
            enriched.append(row)
        cat9_pools[target_s] = enriched

    return cat9_pools, lb_cache, played_before, history_available


# ==============================================================================
# Unified 9-Cat Zero-Leakage Residual Model (Vets + Rookies + Absence + Per-Cat)
# ==============================================================================

def _pos_group(pos_tuple):
    primary = pos_tuple[0] if pos_tuple else "SG"
    if primary == "C":
        return "C"
    if primary in ("PG", "SG"):
        return "G"
    return "F"


def fit_predict_cat9_unified(
    training_rows,
    evaluated_all,
    w9,
    lam=DEFAULT_LAM,
    lam_late=None,
    recency_halflife=None,
    drop_platform_feats=False,
):
    """Fit unified 9-Cat residual model (37.5% Huber + 37.5% Avail*PerGameZ + 25% Shallow L1 GBDT + Rookie Huber)."""
    tr_vets = [
        r for r in training_rows
        if r["player_type"] == "returning_vet" and r.get("adp") is not None and r["adp"] <= MAX_RESIDUAL_VET_ADP
    ]
    tr_rk = [
        r for r in training_rows
        if r["player_type"] == "rookie" and r.get("adp") is not None and r["adp"] <= MAX_RESIDUAL_ROOKIE_ADP
    ]
    tr_abs = [
        r for r in training_rows
        if r["player_type"] == "returning_absence" and r.get("adp") is not None and r["adp"] <= MAX_RESIDUAL_VET_ADP
    ]

    ev_vet_idx = [i for i, r in enumerate(evaluated_all) if r["player_type"] == "returning_vet" and r.get("adp") is not None]
    ev_rk_idx = [i for i, r in enumerate(evaluated_all) if r["player_type"] == "rookie" and r.get("adp") is not None]
    ev_abs_idx = [i for i, r in enumerate(evaluated_all) if r["player_type"] == "returning_absence" and r.get("adp") is not None]

    ev_vets = [evaluated_all[i] for i in ev_vet_idx]
    ev_rk = [evaluated_all[i] for i in ev_rk_idx]

    n = len(evaluated_all)
    raw_residuals = np.zeros(n, dtype=float)
    proj_avail = np.full(n, np.nan, dtype=float)
    proj_pg_z = np.full(n, np.nan, dtype=float)
    vet_coefs, rk_coefs = {}, {}

    plat_drop = {"espn_vs_avg", "adp_platform_std"} if drop_platform_feats else set()
    vet_cfg = tuple(f for f in VET_CAT9_RESIDUAL_FEATURES if f not in plat_drop)
    pg_cfg = tuple(f for f in VET_CAT9_PG_FEATURES if f not in plat_drop)
    av_cfg = tuple(f for f in VET_CAT9_AVAIL_FEATURES if f not in plat_drop)
    rk_cfg = tuple(f for f in ROOKIE_CAT9_FEATURES if f not in plat_drop)
    spline_set = {"log_adp", "z_mkt", "age_at_prior_break"}

    # 1. Returning Veterans: 37.5% Huber + 37.5% Structural (Avail * PerGameZ) + 25% Shallow L1 GBDT
    vet_peer_center = 0.0
    if tr_vets and ev_vets:
        y_vet = np.array([float(r["z_above_adp"]) for r in tr_vets], dtype=float)
        w_vet = de._recency_weights(tr_vets, recency_halflife)
        h_feats = de.active_features(tr_vets, vet_cfg)
        X_tr_h = de.to_matrix(tr_vets, h_feats)
        X_ev_h = de.to_matrix(ev_vets, h_feats)

        h_pipe = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            HuberRegressor(alpha=25.0, epsilon=1.35, max_iter=1000),
        )
        if w_vet is not None:
            h_pipe.fit(X_tr_h, y_vet, huberregressor__sample_weight=w_vet)
        else:
            h_pipe.fit(X_tr_h, y_vet)
        ev_h = h_pipe.predict(X_ev_h)
        vet_coefs = dict(zip(h_feats, map(float, h_pipe.named_steps["huberregressor"].coef_)))

        # Structural leg: predict next_avail_82 & next_per_game_z separately via GroupKFold spline-Ridge
        rate_rows = [r for r in tr_vets if r.get("z_actual_raw") is not None]
        w_rate = de._recency_weights(rate_rows, recency_halflife)
        pg_f = de.active_features(rate_rows, pg_cfg)
        av_f = de.active_features(tr_vets, av_cfg)
        X_pg = de.to_matrix(rate_rows, pg_f)
        y_pg = np.array([float(r["z_actual_raw"]) for r in rate_rows], dtype=float)
        X_av = de.to_matrix(tr_vets, av_f)
        y_av = np.array([float(r["next_avail_82"]) for r in tr_vets], dtype=float)

        def _fit_ridge(pipe, X_, y_, sw_):
            return pipe.fit(X_, y_, ridgecv__sample_weight=sw_) if sw_ is not None else pipe.fit(X_, y_)

        groups = np.array([r["draft_season"] for r in tr_vets])
        n_s = len(set(groups))
        oof_pg = np.zeros(len(tr_vets), dtype=float)
        oof_av = np.zeros(len(tr_vets), dtype=float)
        if n_s >= 2 and len(rate_rows) >= 30:
            for tr_i, va_i in GroupKFold(n_splits=min(n_s, 5)).split(tr_vets, y_av, groups):
                tr_sub = [tr_vets[i] for i in tr_i]
                va_sub = [tr_vets[i] for i in va_i]
                tr_r = [r for r in tr_sub if r.get("z_actual_raw") is not None]
                sw_r = de._recency_weights(tr_r, recency_halflife)
                sw_a = de._recency_weights(tr_sub, recency_halflife)
                oof_pg[va_i] = np.clip(
                    _fit_ridge(de._make_spline_ridge(pg_f, spline_set), de.to_matrix(tr_r, pg_f), [float(r["z_actual_raw"]) for r in tr_r], sw_r).predict(de.to_matrix(va_sub, pg_f)),
                    -10.0, 18.0,
                )
                oof_av[va_i] = np.clip(
                    _fit_ridge(de._make_spline_ridge(av_f, spline_set), de.to_matrix(tr_sub, av_f), [float(r["next_avail_82"]) for r in tr_sub], sw_a).predict(de.to_matrix(va_sub, av_f)),
                    0.0, 1.0,
                )
        else:
            oof_pg = np.clip(_fit_ridge(de._make_spline_ridge(pg_f, spline_set), X_pg, y_pg, w_rate).predict(de.to_matrix(tr_vets, pg_f)), -10.0, 18.0)
            oof_av = np.clip(_fit_ridge(de._make_spline_ridge(av_f, spline_set), X_av, y_av, w_vet).predict(X_av), 0.0, 1.0)

        ev_pg = np.clip(_fit_ridge(de._make_spline_ridge(pg_f, spline_set), X_pg, y_pg, w_rate).predict(de.to_matrix(ev_vets, pg_f)), -10.0, 18.0)
        ev_av = np.clip(_fit_ridge(de._make_spline_ridge(av_f, spline_set), X_av, y_av, w_vet).predict(de.to_matrix(ev_vets, av_f)), 0.0, 1.0)

        tr_zf = np.array([float(r["z_zero_floor"]) for r in tr_vets], dtype=float)
        ev_zf = np.array([float(r["z_zero_floor"]) for r in ev_vets], dtype=float)
        tr_struct = oof_av * oof_pg + (1.0 - oof_av) * tr_zf
        ev_struct = ev_av * ev_pg + (1.0 - ev_av) * ev_zf

        tr_mkt = np.array([float(r["z_mkt"]) for r in tr_vets], dtype=float)
        ev_mkt = np.array([float(r["z_mkt"]) for r in ev_vets], dtype=float)
        tr_la = np.array([float(r["log_adp"]) for r in tr_vets], dtype=float)
        ev_la = np.array([float(r["log_adp"]) for r in ev_vets], dtype=float)
        tr_c = np.array([float(r["z_collapse_excess"]) for r in tr_vets], dtype=float)
        ev_c = np.array([float(r["z_collapse_excess"]) for r in ev_vets], dtype=float)

        meta = make_pipeline(StandardScaler(), HuberRegressor(alpha=10.0, max_iter=1000))
        X_meta = np.column_stack([tr_struct - tr_mkt, (tr_struct - tr_mkt) * tr_la, tr_la, tr_c])
        if w_vet is not None:
            meta.fit(X_meta, y_vet, huberregressor__sample_weight=w_vet)
        else:
            meta.fit(X_meta, y_vet)
        ev_d = meta.predict(np.column_stack([ev_struct - ev_mkt, (ev_struct - ev_mkt) * ev_la, ev_la, ev_c]))

        if len(tr_vets) >= 80:
            gb = HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.03,
                max_iter=100,
                max_depth=2,
                min_samples_leaf=35,
                l2_regularization=20.0,
                max_bins=64,
                random_state=42,
            )
            if w_vet is not None:
                gb.fit(X_tr_h, y_vet, sample_weight=w_vet)
            else:
                gb.fit(X_tr_h, y_vet)
            ev_g = gb.predict(X_ev_h)
            ev_eq = 0.375 * ev_h + 0.375 * ev_d + 0.25 * ev_g
        else:
            ev_eq = 0.5 * ev_h + 0.5 * ev_d

        vet_peer_center = float(np.median(ev_eq))
        for pos, idx in enumerate(ev_vet_idx):
            raw_residuals[idx] = ev_eq[pos]
            proj_avail[idx] = ev_av[pos]
            proj_pg_z[idx] = ev_pg[pos]

    # 2. Incoming Rookies: 11-feature Huber with zero-leakage Rookie Hype Tax + Vegas Win%
    rk_flat_bias = 0.0
    if tr_rk and ev_rk:
        y_rk = np.array([float(r["z_above_adp"]) for r in tr_rk], dtype=float)
        rk_flat_bias = float(np.median(y_rk))
        tr_work = [r.copy() for r in tr_rk]
        ev_work = [r.copy() for r in ev_rk]
        tr_valid = [r for r in tr_work if r.get("log_draft_pick") is not None]
        if len(tr_valid) >= 8:
            p_mod = HuberRegressor(alpha=5.0, max_iter=500).fit(
                np.array([[float(r["log_draft_pick"])] for r in tr_valid], dtype=float),
                np.array([float(r["z_actual"]) for r in tr_valid], dtype=float),
            )
            fb = float(np.median([float(r["z_actual"]) for r in tr_valid]))
            for grp in (tr_work, ev_work):
                for r in grp:
                    pe = float(p_mod.predict([[float(r["log_draft_pick"])]])[0]) if r.get("log_draft_pick") is not None else fb
                    r["adp_minus_pick_expected"] = float(r["z_mkt"]) - pe
        else:
            for grp in (tr_work, ev_work):
                for r in grp:
                    r["adp_minus_pick_expected"] = None

        r_feats = de.active_features(tr_work, rk_cfg)
        if len(tr_work) >= 12 and r_feats:
            r_pipe = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                StandardScaler(),
                HuberRegressor(alpha=30.0, epsilon=1.35, max_iter=1000),
            )
            w_rk = de._recency_weights(tr_work, recency_halflife)
            if w_rk is not None:
                r_pipe.fit(de.to_matrix(tr_work, r_feats), y_rk, huberregressor__sample_weight=w_rk)
            else:
                r_pipe.fit(de.to_matrix(tr_work, r_feats), y_rk)
            ev_r = r_pipe.predict(de.to_matrix(ev_work, r_feats))
            rk_coefs = dict(zip(r_feats, map(float, r_pipe.named_steps["huberregressor"].coef_)))
        else:
            ev_r = np.full(len(ev_rk), rk_flat_bias, dtype=float)
        for pos, idx in enumerate(ev_rk_idx):
            raw_residuals[idx] = ev_r[pos]

    # 3. Returning-Absence Veterans: Empirical-Bayes shrunk median residual
    if len(tr_abs) >= 5:
        abs_med = float(np.median([float(r["z_above_adp"]) for r in tr_abs]))
        abs_shrink = len(tr_abs) / (len(tr_abs) + 10.0)
        abs_pred = abs_shrink * abs_med + (1.0 - abs_shrink) * vet_peer_center
    else:
        abs_pred = vet_peer_center
    for idx in ev_abs_idx:
        raw_residuals[idx] = abs_pred

    # Combine market expectation + clipped scaled residual (with late-ADP ramp)
    eff_lam_late = max(float(lam), DEFAULT_WAIT_LAM) if lam_late is None else float(lam_late)
    ref_spread = float(evaluated_all[0].get("z_spread") or 6.0) if evaluated_all else 6.0
    clip_val = max(1.8, 0.42 * ref_spread)

    final_proj = np.zeros(n, dtype=float)
    scaled_edges = np.zeros(n, dtype=float)
    for i, r in enumerate(evaluated_all):
        mkt = float(r["z_mkt"])
        row_lam = de._effective_residual_lam(r.get("adp"), lam, eff_lam_late)
        adj = row_lam * max(-clip_val, min(clip_val, float(raw_residuals[i])))
        scaled_edges[i] = adj
        final_proj[i] = mkt + adj

    # 4. Per-Category Projected Z-Vectors (for live boards, punt filters, and dynamic punt sim)
    # Build position-group shape priors strictly from training_rows (zero leakage)
    pos_shapes = defaultdict(list)
    for r in training_rows:
        av = r.get("z_actual_vec")
        if av is not None:
            pos_shapes[_pos_group(r.get("positions", ()))].append(av)
    pos_mean = {g: np.mean(vs, axis=0) for g, vs in pos_shapes.items() if vs}
    global_shape = np.mean([r["z_actual_vec"] for r in training_rows if r.get("z_actual_vec") is not None], axis=0) if training_rows else np.zeros(9)

    w_sum = max(float(np.sum(w9)), 1e-6)
    proj_cat_z = np.zeros((n, 9), dtype=float)
    for i, r in enumerate(evaluated_all):
        pg = _pos_group(r.get("positions", ()))
        prior_shape = pos_mean.get(pg, global_shape)
        if r.get("z_prior_vec") is not None:
            base_vec = CAT_YOY_SHRINK * np.asarray(r["z_prior_vec"], dtype=float) + (1.0 - CAT_YOY_SHRINK) * prior_shape
        else:
            base_vec = prior_shape.copy()
        # Shift uniformly across active (weighted) categories so base_vec @ w9 == final_proj[i]
        cur_tot = float(base_vec @ w9)
        shift = (final_proj[i] - cur_tot) / w_sum
        proj_cat_z[i] = base_vec + shift * np.where(w9 > 0, 1.0, 0.0)

    return {
        "residual_proj": final_proj,
        "scaled_edges": scaled_edges,
        "raw_residuals": raw_residuals,
        "proj_cat_z": proj_cat_z,
        "proj_avail": proj_avail,
        "proj_pg_z": proj_pg_z,
        "vet_coefs": vet_coefs,
        "rookie_coefs": rk_coefs,
        "rookie_flat_bias": rk_flat_bias,
        "clip_val": clip_val,
    }


def apply_cat9_injury_layer(rows, proj_tot, proj_cat_z, proj_avail, injury_records, target_year, w9, opening_night=None, as_of_date=None):
    """Apply live ESPN / CSV return-date availability caps onto 9-Cat projections & per-cat Z-vectors."""
    by_id, by_key = {}, {}
    for rec in injury_records:
        if rec.get("athlete_id"):
            by_id[str(rec["athlete_id"])] = rec
        for k in rec.get("name_keys", ()):
            by_key[k] = rec

    adj_tot = np.array(proj_tot, dtype=float)
    adj_cat = np.array(proj_cat_z, dtype=float)
    labels = [""] * len(rows)
    capped_count = 0

    for i, r in enumerate(rows):
        inj = by_id.get(str(r.get("athlete_id") or "")) or by_key.get(name_key(r["player"]))
        if inj is None:
            continue
        avail_cap, _, label = de.evaluate_injury_availability_cap(
            inj, target_year, opening_night=opening_night, as_of_date=as_of_date,
        )
        labels[i] = label
        if avail_cap is None:
            continue

        z_floor = float(r.get("z_zero_floor") or -5.5)
        if avail_cap <= 0.0:
            if z_floor < adj_tot[i]:
                adj_tot[i] = z_floor
                # Zero counting impact: scale counting cats toward zero floor
                adj_cat[i] = np.minimum(adj_cat[i], np.array([-1.2, -1.0, -0.9, -1.0, -0.8, -0.9, 0.0, 0.0, 0.8]))
                capped_count += 1
            continue

        exp_av = float(proj_avail[i]) if math.isfinite(proj_avail[i]) and proj_avail[i] > 0.15 else 0.78
        if avail_cap < exp_av:
            ratio = max(0.0, min(1.0, avail_cap / exp_av))
            capped_tot = ratio * adj_tot[i] + (1.0 - ratio) * z_floor
            if capped_tot < adj_tot[i]:
                adj_tot[i] = capped_tot
                # Scale positive counting/percentage contributions proportionally; missed games reduce TO volume
                adj_cat[i, :8] = ratio * adj_cat[i, :8] + (1.0 - ratio) * np.array([-1.0, -0.8, -0.7, -0.8, -0.6, -0.7, 0.0, 0.0])
                adj_cat[i, 8] = ratio * adj_cat[i, 8] + (1.0 - ratio) * 0.6
                capped_count += 1

    return adj_tot, adj_cat, labels, capped_count


# ==============================================================================
# Exact Daily-Active-Lineup 9-Cat Multi-Format Scorer & 12-Team Simulator
# ==============================================================================

def score_roster_cat9_active(roster, bundle):
    """Aggregate exact daily 10-starter + 3-bench 11-stat totals (season + weekly) for one roster."""
    key = tuple(sorted(roster))
    cached = bundle["roster_cache"].get(key)
    if cached is not None:
        return cached

    prio = bundle["prio"]
    ordered = sorted(roster, key=lambda idx: -prio[idx])
    played_mat = bundle["played_mat"]
    counts = played_mat[ordered].sum(axis=0)
    crowded_days = np.flatnonzero(counts >= 5).tolist()
    played_lists = bundle["played_lists"]
    masks10 = bundle["masks10"]
    daily_11 = bundle["daily_11"]
    day_to_week = bundle["day_to_week"]

    tot_11 = bundle["player_tot11"][ordered].sum(axis=0).copy()
    wk_11 = bundle["player_wk11"][ordered].sum(axis=0).copy()
    benched_g = 0

    for d in crowded_days:
        free_bits = 0x3FF
        match_slot = None
        filled = 0
        w = int(day_to_week[d])
        for idx in ordered:
            if not played_lists[idx][d]:
                continue
            if filled == 10:
                v = daily_11[idx][d]
                tot_11 -= v
                wk_11[w] -= v
                benched_g += 1
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
                v = daily_11[idx][d]
                tot_11 -= v
                wk_11[w] -= v
                benched_g += 1

    res = (tot_11, wk_11, benched_g)
    bundle["roster_cache"][key] = res
    return res


def score_cat9_league(team_tot11, team_wk11, valid_wks):
    """Score N teams simultaneously across 9-Cat Roto, H2H Each-Category, and H2H Most-Categories.

    Returns (roto_pts, h2h_each_win_rate, h2h_most_win_rate), each shape (N,).
    """
    n_teams = len(team_tot11)
    t_cat = team_cat9_from_stats11(team_tot11)  # (N, 9)
    roto_pts = np.zeros(n_teams, dtype=np.float64)
    for c in range(9):
        vals = t_cat[:, c]
        for i in range(n_teams):
            less = float(np.sum(vals < vals[i]))
            eq = float(np.sum(vals == vals[i]))
            roto_pts[i] += 1.0 + less + 0.5 * (eq - 1.0)

    t_wk_cat = team_cat9_from_stats11(team_wk11[:, valid_wks, :])  # (N, W, 9)
    a = t_wk_cat[:, None, :, :]
    b = t_wk_cat[None, :, :, :]
    cat_win = (a > b).astype(np.float64) + 0.5 * (a == b).astype(np.float64)
    mask = ~np.eye(n_teams, dtype=bool)

    each_rate = cat_win.mean(axis=(2, 3))
    h2h_each = each_rate[mask].reshape(n_teams, n_teams - 1).mean(axis=1)

    wk_score = cat_win.sum(axis=3)  # in [0, 9]
    wk_win = (wk_score > 4.5).astype(np.float64) + 0.5 * (wk_score == 4.5).astype(np.float64)
    most_rate = wk_win.mean(axis=2)
    h2h_most = most_rate[mask].reshape(n_teams, n_teams - 1).mean(axis=1)

    return roto_pts, h2h_each, h2h_most


def _rank_desc(arr):
    n = len(arr)
    ranks = np.zeros(n, dtype=float)
    wins = np.zeros(n, dtype=float)
    for i in range(n):
        better = int(np.sum(arr > arr[i]))
        tied = int(np.sum(arr == arr[i])) - 1
        ranks[i] = 1.0 + better + 0.5 * tied
        wins[i] = (1.0 / (1 + tied)) if better == 0 else 0.0
    return ranks, wins


def _pick_legal_from_board(board, taken, roster, cur_c, cur_m, rem_after, masks7, is_c, max_centers):
    for idx in board:
        if taken[idx] or (is_c[idx] and cur_c >= max_centers):
            continue
        if cur_m + rem_after >= 7:
            return idx, (de.max_matching_7(roster + [idx], masks7) if cur_m < 7 else 7)
        cm = de.max_matching_7(roster + [idx], masks7)
        if cm + rem_after >= 7:
            return idx, cm
    choice = next(
        (idx for idx in board if not taken[idx] and not (is_c[idx] and cur_c >= max_centers)),
        next(idx for idx in board if not taken[idx]),
    )
    return choice, de.max_matching_7(roster + [choice], masks7)


def run_cat9_backtest_sim(
    sim_bundles,
    *,
    strategies=CAT9_STRATEGIES,
    n_teams=12,
    n_rounds=13,
    trials_per_slot=12,
    adp_var=2.0,
    max_centers=4,
    seed=42,
    opp_use_yahoo_adp=False,
    waiver_fill_roto=False,
    quiet=False,
):
    """Run 12-team x 13-round daily-active-lineup 9-Cat simulator scoring Roto, H2H-Each, and H2H-Most."""
    rng = random.Random(seed)
    seasons = sorted(sim_bundles)
    pick_order = [
        (rnd + 1, team)
        for rnd in range(n_rounds)
        for team in (range(n_teams - 1, -1, -1) if rnd % 2 == 1 else range(n_teams))
    ]
    next_k_by_pick = []
    for p_idx, (_, team) in enumerate(pick_order):
        nxt = next((j for j in range(p_idx + 1, len(pick_order)) if pick_order[j][1] == team), None)
        next_k_by_pick.append((nxt - p_idx - 1) if nxt is not None else None)
    surv_sigma = max(2.5, 0.7 * adp_var)
    surv_margin = max(1.0, 0.4 * adp_var)

    records_split = defaultdict(list)
    records_season = defaultdict(list)
    sample_info = None
    sample_season = seasons[-1]
    sample_slot = rng.randrange(n_teams)
    sample_trial = rng.randrange(trials_per_slot)

    for s in seasons:
        b = sim_bundles[s]
        split = b["split"]
        players = b["players"]
        n_p = len(players)
        opp_adps = [float(p["yahoo_adp"] if opp_use_yahoo_adp else p["adp"]) for p in players]
        straight_order = sorted(range(n_p), key=lambda i: (opp_adps[i], i))
        proj_cat = b["proj_cat_z"]  # (n_p, 9)
        w9 = b["w9"]

        hero_orders, hero_vals = {}, {}
        for strat in strategies:
            if strat == "market":
                hero_orders[strat] = straight_order
                hero_vals[strat] = [float(p["z_mkt"]) for p in players]
            elif strat == "residual_punt_to":
                vals = [float(proj_cat[i, :8].sum()) for i in range(n_p)]
                scored = sorted([(vals[i], -p["market_rank"], i) for i, p in enumerate(players)], key=lambda t: (-t[0], -t[1], t[2]))
                hero_orders[strat] = [i for _, _, i in scored]
                hero_vals[strat] = vals
            elif strat == "residual_punt1_h2h":
                vals = [float(p["residual_unified_proj"]) for p in players]
                scored = sorted([(vals[i], -p["market_rank"], i) for i, p in enumerate(players)], key=lambda t: (-t[0], -t[1], t[2]))
                hero_orders[strat] = [i for _, _, i in scored]
                hero_vals[strat] = vals
            else:
                vals = [float(p[f"{strat}_proj"]) for p in players]
                scored = sorted([(vals[i], -p["market_rank"], i) for i, p in enumerate(players)], key=lambda t: (-t[0], -t[1], t[2]))
                hero_orders[strat] = [i for _, _, i in scored]
                hero_vals[strat] = vals

        masks7, is_c = b["masks7"], b["is_center"]
        valid_wks = b["valid_wks"]
        waiv_pg11 = b["waiver_pg_11"]

        for slot in range(n_teams):
            for trial in range(trials_per_slot):
                opp_noisy = []
                for _ in range(n_teams):
                    noisy = [(opp_adps[i] + rng.uniform(-adp_var, adp_var), i) for i in range(n_p)]
                    noisy.sort(key=lambda t: (t[0], t[1]))
                    opp_noisy.append([i for _, i in noisy])

                for strat in strategies:
                    taken = [False] * n_p
                    rosters = [[] for _ in range(n_teams)]
                    c_cnt = [0] * n_teams
                    m_sz = [0] * n_teams
                    hero_picks = []
                    punt_cat = None
                    punt_board = None

                    for p_idx, (rnd, team) in enumerate(pick_order):
                        pick_num = p_idx + 1
                        roster = rosters[team]
                        rem_after = n_rounds - len(roster) - 1
                        cur_c, cur_m = c_cnt[team], m_sz[team]

                        if team == slot and strat == "residual_wait":
                            choice, new_m = de.choose_reach_or_wait_pick(
                                hero_orders[strat], hero_vals[strat], opp_adps, taken,
                                roster, cur_c, cur_m, rem_after, masks7, is_c, max_centers,
                                next_k_by_pick[p_idx], surv_sigma, rnd, surv_margin=surv_margin,
                                min_wait_gain=0.02,
                            )
                        elif team == slot and strat == "residual_punt1_h2h":
                            if rnd == 3 and punt_cat is None:
                                cur_vec = proj_cat[roster].sum(axis=0)
                                punt_cat = int(np.argmin(cur_vec))
                                keep = [c for c in range(9) if c != punt_cat]
                                w_p = w9.copy()
                                w_p[punt_cat] = 0.0
                                vals_p = [float(proj_cat[i] @ w_p) for i in range(n_p)]
                                scored_p = sorted([(vals_p[i], -players[i]["market_rank"], i) for i in range(n_p)], key=lambda t: (-t[0], -t[1], t[2]))
                                punt_board = [i for _, _, i in scored_p]
                            board = punt_board if punt_board is not None else hero_orders[strat]
                            choice, new_m = _pick_legal_from_board(board, taken, roster, cur_c, cur_m, rem_after, masks7, is_c, max_centers)
                        else:
                            board = hero_orders[strat] if team == slot else (straight_order if rnd == 1 else opp_noisy[team])
                            choice, new_m = _pick_legal_from_board(board, taken, roster, cur_c, cur_m, rem_after, masks7, is_c, max_centers)

                        taken[choice] = True
                        roster.append(choice)
                        if is_c[choice]:
                            c_cnt[team] += 1
                        m_sz[team] = new_m
                        if team == slot:
                            hero_picks.append((rnd, pick_num, choice))

                    team_tot11 = np.zeros((n_teams, 11), dtype=np.float64)
                    team_wk11 = np.zeros((n_teams, b["n_wks"], 11), dtype=np.float64)
                    for t in range(n_teams):
                        t11, w11, _bg = score_roster_cat9_active(rosters[t], b)
                        if waiver_fill_roto:
                            # Redraft waiver streaming check: fill missed games up to 65g/player with waiver replacement
                            miss_g = sum(max(0.0, REF_GAMES_WAIVER - float(players[i]["next_games"])) for i in rosters[t])
                            t11 = t11 + 0.75 * miss_g * waiv_pg11
                        team_tot11[t] = t11
                        team_wk11[t] = w11

                    roto, h2he, h2hm = score_cat9_league(team_tot11, team_wk11, valid_wks)
                    r_rk, r_win = _rank_desc(roto)
                    e_rk, e_win = _rank_desc(h2he)
                    m_rk, m_win = _rank_desc(h2hm)

                    field_roto = float(np.mean([roto[t] for t in range(n_teams) if t != slot]))
                    rec = {
                        "roto_pts": float(roto[slot]),
                        "roto_diff": float(roto[slot] - field_roto),
                        "roto_rk": float(r_rk[slot]),
                        "roto_win": float(r_win[slot]),
                        "roto_top4": 1.0 if r_rk[slot] <= 4.0 else 0.0,
                        "h2he_pct": 100.0 * float(h2he[slot]),
                        "h2he_rk": float(e_rk[slot]),
                        "h2he_win": float(e_win[slot]),
                        "h2he_top4": 1.0 if e_rk[slot] <= 4.0 else 0.0,
                        "h2hm_pct": 100.0 * float(h2hm[slot]),
                        "h2hm_rk": float(m_rk[slot]),
                        "h2hm_win": float(m_win[slot]),
                        "h2hm_top4": 1.0 if m_rk[slot] <= 4.0 else 0.0,
                        "hero_rk_cnt": sum(1 for i in rosters[slot] if players[i]["player_type"] == "rookie"),
                        "field_rk_cnt": sum(1 for t in range(n_teams) if t != slot for i in rosters[t] if players[i]["player_type"] == "rookie") / (n_teams - 1),
                        "hero_0g": sum(1 for i in rosters[slot] if players[i]["next_games"] == 0),
                        "field_0g": sum(1 for t in range(n_teams) if t != slot for i in rosters[t] if players[i]["next_games"] == 0) / (n_teams - 1),
                        "hero_c": c_cnt[slot],
                    }
                    records_split[(split, strat)].append(rec)
                    records_split[("all", strat)].append(rec)
                    records_season[(s, split, strat)].append(rec)
                    if s == sample_season and slot == sample_slot and trial == sample_trial and strat == "residual_unified":
                        sample_info = (s, slot + 1, b, hero_orders[strat], hero_picks, rosters[slot], rec)

    def _agg(recs, meta):
        return {
            **meta,
            "drafts": len(recs),
            "roto_pts": round(mean(r["roto_pts"] for r in recs), 2),
            "roto_diff": round(mean(r["roto_diff"] for r in recs), 2),
            "roto_rk": round(mean(r["roto_rk"] for r in recs), 2),
            "roto_win_pct": round(100.0 * mean(r["roto_win"] for r in recs), 1),
            "roto_top4_pct": round(100.0 * mean(r["roto_top4"] for r in recs), 1),
            "h2he_pct": round(mean(r["h2he_pct"] for r in recs), 2),
            "h2he_rk": round(mean(r["h2he_rk"] for r in recs), 2),
            "h2he_win_pct": round(100.0 * mean(r["h2he_win"] for r in recs), 1),
            "h2he_top4_pct": round(100.0 * mean(r["h2he_top4"] for r in recs), 1),
            "h2hm_pct": round(mean(r["h2hm_pct"] for r in recs), 2),
            "h2hm_rk": round(mean(r["h2hm_rk"] for r in recs), 2),
            "h2hm_win_pct": round(100.0 * mean(r["h2hm_win"] for r in recs), 1),
            "h2hm_top4_pct": round(100.0 * mean(r["h2hm_top4"] for r in recs), 1),
            "hero_rk_cnt": round(mean(r["hero_rk_cnt"] for r in recs), 2),
            "field_rk_cnt": round(mean(r["field_rk_cnt"] for r in recs), 2),
            "hero_0g": round(mean(r["hero_0g"] for r in recs), 2),
            "field_0g": round(mean(r["field_0g"] for r in recs), 2),
        }

    sim_summary, sim_seasons = [], []
    for split in ("cv", "holdout", "all"):
        for strat in strategies:
            if (split, strat) in records_split:
                sim_summary.append(_agg(records_split[(split, strat)], {"split": split, "strategy": strat}))
    for s in seasons:
        sp = sim_bundles[s]["split"]
        for strat in strategies:
            sim_seasons.append(_agg(records_season[(s, sp, strat)], {"season": s, "split": sp, "strategy": strat}))

    if not quiet:
        opp_tag = "Yahoo ADP" if opp_use_yahoo_adp else "FantasyPros AVG"
        waiv_tag = " + 65g Waiver Fill" if waiver_fill_roto else ""
        print(f"\n================ 12-TEAM 9-CAT MULTI-FORMAT SIMULATOR (Opps: {opp_tag}, var=+-{adp_var:g}{waiv_tag}) ================")
        print(f"Rules: {n_teams}T x {n_rounds}R | PG,SG,SF,PF,C,G,F,3xUTIL + 3 BE | Max {max_centers} Primary C | {trials_per_slot * n_teams} drafts/season | Seed: {seed}")
        for split_label, title in (("cv", "CV SEASONS"), ("holdout", "HOLDOUT SEASONS"), ("all", "ALL SEASONS COMBINED")):
            rows = [r for r in sim_summary if r["split"] == split_label]
            if not rows:
                continue
            print(f"\n=== 9-Cat Simulation — {title} ({rows[0]['drafts']} drafts/strategy) ===")
            print(
                f"  {'Strategy':18s} | {'RotoPts':>7s} {'Rk':>4s} {'Win%':>5s} {'T4%':>5s} | "
                f"{'H2H-E%':>6s} {'Rk':>4s} {'Win%':>5s} | {'H2H-M%':>6s} {'Rk':>4s} {'Win%':>5s} | {'0g(H/F)':>9s}"
            )
            print("  " + "-" * 102)
            for r in rows:
                print(
                    f"  {r['strategy']:18s} | {r['roto_pts']:7.2f} {r['roto_rk']:4.2f} {r['roto_win_pct']:4.1f}% {r['roto_top4_pct']:4.1f}% | "
                    f"{r['h2he_pct']:5.2f}% {r['h2he_rk']:4.2f} {r['h2he_win_pct']:4.1f}% | "
                    f"{r['h2hm_pct']:5.2f}% {r['h2hm_rk']:4.2f} {r['h2hm_win_pct']:4.1f}% | {r['hero_0g']:4.2f}/{r['field_0g']:4.2f}"
                )

        if "residual_unified" in strategies and "market" in strategies:
            print("\n=== Season-by-Season 9-Cat Performance (Market vs. Residual Greedy — Pure Board) ===")
            print(f"  {'Season':7s} {'Split':7s} | {'Mkt Roto':>8s} | {'Grdy Roto':>9s} {'Rk':>4s} {'Win%':>5s} | {'H2H-E Rk':>8s} {'Win%':>5s} | {'H2H-M Rk':>8s} {'Win%':>5s} | {'0g(H/F)':>9s}")
            print("  " + "-" * 103)
            by_sm = {(r["season"], r["strategy"]): r for r in sim_seasons}
            for s in seasons:
                m = by_sm[(s, "market")]
                ru = by_sm[(s, "residual_unified")]
                print(
                    f"  {s:7s} {ru['split']:7s} | {m['roto_pts']:8.2f} | "
                    f"{ru['roto_pts']:9.2f} {ru['roto_rk']:4.2f} {ru['roto_win_pct']:4.1f}% | "
                    f"{ru['h2he_rk']:8.2f} {ru['h2he_win_pct']:4.1f}% | "
                    f"{ru['h2hm_rk']:8.2f} {ru['h2hm_win_pct']:4.1f}% | {ru['hero_0g']:4.2f}/{ru['field_0g']:4.2f}"
                )

        if sample_info:
            s, slot_num, b, hero_order, picks, roster, rec = sample_info
            players = b["players"]
            rank_of = {idx: r for r, idx in enumerate(hero_order, start=1)}
            slot_of = de.assign_lineup_slots(roster, b["masks10"], b["prio"])
            tag_map = {"returning_vet": "VET", "rookie": "ROOK", "returning_absence": "RET"}
            print(f"\n=== Sample {s} 9-Cat Roster — Hero (residual_unified, Pure Greedy) at Random Slot #{slot_num}/{n_teams} ===")
            print(
                f"  Roto: {rec['roto_pts']:.1f}/108 (#{rec['roto_rk']:.1f}) | "
                f"H2H-Each: {rec['h2he_pct']:.1f}% (#{rec['h2he_rk']:.1f}) | "
                f"H2H-Most: {rec['h2hm_pct']:.1f}% (#{rec['h2hm_rk']:.1f}) | "
                f"Primary C: {rec['hero_c']}/{max_centers} | 0g Busts: {rec['hero_0g']}"
            )
            print(f"  {'Rd':>2s} {'Pk':>3s} {'Slot':>4s} | {'Player':22s} {'Type':>4s} {'Pos':>8s} | {'ADP':>5s} {'HeroRk':>6s} | {'Gm':>3s} {'ActZ':>6s}")
            print("  " + "-" * 78)
            for rnd, pick_num, idx in picks:
                p = players[idx]
                print(
                    f"  {rnd:2d} {pick_num:3d} {slot_of[idx]:>4s} | {p['player'][:22]:22s} {tag_map[p['player_type']]:>4s} {p['pos_str'][:8]:>8s} | "
                    f"{p['adp']:5.1f} {rank_of[idx]:6d} | {p['next_games']:3d} {p['z_actual']:+6.2f}"
                )

    return sim_summary, sim_seasons


def _build_sim_bundles(
    scored_seasons,
    development,
    holdout,
    cat9_pools,
    cat9_by_year,
    w9,
    eff_min_s,
    lam=DEFAULT_LAM,
    wait_lam=DEFAULT_WAIT_LAM,
    recency_halflife=None,
    drop_platform_feats=False,
):
    board_rows_by_split = defaultdict(list)
    sim_bundles = {}
    final_meta = {}

    for target_s in scored_seasons:
        target_yr = int(target_s[:4])
        split = "holdout" if target_s in holdout else "cv"
        cutoff = development[-1] if split == "holdout" else target_s

        tr_rows = [
            r for s, grp in cat9_pools.items()
            if (eff_min_s is None or s >= eff_min_s) and (s <= cutoff if split == "holdout" else s < cutoff)
            for r in grp
        ]
        cands = [r.copy() for r in cat9_pools[target_s]]
        res = fit_predict_cat9_unified(
            tr_rows, cands, w9, lam=lam, recency_halflife=recency_halflife, drop_platform_feats=drop_platform_feats,
        )
        clip_val = res["clip_val"]
        raw_res = res["raw_residuals"]
        resid_preds = res["residual_proj"]

        for i, row in enumerate(cands):
            row["split"] = split
            row["residual_unified_proj"] = round(float(resid_preds[i]), 4)
            row["residual_wait_proj"] = float(row["z_mkt"]) + float(wait_lam) * max(-clip_val, min(clip_val, float(raw_res[i])))
            row["pred_edge_raw"] = float(raw_res[i])

        board_rows_by_split[split].extend(cands)
        if split == "holdout":
            final_meta = {
                "n_train_rows": len(tr_rows),
                "vet_coefs": res["vet_coefs"],
                "rk_bias": res["rookie_flat_bias"],
            }

        c9_s = cat9_by_year[target_yr]
        n_p = len(cands)
        n_days, n_wks = c9_s["n_days"], c9_s["n_wks"]
        day_to_week = c9_s["day_to_week"]
        played_mat = np.zeros((n_p, max(n_days, 1)), dtype=np.uint8)
        player_tot11 = np.zeros((n_p, 11), dtype=np.float64)
        player_wk11 = np.zeros((n_p, n_wks, 11), dtype=np.float64)
        daily_11 = []
        masks7, masks10, is_center, prio = [], [], [], []

        for idx, p in enumerate(cands):
            ptup = p["positions"]
            p["pos_str"] = ",".join(ptup)
            m7, m10 = de._pos_to_masks(ptup)
            masks7.append(m7)
            masks10.append(m10)
            is_center.append(ptup[0] == "C")
            # Active lineup priority: start higher-value players on crowded days
            prio.append(float(p["z_actual_raw"]) if p["z_actual_raw"] is not None else -99.0)
            dmap = c9_s["daily_11"].get(p["athlete_id"], {})
            daily_11.append(dmap)
            for d_idx, v11 in dmap.items():
                played_mat[idx, d_idx] = 1
                player_tot11[idx] += v11
                player_wk11[idx, int(day_to_week[d_idx])] += v11

        if len(cands) >= 156:
            sim_bundles[target_s] = {
                "split": split,
                "players": cands,
                "played_mat": played_mat,
                "played_lists": played_mat.tolist(),
                "daily_11": daily_11,
                "player_tot11": player_tot11,
                "player_wk11": player_wk11,
                "day_to_week": day_to_week,
                "valid_wks": c9_s["valid_wks"],
                "n_wks": n_wks,
                "masks7": masks7,
                "masks10": masks10,
                "is_center": is_center,
                "prio": prio,
                "proj_cat_z": res["proj_cat_z"],
                "w9": w9,
                "waiver_pg_11": c9_s["waiver_pg_11"],
                "roster_cache": {},
            }

    return board_rows_by_split, sim_bundles, final_meta


def run_cat9_backtest(
    data,
    *,
    to_weight=DEFAULT_TO_WEIGHT,
    lam=DEFAULT_LAM,
    wait_lam=DEFAULT_WAIT_LAM,
    holdout_seasons=2,
    cv_folds=6,
    trials_per_slot=12,
    adp_var=2.0,
    seed=42,
    min_train_season="2016-17",
    recency_halflife=None,
    skip_sim=False,
    stress_test=False,
    report_suffix="",
):
    w9 = make_cat_weights(to_weight=to_weight)
    seasons_dirs, season_data, births_cache, profiles_cache, cat9_by_year = load_all_cat9_seasons(data)
    cat9_pools, _, _, _ = build_cat9_adp_pools(
        data, seasons_dirs, season_data, births_cache, profiles_cache, cat9_by_year, w9,
    )

    eff_min_s = None if (not min_train_season or str(min_train_season).lower() == "all") else str(min_train_season)
    train_seasons = sorted(s for s in cat9_pools if eff_min_s is None or s >= eff_min_s)
    eval_seasons = [s for s in train_seasons if len(cat9_pools[s]) >= 156]
    if len(eval_seasons) <= holdout_seasons:
        eval_seasons = train_seasons
    development, holdout = eval_seasons[:-holdout_seasons], eval_seasons[-holdout_seasons:]
    cv_eligible = [s for s in development if sum(1 for ts in train_seasons if ts < s) >= 2]
    cv_seasons = cv_eligible[-cv_folds:] if cv_eligible else development[1:]
    scored_seasons = sorted(set(cv_seasons) | set(holdout))

    board_rows_by_split, sim_bundles, final_meta = _build_sim_bundles(
        scored_seasons, development, holdout, cat9_pools, cat9_by_year, w9, eff_min_s,
        lam=lam, wait_lam=wait_lam, recency_halflife=recency_halflife,
    )

    print(f"================ UNIFIED 9-CAT DRAFT ENGINE BACKTEST (TO weight={to_weight:.2f}, lam={lam:.2f}, wait_lam={wait_lam:.2f}) ================")
    print(f"Development drafts: {', '.join(development)} | Holdout drafts: {', '.join(holdout)}")
    if final_meta:
        cap_w = final_meta["vet_coefs"].get("z_prior_minus_mkt_capped")
        col_w = final_meta["vet_coefs"].get("z_collapse_excess")
        hinge = f" | Collapse hinge capped={cap_w:+.2f}, excess={col_w:+.2f}" if (cap_w is not None and col_w is not None) else ""
        print(f"Train rows: {final_meta['n_train_rows']} | Rookie Z-bias: {final_meta['rk_bias']:+.2f}{hinge}")

    board_summary = []
    for split_label in ("cv", "holdout"):
        rows = board_rows_by_split.get(split_label, [])
        if not rows:
            continue
        by_s = defaultdict(list)
        for r in rows:
            by_s[r["draft_season"]].append(r)
        s_list = sorted(by_s)
        print(f"\n=== GLOBAL Top-150 9-Cat Board — {split_label.upper()} ({len(s_list)} seasons, TO weight={to_weight:.2f}) ===")
        for strat in ("market", "residual_unified"):
            items, added, dropped, corrs = [], [], [], []
            for s in s_list:
                grp = by_s[s]
                ordered = (
                    sorted(grp, key=lambda r: r["market_rank"]) if strat == "market"
                    else sorted(grp, key=lambda r: (-r[f"{strat}_proj"], r["market_rank"]))
                )
                top150, out = ordered[:BOARD_SIZE], ordered[BOARD_SIZE:]
                for rk, it in enumerate(top150, start=1):
                    items.append((rk, it))
                    if it["market_rank"] > BOARD_SIZE:
                        added.append(it)
                for it in out:
                    if it["market_rank"] <= BOARD_SIZE:
                        dropped.append(it)
                t150_m = [r for r in grp if r["market_rank"] <= BOARD_SIZE]
                if strat != "market":
                    corrs.append(float(np.corrcoef([r["pred_edge_raw"] for r in t150_m], [r["z_above_adp"] for r in t150_m])[0, 1]))
            cap = sum(1 for _, it in items if it["z_next_rank"] <= BOARD_SIZE)
            avg_z = mean(it["z_actual"] for _, it in items)
            rk_mae = mean(abs(rk - it["z_next_rank"]) for rk, it in items)
            z0 = sum(1 for _, it in items if it["next_games"] == 0)
            proj_mae = (
                mean(abs(it[f"{strat}_proj"] - it["z_actual"]) for _, it in items) if strat != "market"
                else mean(abs(it["z_mkt"] - it["z_actual"]) for _, it in items)
            )
            r_str = f"r={mean(corrs):+.3f}" if corrs else "r=  n/a"
            swap_s = (
                f"Swaps({len(added):2d}): {mean(x['z_actual'] for x in added) - mean(x['z_actual'] for x in dropped):+.2f} Z"
                if added and dropped else "Swaps( 0): baseline"
            )
            print(
                f"  {strat:18s} | Top150: {cap:3d}/{len(items)} ({100*cap/len(items):5.2f}%) | "
                f"AvgZ: {avg_z:+5.2f} | ProjMAE: {proj_mae:4.2f} | RankMAE: {rk_mae:5.2f} | {r_str} | 0g:{z0:2d} | {swap_s}"
            )
            board_summary.append({
                "split": split_label, "strategy": strat, "seasons": len(s_list),
                "top150_captured": cap, "board_slots": len(items), "avg_z": round(avg_z, 3),
                "proj_mae_z": round(proj_mae, 3), "rank_mae": round(rk_mae, 2),
                "corr_r": round(mean(corrs), 4) if corrs else None, "zero_game_busts": z0,
            })

    tag = f"to{int(round(to_weight * 100))}_{report_suffix}" if report_suffix else f"to{int(round(to_weight * 100))}"
    write_csv(data / f"cat9_engine_board_summary_{tag}.csv", board_summary)

    if not skip_sim and sim_bundles:
        sim_summary, sim_seasons = run_cat9_backtest_sim(
            sim_bundles, trials_per_slot=trials_per_slot, adp_var=adp_var, seed=seed,
        )
        write_csv(data / f"cat9_engine_sim_summary_{tag}.csv", sim_summary)
        write_csv(data / f"cat9_engine_sim_by_season_{tag}.csv", sim_seasons)

    if stress_test and sim_bundles:
        print("\n================ 9-CAT SKEPTIC STRESS TESTS (All OOS Seasons, 120 Drafts/Season) ================")
        _, noplat_bundles, _ = _build_sim_bundles(
            scored_seasons, development, holdout, cat9_pools, cat9_by_year, w9, eff_min_s,
            lam=lam, wait_lam=wait_lam, recency_halflife=recency_halflife, drop_platform_feats=True,
        )
        st_strats = ("market", "residual_unified")
        scenarios = [
            ("1. Baseline (Opps=FP AVG, var=+-2)", sim_bundles, dict(adp_var=2.0, opp_use_yahoo_adp=False, waiver_fill_roto=False)),
            ("2. Opps=Yahoo's OWN ADP (hardest 9-Cat room)", sim_bundles, dict(adp_var=2.0, opp_use_yahoo_adp=True, waiver_fill_roto=False)),
            ("3. Realistic Human Noise (adp_var=+-6)", sim_bundles, dict(adp_var=6.0, opp_use_yahoo_adp=False, waiver_fill_roto=False)),
            ("4. Redraft Waiver Fill on Missed Games (65g)", sim_bundles, dict(adp_var=2.0, opp_use_yahoo_adp=False, waiver_fill_roto=True)),
            ("5. Drop Platform Feats (no espn_vs_avg/std)", noplat_bundles, dict(adp_var=2.0, opp_use_yahoo_adp=False, waiver_fill_roto=False)),
            ("6. Yahoo ADP + Human +-6 + No Platform Feats", noplat_bundles, dict(adp_var=6.0, opp_use_yahoo_adp=True, waiver_fill_roto=False)),
        ]
        print(f"  {'Scenario':48s} | {'Grdy Roto Rk (Win%)':>19s} | {'Grdy H2H-E Rk (Win%)':>20s} | {'Grdy H2H-M Rk (Win%)':>20s}")
        print("  " + "-" * 116)
        for label, b_use, kw in scenarios:
            s_sum, _ = run_cat9_backtest_sim(b_use, strategies=st_strats, trials_per_slot=10, seed=seed, quiet=True, **kw)
            ru = next(r for r in s_sum if r["split"] == "all" and r["strategy"] == "residual_unified")
            print(
                f"  {label:48s} | {ru['roto_rk']:5.2f} ({ru['roto_win_pct']:5.1f}%)      | "
                f"{ru['h2he_rk']:5.2f} ({ru['h2he_win_pct']:5.1f}%)       | {ru['h2hm_rk']:5.2f} ({ru['h2hm_win_pct']:5.1f}%)"
            )

    print(f"\nSaved 9-Cat backtest reports under {data}/cat9_engine_*_{tag}.csv")


# ==============================================================================
# Live 9-Cat Draft Board Generator
# ==============================================================================

def generate_cat9_live_board(
    data,
    target_season,
    *,
    to_weight=DEFAULT_TO_WEIGHT,
    punt_str="",
    lam=DEFAULT_LAM,
    use_live_injuries=False,
    injuries_csv=None,
    opening_night_str=None,
    as_of_date_str=None,
    exclude_players="",
    min_train_season="2016-17",
    recency_halflife=None,
    out_path=None,
):
    if not re.fullmatch(r"\d{4}-\d{2}", target_season):
        raise ValueError(f"--generate-board expects YYYY-YY format (got {target_season!r})")
    target_yr = int(target_season[:4])
    prior_yr = target_yr - 1
    punt_idx = parse_punt_cats(punt_str)
    w9_train = make_cat_weights(to_weight=to_weight)
    w9_punt = make_cat_weights(to_weight=to_weight, punt_indices=punt_idx)

    seasons_dirs, season_data, births_cache, profiles_cache, cat9_by_year = load_all_cat9_seasons(data)
    if prior_yr not in cat9_by_year:
        raise ValueError(f"missing completed prior season {de.season_label(prior_yr)} in {data}")

    cat9_pools, lb_cache, played_before, history_available = build_cat9_adp_pools(
        data, seasons_dirs, season_data, births_cache, profiles_cache, cat9_by_year, w9_train,
    )
    eff_min_s = None if (not min_train_season or str(min_train_season).lower() == "all") else str(min_train_season)
    tr_rows = [
        r for s, grp in cat9_pools.items()
        if s < target_season and (eff_min_s is None or s >= eff_min_s)
        for r in grp
    ]
    if not tr_rows:
        raise ValueError(f"no completed 9-Cat training seasons before {target_season}")

    adp_path = data / target_season / "adp.csv"
    if not adp_path.is_file():
        raise ValueError(f"missing {adp_path} for live 9-Cat board")

    # Build target candidate pool using draft_engine's pool matcher + attach 9-Cat prior fields
    vegas_odds = de.load_vegas_preseason_odds(data)
    base_cands = de.build_eval_adp_pool(
        data, prior_yr, seasons_dirs, season_data, births_cache, profiles_cache,
        set(played_before), history_available, max_pool_adp=225.0, vegas_odds=vegas_odds,
    )

    pri_c9 = cat9_by_year[prior_yr]
    pri_tot_map, pri_raw_map, pri_lb, pri_raw_rk, pri_zero, pri_waiv_raw = lb_cache[prior_yr]
    p2_tot_map = lb_cache[prior_yr - 1][0] if (prior_yr - 1) in lb_cache else {}
    spread = max(1.0, z_slot_value(pri_lb, 10.0) - z_slot_value(pri_lb, 150.0))
    col_thresh = COLLAPSE_SPREAD_RATIO * spread

    excluded_keys = {name_key(p) for p in exclude_players.split(",") if p.strip()} if exclude_players else set()
    cands = []
    for r in base_cands:
        if name_key(r["player"]) in excluded_keys:
            continue
        row = r.copy()
        aid = row["athlete_id"]
        adp = float(row["adp"])
        z_mkt = z_slot_value(pri_lb, adp)
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
            is_col = slot_drop >= COLLAPSE_MIN_SLOT_DROP and raw_gap > col_thresh
            row["z_prior_minus_mkt_capped"] = max(-1.25 * col_thresh, min(col_thresh, raw_gap))
            row["z_collapse_excess"] = (raw_gap - col_thresh) if is_col else 0.0
        else:
            row["z_prior_minus_mkt_capped"] = None
            row["z_collapse_excess"] = 0.0
        cands.append(row)

    res = fit_predict_cat9_unified(tr_rows, cands, w9_train, lam=lam, recency_halflife=recency_halflife)
    proj_tot = res["residual_proj"]
    proj_cat_z = res["proj_cat_z"]
    proj_avail = res["proj_avail"]

    injury_records = []
    inj_notes = []
    if use_live_injuries:
        cache_file = data / target_season / "espn_injuries_cache.json"
        live_recs, src_tag = de.fetch_espn_live_injuries(cache_path=cache_file)
        injury_records.extend(live_recs)
        inj_notes.append(f"{src_tag} ({len(live_recs)} active)")
    if injuries_csv:
        csv_recs = de.load_injuries_csv(Path(injuries_csv))
        injury_records.extend(csv_recs)
        inj_notes.append(f"{Path(injuries_csv).name} ({len(csv_recs)} rows)")

    opening_night = de._parse_iso_date(opening_night_str) if opening_night_str else date(target_yr, 10, 21)
    as_of_date = de._parse_iso_date(as_of_date_str) if as_of_date_str else None
    inj_labels = [""] * len(cands)
    inj_capped = 0
    if injury_records:
        proj_tot, proj_cat_z, inj_labels, inj_capped = apply_cat9_injury_layer(
            cands, proj_tot, proj_cat_z, proj_avail, injury_records, target_yr, w9_train,
            opening_night=opening_night, as_of_date=as_of_date,
        )

    # Apply optional punt filter to ranking score while preserving full 9-cat columns
    rank_scores = np.array([float(proj_cat_z[i] @ w9_punt) for i in range(len(cands))], dtype=float) if punt_idx else proj_tot
    peer_center = float(np.median(rank_scores[:BOARD_SIZE] - np.array([c["z_mkt"] for c in cands[:BOARD_SIZE]])))

    order = sorted(range(len(cands)), key=lambda i: (-rank_scores[i], cands[i]["market_rank"]))
    board_rows = []
    for rk, idx in enumerate(order, start=1):
        r = cands[idx]
        cat_z = proj_cat_z[idx]
        edge_raw = float(rank_scores[idx] - r["z_mkt"])
        row_out = {
            "rank": rk,
            "player": r["player"],
            "type": r["player_type"],
            "positions": ",".join(r["positions"]),
            "adp": round(float(r["adp"]), 1),
            "slots_ahead_of_adp": round(float(r["adp"]) - rk, 1),
            "proj_z9": round(float(rank_scores[idx]), 3),
            "mkt_z9": round(float(r["z_mkt"]), 3),
            "edge_vs_peers": round(edge_raw - peer_center, 3),
            "injury_status": inj_labels[idx],
            "in_top_150": rk <= BOARD_SIZE,
            "draft_pick": r.get("draft_pick"),
        }
        for c_i, cname in enumerate(CAT_NAMES):
            row_out[f"z_{cname.replace('%', 'pct')}"] = round(float(cat_z[c_i]), 2)
        board_rows.append(row_out)

    punt_tag = ("_punt_" + "_".join(CAT_NAMES[i].replace("%", "") for i in punt_idx)) if punt_idx else ""
    dest = Path(out_path) if out_path else (data / f"cat9_draft_board_{target_season}{punt_tag}.csv")
    write_csv(dest, board_rows)

    punt_disp = ", ".join(CAT_NAMES[i] for i in punt_idx) if punt_idx else "None (Balanced)"
    print(f"================ {target_season} 9-CAT DRAFT BOARD (TO weight={to_weight:.2f}, Punt: {punt_disp}) ================")
    if inj_notes:
        print(f"Injuries: {' + '.join(inj_notes)} | Injury-capped: {inj_capped}")
    print(f"\n=== Top 25 9-Cat Draft Board ({target_season}) ===")
    print(f"  {'Rk':>3s} {'ADP':>5s} {'Diff':>6s} | {'Player':22s} {'Pos':>8s} | {'ProjZ':>6s} {'Edge':>6s} | {'PTS':>5s} {'REB':>5s} {'AST':>5s} {'STL':>5s} {'BLK':>5s} {'3PM':>5s} {'FG%':>5s} {'FT%':>5s} {'TO':>5s} | Note")
    print("  " + "-" * 130)
    for r in board_rows[:25]:
        print(
            f"  {r['rank']:3d} {r['adp']:5.1f} {r['slots_ahead_of_adp']:+6.1f} | {r['player'][:22]:22s} {r['positions'][:8]:>8s} | "
            f"{r['proj_z9']:+6.2f} {r['edge_vs_peers']:+6.2f} | "
            f"{r['z_PTS']:+5.2f} {r['z_REB']:+5.2f} {r['z_AST']:+5.2f} {r['z_STL']:+5.2f} {r['z_BLK']:+5.2f} "
            f"{r['z_3PM']:+5.2f} {r['z_FGpct']:+5.2f} {r['z_FTpct']:+5.2f} {r['z_TO']:+5.2f} | {r['injury_status']}"
        )

    targets = sorted([r for r in board_rows if r["adp"] <= 180 and r["edge_vs_peers"] > 0], key=lambda r: -r["edge_vs_peers"])[:10]
    fades = sorted([r for r in board_rows if r["adp"] <= BOARD_SIZE and r["edge_vs_peers"] < 0], key=lambda r: r["edge_vs_peers"])[:10]
    print("\n=== Top 10 9-Cat Targets vs. ADP ===")
    for r in targets:
        print(f"  Rk #{r['rank']:3d} (ADP {r['adp']:5.1f}, {r['slots_ahead_of_adp']:+5.1f}) | {r['player']:22s} | ProjZ {r['proj_z9']:+5.2f} | Edge {r['edge_vs_peers']:+5.2f} {r['injury_status']}")
    print("\n=== Top 10 9-Cat Fades vs. ADP ===")
    for r in fades:
        cut = " [CUT]" if not r["in_top_150"] else ""
        print(f"  Rk #{r['rank']:3d} (ADP {r['adp']:5.1f}, {r['slots_ahead_of_adp']:+5.1f}) | {r['player']:22s} | ProjZ {r['proj_z9']:+5.2f} | Edge {r['edge_vs_peers']:+5.2f}{cut} {r['injury_status']}")

    # Quick live out-of-sample sanity check if partial current-season box scores already exist
    live_s = load_cat9_season(data / target_season, require_completed=False)
    if live_s is not None and len(live_s["p_g"]) >= 200:
        # Compare our pre-draft edge vs actual early-season 9-Cat Z!
        _, _, live_lb, _, live_zero, _ = _season_weighted_lb(live_s, w9_train)
        live_z_by_id = {aid: float(vec @ w9_train) for aid, vec in live_s["z_tot"].items()}
        edges_l, actuals_l = [], []
        for idx, r in enumerate(cands):
            if r["market_rank"] <= BOARD_SIZE:
                act = live_z_by_id.get(r["athlete_id"], live_zero)
                act_vs_adp = act - z_slot_value(live_lb, r["adp"])
                edges_l.append(float(res["raw_residuals"][idx]))
                actuals_l.append(act_vs_adp)
        if len(edges_l) >= 50:
            live_r = float(np.corrcoef(edges_l, actuals_l)[0, 1])
            print(f"\n=== Live {target_season} Zero-Leakage Early-Season Check ===")
            print(f"  Top-150 9-Cat r(pred_edge, actual_Z_vs_ADP) = {live_r:+.3f} (n={len(edges_l)} players)")

    print(f"\nSaved 9-Cat board to {dest}")
    return board_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"), help="season data directory (default: data)")
    parser.add_argument("--backtest", action="store_true", help="run chronological 9-Cat backtest + Roto/H2H simulator")
    parser.add_argument("--generate-board", metavar="YYYY-YY", help="generate ranked 9-Cat draft board for a season")
    parser.add_argument("--to-weight", type=float, default=DEFAULT_TO_WEIGHT, help=f"Turnover category weight in [0, 1] (default: {DEFAULT_TO_WEIGHT})")
    parser.add_argument("--punt", default="", help="comma-separated categories to punt on live board (e.g., FT or FT,TO)")
    parser.add_argument("--lam", type=float, default=DEFAULT_LAM, help=f"Stage-2 residual scale lambda (default: {DEFAULT_LAM})")
    parser.add_argument("--wait-lam", type=float, default=DEFAULT_WAIT_LAM, help=f"Reach-vs-Wait residual scale (default: {DEFAULT_WAIT_LAM})")
    parser.add_argument("--min-train-season", default="2016-17", help="earliest training season (default: 2016-17; 2014-15 or 2008-09 supported)")
    parser.add_argument("--recency-halflife", type=float, default=None, help="optional exponential recency half-life in seasons")
    parser.add_argument("--holdout-seasons", type=int, default=2, help="untouched final seasons for holdout (default: 2)")
    parser.add_argument("--cv-folds", type=int, default=6, help="expanding-window CV seasons (default: 6)")
    parser.add_argument("--trials-per-slot", type=int, default=12, help="sim drafts per draft slot per season (default: 12)")
    parser.add_argument("--adp-var", type=float, default=2.0, help="opponent Uniform[-v, +v] ADP noise in R2-13 (default: 2.0)")
    parser.add_argument("--skip-sim", action="store_true", help="skip the 12-team simulator in --backtest")
    parser.add_argument("--stress-test", action="store_true", help="run built-in skeptic stress tests (Yahoo ADP, +-6 noise, waiver fill, no platform feats)")
    parser.add_argument("--report-suffix", default="", help="optional suffix for backtest CSV reports")
    parser.add_argument("--seed", type=int, default=42, help="random seed for simulator (default: 42)")
    parser.add_argument("--use-live-injuries", action="store_true", help="fetch live ESPN NBA injuries for --generate-board")
    parser.add_argument("--injuries-csv", type=Path, default=None, help="optional local injuries CSV")
    parser.add_argument("--opening-night", default=None, help="override Opening Night YYYY-MM-DD")
    parser.add_argument("--as-of-date", default=None, help="override reference date YYYY-MM-DD for injury cap")
    parser.add_argument("--exclude-players", default="", help="comma-separated confirmed out players to exclude")
    parser.add_argument("--out", type=Path, default=None, help="output CSV path for --generate-board")
    args = parser.parse_args()

    if not args.backtest and not args.generate_board:
        args.backtest = True
    if not 0.0 <= args.to_weight <= 1.0:
        parser.error("--to-weight must be in [0, 1]")

    try:
        if args.generate_board:
            generate_cat9_live_board(
                args.data,
                args.generate_board,
                to_weight=args.to_weight,
                punt_str=args.punt,
                lam=args.lam,
                use_live_injuries=args.use_live_injuries,
                injuries_csv=args.injuries_csv,
                opening_night_str=args.opening_night,
                as_of_date_str=args.as_of_date,
                exclude_players=args.exclude_players,
                min_train_season=args.min_train_season,
                recency_halflife=args.recency_halflife,
                out_path=args.out,
            )
        else:
            run_cat9_backtest(
                args.data,
                to_weight=args.to_weight,
                lam=args.lam,
                wait_lam=args.wait_lam,
                holdout_seasons=args.holdout_seasons,
                cv_folds=args.cv_folds,
                trials_per_slot=args.trials_per_slot,
                adp_var=args.adp_var,
                seed=args.seed,
                min_train_season=args.min_train_season,
                recency_halflife=args.recency_halflife,
                skip_sim=args.skip_sim,
                stress_test=args.stress_test,
                report_suffix=args.report_suffix,
            )
    except (OSError, ValueError, KeyError, csv.Error) as err:
        parser.exit(1, f"9-Cat engine failed: {err}\n")


if __name__ == "__main__":
    main()
