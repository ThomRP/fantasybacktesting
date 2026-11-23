"""Compare residual, decomposed (MPG x FPPM), quantile-tier, and GBDT models on a top-150 draft board."""

import argparse
import csv
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LogisticRegression, QuantileRegressor, RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

from adp_names import clean_adp_player_name
from backtest_common import (
    SCORING,
    adp_hit_counts,
    adp_slot_points,
    draft_outcomes,
    fantasy_points,
    load_birth_dates,
    name_key,
    parse_minutes,
    player_box_scores,
    read_adp,
    read_csv,
    read_season,
    regular_season_game,
    write_csv,
)
from backtest_value_benchmark import price_band_metrics


BOARD_SIZE = 150
BANDS = ((1, 25), (26, 50), (51, 75), (76, 100), (101, 125), (126, 175))
TARGETS_PER_BAND = 5
RIDGE_ALPHAS = np.logspace(-1, 3.5, 25)

BASE_FEATURES = (
    "adp",
    "age_at_prior_break",
    "prior_games",
    "prior_fppg",
    "prior_post_break_fppg_change",
    "prior_start_pct",
)

RESIDUAL_FEATURES = (
    "log_adp",
    "mkt_expected_fppg",
    "prior_minus_mkt_fppg",
    "age_at_prior_break",
    "age_under_24",
    "age_over_30",
    "prior_games",
    "prior_mpg",
    "prior_fppm",
    "prior_start_pct",
    "shrunk_post_break_mpg_change",
    "shrunk_post_break_fppm_change",
    "shrunk_last20_mpg_delta",
    "shrunk_last20_start_delta",
    "prior_ts_pct",
    "prior_stocks_per36",
    "prior_ast_per36",
    "prior_reb_per36",
    "prior_tov_per36",
    "prior_shooting_fpts_share",
    "prior_fga_team_share",
    "prior_ast_team_share",
    "prior_reb_team_share",
    "prior_team_pace",
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
    "mkt_expected_fppg",
    "age_at_prior_break",
    "age_under_24",
    "age_over_30",
    "prior_games",
    "prior_mpg",
    "prior_start_pct",
    "shrunk_post_break_mpg_change",
    "shrunk_last20_mpg_delta",
    "shrunk_last20_start_delta",
    "yoy_mpg_delta",
    "two_yr_games_avg",
    "no_prev2_season",
)

FPPM_FEATURES = (
    "log_adp",
    "mkt_expected_fppg",
    "age_at_prior_break",
    "age_under_24",
    "age_over_30",
    "prior_fppm",
    "prior_mpg",
    "shrunk_post_break_fppm_change",
    "prior_ts_pct",
    "prior_stocks_per36",
    "prior_ast_per36",
    "prior_reb_per36",
    "prior_tov_per36",
    "prior_shooting_fpts_share",
    "prior_fga_team_share",
    "prior_ast_team_share",
    "prior_reb_team_share",
    "prior_team_pace",
    "yoy_fppg_delta",
    "two_yr_weighted_fppg",
    "no_prev2_season",
)

MODEL_NAMES = (
    "logistic_base",
    "huber_residual",
    "decomposed_eb",
    "quantile_tier",
    "gbdt_huber",
)


@dataclass
class GameLine:
    game_date: str
    game_id: str
    minutes: float
    fpts: float
    fpts_shooting: float
    fga: float
    fta: float
    pts: float
    reb: float
    ast: float
    stocks: float
    tov: float
    starter: float | None
    team_fga: float | None
    team_ast: float | None
    team_reb: float | None
    team_poss: float | None


def _safe_float(raw):
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


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


def load_team_game_stats(folder, reg_game_ids):
    """Load regular-season team totals keyed by (game_id, team_id)."""
    path = folder / "team_box_scores.csv"
    if not path.is_file():
        return {}
    team_stats = {}
    for row in read_csv(path):
        gid = row.get("game_id")
        tid = row.get("team_id")
        if gid not in reg_game_ids or not tid:
            continue
        fga = _safe_float(row.get("field_goals_attempted"))
        fta = _safe_float(row.get("free_throws_attempted"))
        tov = _safe_float(row.get("turnovers"))
        ast = _safe_float(row.get("assists"))
        reb = _safe_float(row.get("total_rebounds"))
        poss = (fga + 0.44 * fta + tov) if (fga is not None and fta is not None and tov is not None) else None
        team_stats[(gid, str(tid))] = (fga, ast, reb, poss)
    return team_stats


def extract_rich_season_features(folder, all_star_date):
    """Compute rate/volume, recency-shrunk, efficiency, and team-share features for a season."""
    schedule = list(read_csv(folder / "games.csv"))
    game_dates = {g["game_id"]: g["game_date"] for g in schedule if regular_season_game(g)}
    team_stats = load_team_game_stats(folder, set(game_dates))

    by_player = defaultdict(list)
    for gid, aid, _, stats, _, starter, tid in player_box_scores(folder, game_dates):
        mins = parse_minutes(stats.get("minutes"))
        if mins is None or mins <= 0:
            continue
        try:
            fpts = float(fantasy_points(stats))
            fgm = float(stats["field_goals_made"])
            fga = float(stats["field_goals_attempted"])
            ftm = float(stats["free_throws_made"])
            fta = float(stats["free_throws_attempted"])
            tpm = float(stats["three_point_field_goals_made"])
            pts = float(stats["points"])
            reb = float(stats["rebounds"])
            ast = float(stats["assists"])
            stl = float(stats["steals"])
            blk = float(stats["blocks"])
            tov = float(stats["turnovers"])
        except (KeyError, TypeError, ValueError):
            continue

        fpts_shooting = (
            fgm * SCORING["field_goals_made"]
            + fga * SCORING["field_goals_attempted"]
            + ftm * SCORING["free_throws_made"]
            + fta * SCORING["free_throws_attempted"]
            + tpm * SCORING["three_point_field_goals_made"]
            + pts * SCORING["points"]
        )
        start_val = {"true": 1.0, "false": 0.0}.get(str(starter).strip().lower())
        t_fga, t_ast, t_reb, t_poss = team_stats.get((gid, str(tid or "")), (None, None, None, None))
        by_player[aid].append(GameLine(
            game_date=game_dates[gid],
            game_id=gid,
            minutes=mins,
            fpts=fpts,
            fpts_shooting=fpts_shooting,
            fga=fga,
            fta=fta,
            pts=pts,
            reb=reb,
            ast=ast,
            stocks=stl + blk,
            tov=tov,
            starter=start_val,
            team_fga=t_fga,
            team_ast=t_ast,
            team_reb=t_reb,
            team_poss=t_poss,
        ))

    cutoff = str(all_star_date) if all_star_date is not None else None
    out = {}
    for aid, lines in by_player.items():
        lines.sort(key=lambda x: (x.game_date, x.game_id))
        n = len(lines)
        tot_min = sum(x.minutes for x in lines)
        tot_fpts = sum(x.fpts for x in lines)
        if tot_min <= 0:
            continue

        mpg = tot_min / n
        fppg = tot_fpts / n
        fppm = tot_fpts / tot_min

        tot_pts = sum(x.pts for x in lines)
        tot_fga = sum(x.fga for x in lines)
        tot_fta = sum(x.fta for x in lines)
        ts_denom = 2.0 * (tot_fga + 0.44 * tot_fta)
        ts_pct = tot_pts / ts_denom if ts_denom > 0 else None

        tot_shoot_fpts = sum(x.fpts_shooting for x in lines)
        shooting_share = tot_shoot_fpts / tot_fpts if abs(tot_fpts) > 1e-6 else None

        stocks_p36 = 36.0 * sum(x.stocks for x in lines) / tot_min
        ast_p36 = 36.0 * sum(x.ast for x in lines) / tot_min
        reb_p36 = 36.0 * sum(x.reb for x in lines) / tot_min
        tov_p36 = 36.0 * sum(x.tov for x in lines) / tot_min

        # Post-All-Star splits (raw + Empirical Bayes shrunk by n_post / (n_post + 15))
        pre = [x for x in lines if cutoff is None or x.game_date <= cutoff]
        post = [x for x in lines if cutoff is not None and x.game_date > cutoff]
        raw_post_fppg_chg = None
        shrunk_post_mpg_chg = None
        shrunk_post_fppm_chg = None
        if pre and post:
            pre_fppg = sum(x.fpts for x in pre) / len(pre)
            post_fppg = sum(x.fpts for x in post) / len(post)
            raw_post_fppg_chg = post_fppg - pre_fppg

            w_post = len(post) / (len(post) + 15.0)
            pre_mpg = sum(x.minutes for x in pre) / len(pre)
            post_mpg = sum(x.minutes for x in post) / len(post)
            shrunk_post_mpg_chg = (post_mpg - pre_mpg) * w_post

            pre_min = sum(x.minutes for x in pre)
            post_min = sum(x.minutes for x in post)
            if pre_min > 0 and post_min > 0:
                shrunk_post_fppm_chg = (
                    sum(x.fpts for x in post) / post_min - sum(x.fpts for x in pre) / pre_min
                ) * w_post

        # Last-20-games shrunk role delta (captures trade-deadline promotions)
        last20 = lines[-20:]
        w_last = len(last20) / (len(last20) + 15.0)
        shrunk_last20_mpg = (sum(x.minutes for x in last20) / len(last20) - mpg) * w_last if n >= 10 else 0.0

        starts_all = [x.starter for x in lines if x.starter is not None]
        start_pct = 100.0 * sum(starts_all) / len(starts_all) if len(starts_all) == n else None
        starts_last = [x.starter for x in last20 if x.starter is not None]
        if start_pct is not None and len(starts_last) == len(last20) and n >= 10:
            shrunk_last20_start = (100.0 * sum(starts_last) / len(starts_last) - start_pct) * w_last
        else:
            shrunk_last20_start = 0.0 if start_pct is not None else None

        # Team shares & pace
        t_fga_lines = [(x.fga, x.team_fga) for x in lines if x.team_fga and x.team_fga > 0]
        t_ast_lines = [(x.ast, x.team_ast) for x in lines if x.team_ast and x.team_ast > 0]
        t_reb_lines = [(x.reb, x.team_reb) for x in lines if x.team_reb and x.team_reb > 0]
        t_poss_vals = [x.team_poss for x in lines if x.team_poss and x.team_poss > 0]

        out[aid] = {
            "games": n,
            "mpg": mpg,
            "fppg": fppg,
            "fppm": fppm,
            "start_pct": start_pct,
            "raw_post_fppg_chg": raw_post_fppg_chg,
            "shrunk_post_mpg_chg": shrunk_post_mpg_chg,
            "shrunk_post_fppm_chg": shrunk_post_fppm_chg,
            "shrunk_last20_mpg": shrunk_last20_mpg,
            "shrunk_last20_start": shrunk_last20_start,
            "ts_pct": ts_pct,
            "stocks_p36": stocks_p36,
            "ast_p36": ast_p36,
            "reb_p36": reb_p36,
            "tov_p36": tov_p36,
            "shooting_share": shooting_share,
            "fga_team_share": sum(a for a, _ in t_fga_lines) / sum(b for _, b in t_fga_lines) if t_fga_lines else None,
            "ast_team_share": sum(a for a, _ in t_ast_lines) / sum(b for _, b in t_ast_lines) if t_ast_lines else None,
            "reb_team_share": sum(a for a, _ in t_reb_lines) / sum(b for _, b in t_reb_lines) if t_reb_lines else None,
            "team_pace": mean(t_poss_vals) if t_poss_vals else None,
        }
    return out


def build_dataset(data):
    """Build multi-season player-draft rows and full market candidate pools (zero leakage)."""
    seasons = {int(p.name[:4]): p for p in data.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}", p.name)}
    season_stats = {}
    rich_cache = {}

    def get_season(year):
        if year not in season_stats:
            folder = seasons.get(year)
            season_stats[year] = (
                read_season(folder, split_at_all_star=True, include_box_scores=False) if folder else None
            )
        return season_stats[year]

    def get_rich(year):
        if year not in rich_cache:
            res = get_season(year)
            rich_cache[year] = extract_rich_season_features(seasons[year], res[0]) if res else {}
        return rich_cache[year]

    rows = []
    market_pools = {}
    for year in sorted(seasons):
        following = seasons.get(year + 1)
        if following is None or not (following / "adp.csv").is_file():
            continue
        prior_res = get_season(year)
        next_res = get_season(year + 1)
        if prior_res is None or next_res is None:
            continue

        all_star_date, prior_players = prior_res
        _, next_players = next_res
        prior_rich = get_rich(year)
        next_rich = get_rich(year + 1)
        prev2_rich = get_rich(year - 1) if (year - 1) in seasons else None

        prior_leaderboard = sorted(
            (p.total_points / p.total_games for p in prior_players.values() if p.total_games > 0),
            reverse=True,
        )
        platform_stats = read_adp_platform_stats(following / "adp.csv")

        try:
            births = load_birth_dates(data, year + 1)
        except Exception as err:
            print(f"{seasons[year].name}: birth dates unavailable ({err})")
            births = {}

        outcomes_prior, total_picks = draft_outcomes(
            following, prior_players, next_players, match_prior_only=True, rank_per_game=True,
        )
        outcomes_all, _ = draft_outcomes(
            following, prior_players, next_players, match_prior_only=False, rank_per_game=True,
        )
        outcomes_all_by_key = {name_key(n): o for n, _, o in outcomes_all}
        prior_id_by_key = {name_key(n): aid for n, aid, _ in outcomes_prior}

        print(f"{seasons[year].name} -> {following.name}: matched {len(outcomes_prior)}/{total_picks} returning ADP players")

        for name, aid, outcome in outcomes_prior:
            pr = prior_rich.get(aid)
            if pr is None:
                continue
            nr = next_rich.get(aid)
            p2 = prev2_rich.get(aid) if prev2_rich is not None else None

            age = None
            birth = births.get(aid)
            if birth is not None:
                age = all_star_date.year - birth.year
                if (all_star_date.month, all_star_date.day) < (birth.month, birth.day):
                    age -= 1

            adp = outcome["adp"]
            mkt_exp = adp_slot_points(prior_leaderboard, adp)
            pstats = platform_stats.get(name_key(name), {})

            if p2 is not None:
                yoy_fppg = pr["fppg"] - p2["fppg"]
                yoy_mpg = pr["mpg"] - p2["mpg"]
                two_yr_fppg = 0.65 * pr["fppg"] + 0.35 * p2["fppg"]
                two_yr_games = 0.5 * (pr["games"] + p2["games"])
                no_prev2 = 0.0
            else:
                yoy_fppg = None
                yoy_mpg = None
                two_yr_fppg = pr["fppg"]
                two_yr_games = float(pr["games"])
                no_prev2 = 1.0 if prev2_rich is not None else None

            rows.append({
                "prior_season": seasons[year].name,
                "draft_season": following.name,
                "player": name,
                "athlete_id": aid,
                # Base features
                "adp": adp,
                "age_at_prior_break": age,
                "prior_games": pr["games"],
                "prior_fppg": pr["fppg"],
                "prior_post_break_fppg_change": pr["raw_post_fppg_chg"],
                "prior_start_pct": pr["start_pct"],
                # Engineered features
                "log_adp": math.log(max(adp, 1.0)),
                "mkt_expected_fppg": mkt_exp,
                "prior_minus_mkt_fppg": pr["fppg"] - mkt_exp,
                "age_under_24": max(0.0, 24.0 - age) if age is not None else None,
                "age_over_30": max(0.0, age - 30.0) if age is not None else None,
                "prior_mpg": pr["mpg"],
                "prior_fppm": pr["fppm"],
                "shrunk_post_break_mpg_change": pr["shrunk_post_mpg_chg"],
                "shrunk_post_break_fppm_change": pr["shrunk_post_fppm_chg"],
                "shrunk_last20_mpg_delta": pr["shrunk_last20_mpg"],
                "shrunk_last20_start_delta": pr["shrunk_last20_start"],
                "prior_ts_pct": pr["ts_pct"],
                "prior_stocks_per36": pr["stocks_p36"],
                "prior_ast_per36": pr["ast_p36"],
                "prior_reb_per36": pr["reb_p36"],
                "prior_tov_per36": pr["tov_p36"],
                "prior_shooting_fpts_share": pr["shooting_share"],
                "prior_fga_team_share": pr["fga_team_share"],
                "prior_ast_team_share": pr["ast_team_share"],
                "prior_reb_team_share": pr["reb_team_share"],
                "prior_team_pace": pr["team_pace"],
                "yoy_fppg_delta": yoy_fppg,
                "yoy_mpg_delta": yoy_mpg,
                "two_yr_weighted_fppg": two_yr_fppg,
                "two_yr_games_avg": two_yr_games,
                "no_prev2_season": no_prev2,
                "adp_platform_std": pstats.get("adp_platform_std"),
                "espn_vs_avg": pstats.get("espn_vs_avg"),
                # Extra next-season targets for decomposed model
                "next_mpg": nr["mpg"] if nr else None,
                "next_fppm": nr["fppm"] if nr else None,
                **outcome,
            })

        market = {}
        for name, adp in sorted(read_adp(following / "adp.csv"), key=lambda item: (item[1], item[0].casefold())):
            market.setdefault(name_key(name), (name, adp))
        pool = []
        for market_rank, (key, (name, adp)) in enumerate(list(market.items())[:BANDS[-1][1]], start=1):
            out = outcomes_all_by_key.get(key, {})
            pool.append({
                "draft_season": following.name,
                "player": name,
                "athlete_id": prior_id_by_key.get(key),
                "adp": adp,
                "market_rank": market_rank,
                "actual_fppg": out.get("next_fppg"),
                "actual_rank": out.get("next_rank"),
                "beat_adp": out.get("beat_adp"),
                "fppg_above_adp": out.get("fppg_above_adp"),
            })
        if len(pool) >= BOARD_SIZE:
            market_pools[following.name] = pool

    return rows, market_pools


def to_matrix(rows, features):
    return np.array(
        [[np.nan if row.get(f) is None else float(row[f]) for f in features] for row in rows],
        dtype=float,
    )


def active_features(training, features):
    """Drop features that are completely missing or constant in training."""
    return [f for f in features if len({row[f] for row in training if row.get(f) is not None}) > 1]


def fit_and_score_models(training, evaluated):
    """Fit all candidate models strictly on `training` and return score arrays for `evaluated`."""
    scores = {}

    # 1. Repo baseline: quadratic logistic regression on original 6 features
    base_feats = active_features(training, BASE_FEATURES)
    y_cls = np.array([1 if r["beat_adp"] else 0 for r in training], dtype=int)
    logistic_base = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=1000),
    )
    logistic_base.fit(to_matrix(training, base_feats), y_cls)
    scores["logistic_base"] = logistic_base.predict_proba(to_matrix(evaluated, base_feats))[:, 1]

    # Shared engineered matrix & continuous market-residual target
    res_feats = active_features(training, RESIDUAL_FEATURES)
    X_tr = to_matrix(training, res_feats)
    X_ev = to_matrix(evaluated, res_feats)
    y_res = np.array([float(r["fppg_above_adp"]) for r in training], dtype=float)

    # 2. Huber Robust Residual Regression (resistant to low-game injury outliers)
    huber = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        HuberRegressor(alpha=25.0, epsilon=1.35, max_iter=1000),
    )
    huber.fit(X_tr, y_res)
    scores["huber_residual"] = huber.predict(X_ev)

    # 3. Structural Decomposition (MPG x FPPM with splines) + Out-of-Fold Market-Anchor Meta Layer
    decomp_rows = [r for r in training if r.get("next_mpg") is not None and r.get("next_fppm") is not None]
    mpg_feats = active_features(decomp_rows, MPG_FEATURES)
    fppm_feats = active_features(decomp_rows, FPPM_FEATURES)

    def make_spline_ridge(feat_list):
        spline_targets = {"log_adp", "mkt_expected_fppg", "age_at_prior_break"}
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

    X_mpg_tr = to_matrix(decomp_rows, mpg_feats)
    y_mpg_tr = np.array([float(r["next_mpg"]) for r in decomp_rows], dtype=float)
    X_fppm_tr = to_matrix(decomp_rows, fppm_feats)
    y_fppm_tr = np.array([float(r["next_fppm"]) for r in decomp_rows], dtype=float)

    groups = np.array([r["draft_season"] for r in decomp_rows])
    n_seasons = len(set(groups))
    if n_seasons >= 3:
        gkf = GroupKFold(n_splits=min(n_seasons, 5))
        oof_mpg = cross_val_predict(make_spline_ridge(mpg_feats), X_mpg_tr, y_mpg_tr, cv=gkf, groups=groups)
        oof_fppm = cross_val_predict(make_spline_ridge(fppm_feats), X_fppm_tr, y_fppm_tr, cv=gkf, groups=groups)
        tr_struct_fppg = oof_mpg * oof_fppm
    else:
        tmp_mpg = make_spline_ridge(mpg_feats).fit(X_mpg_tr, y_mpg_tr)
        tmp_fppm = make_spline_ridge(fppm_feats).fit(X_fppm_tr, y_fppm_tr)
        tr_struct_fppg = tmp_mpg.predict(X_mpg_tr) * tmp_fppm.predict(X_fppm_tr)

    mpg_model = make_spline_ridge(mpg_feats).fit(X_mpg_tr, y_mpg_tr)
    fppm_model = make_spline_ridge(fppm_feats).fit(X_fppm_tr, y_fppm_tr)
    ev_struct_fppg = mpg_model.predict(to_matrix(evaluated, mpg_feats)) * fppm_model.predict(
        to_matrix(evaluated, fppm_feats)
    )

    tr_mkt = np.array([float(r["mkt_expected_fppg"]) for r in decomp_rows])
    ev_mkt = np.array([float(r["mkt_expected_fppg"]) for r in evaluated])
    tr_log_adp = np.array([float(r["log_adp"]) for r in decomp_rows])
    ev_log_adp = np.array([float(r["log_adp"]) for r in evaluated])
    tr_games = np.array([float(r["prior_games"]) for r in decomp_rows])
    ev_games = np.array([float(r["prior_games"]) for r in evaluated])

    meta_tr = np.column_stack([
        tr_struct_fppg - tr_mkt,
        (tr_struct_fppg - tr_mkt) * tr_log_adp,
        tr_log_adp,
        np.minimum(tr_games, 70.0) / 70.0,
    ])
    meta_ev = np.column_stack([
        ev_struct_fppg - ev_mkt,
        (ev_struct_fppg - ev_mkt) * ev_log_adp,
        ev_log_adp,
        np.minimum(ev_games, 70.0) / 70.0,
    ])
    meta = make_pipeline(StandardScaler(), HuberRegressor(alpha=10.0, max_iter=1000))
    meta.fit(meta_tr, [float(r["fppg_above_adp"]) for r in decomp_rows])
    scores["decomposed_eb"] = meta.predict(meta_ev)

    # 4. Tier-Aware Quantile Regression (floor early, median mid, upside late)
    preproc = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
    )
    X_tr_scaled = preproc.fit_transform(X_tr)
    X_ev_scaled = preproc.transform(X_ev)

    q_preds = {}
    for q in (0.25, 0.50, 0.75):
        qr = QuantileRegressor(quantile=q, alpha=0.12, solver="highs")
        qr.fit(X_tr_scaled, y_res)
        q_preds[q] = qr.predict(X_ev_scaled)

    tier_scores = np.empty(len(evaluated), dtype=float)
    for i, row in enumerate(evaluated):
        adp = row["adp"]
        if adp <= 50:
            tier_scores[i] = 0.55 * q_preds[0.25][i] + 0.45 * q_preds[0.50][i]
        elif adp <= 100:
            tier_scores[i] = q_preds[0.50][i]
        else:
            tier_scores[i] = 0.45 * q_preds[0.50][i] + 0.55 * q_preds[0.75][i]
    scores["quantile_tier"] = tier_scores

    # 5. Heavily Regularized Shallow GBDT with Absolute Error (native NaN handling)
    gbdt = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_depth=2,
        min_samples_leaf=25,
        l2_regularization=5.0,
        max_iter=80,
        learning_rate=0.05,
        early_stopping=False,
        random_state=42,
    )
    gbdt.fit(X_tr, y_res)
    scores["gbdt_huber"] = gbdt.predict(X_ev)

    return scores


def select_top_pct_indices(rows, score_col, top_pct):
    selected = set()
    for season in sorted({r["draft_season"] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r["draft_season"] == season]
        indices.sort(key=lambda i: (-rows[i][score_col], rows[i]["adp"], rows[i]["athlete_id"]))
        selected.update(indices[:math.ceil(len(indices) * top_pct / 100)])
    return selected


def attach_selection_flags(rows, top_pct):
    """Attach {model}_selected flags for compatibility with backtest_adp_bands.py and backtest_rank_band.py."""
    for model in MODEL_NAMES:
        selected = select_top_pct_indices(rows, f"{model}_score", top_pct)
        for i, row in enumerate(rows):
            row[f"{model}_selected"] = i in selected


def paired_bootstrap_ci(diffs, n_boot=2000, seed=42):
    """Return (mean_diff, ci_low, ci_high, p_value_gt_zero) across season-level paired lifts."""
    valid = [d for d in diffs if d is not None and math.isfinite(d)]
    if len(valid) < 2:
        m = valid[0] if valid else None
        return m, None, None, None
    rng = random.Random(seed)
    n = len(valid)
    boots = [mean(rng.choices(valid, k=n)) for _ in range(n_boot)]
    boots.sort()
    p_val = sum(b <= 0 for b in boots) / n_boot
    return mean(valid), boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)], p_val


def evaluate_band_picks(rows, split_label, top_pct):
    """Evaluate top-pct selections within each 25-slot ADP band + paired season bootstrap."""
    band_edges = ((0, 25), (25, 50), (50, 75), (75, 100), (100, 125), (125, 150))
    summary = []
    for lo, hi in band_edges:
        label = f"{lo + 1}-{hi}"
        pool = [r for r in rows if lo < r["adp"] <= hi]
        if not pool:
            continue
        seasons = sorted({r["draft_season"] for r in pool})
        for model in ("market_pool", *MODEL_NAMES):
            if model == "market_pool":
                picks = pool
            else:
                idx = select_top_pct_indices(pool, f"{model}_score", top_pct)
                picks = [pool[i] for i in sorted(idx)]
            hits, ranked = adp_hit_counts(picks)
            margins = [r["fppg_above_adp"] for r in picks if r["fppg_above_adp"] is not None]
            pm = price_band_metrics(pool, picks)

            season_lifts = []
            if model != "market_pool":
                for s in seasons:
                    s_pool = [r for r in pool if r["draft_season"] == s]
                    s_picks = [r for r in picks if r["draft_season"] == s]
                    s_pm = price_band_metrics(s_pool, s_picks)
                    if s_pm["price_matched_fppg_lift"] is not None:
                        season_lifts.append(s_pm["price_matched_fppg_lift"])
            _, ci_lo, ci_hi, p_val = paired_bootstrap_ci(season_lifts)

            summary.append({
                "split": split_label,
                "adp_band": label,
                "model": model,
                "seasons": len(seasons),
                "selected_players": len(picks),
                "selected_ranked": ranked,
                "selected_unranked": len(picks) - ranked,
                "selected_hits": hits,
                "hit_rate_pct": round(100 * hits / ranked, 2) if ranked else None,
                "avg_fppg_above_adp": round(mean(margins), 3) if margins else None,
                "price_matched_fppg_lift": pm["price_matched_fppg_lift"],
                "price_matched_hit_rate_lift_pp": pm["price_matched_hit_rate_lift_pp"],
                "season_lift_ci95_low": round(ci_lo, 3) if ci_lo is not None else None,
                "season_lift_ci95_high": round(ci_hi, 3) if ci_hi is not None else None,
                "season_lift_p_value": round(p_val, 4) if p_val is not None else None,
            })
    return summary


def evaluate_draft_boards(scored_rows, market_pools, split_label):
    """Build and score top-150 boards (band-reordered) for market and each model."""
    score_lookup = {(r["draft_season"], r["athlete_id"]): r for r in scored_rows}
    seasons = sorted({r["draft_season"] for r in scored_rows} & set(market_pools))
    board_summary = []
    board_rankings = []

    for model in ("market", *MODEL_NAMES):
        placements = []
        season_target_lifts = []
        for season in seasons:
            pool = market_pools[season]
            season_targets = []
            market_targets = []
            for first, last in BANDS:
                band = pool[first - 1:last]
                market_targets.extend(band[:TARGETS_PER_BAND])
                if model == "market":
                    ordered = band
                else:
                    col = f"{model}_score"
                    scored_band = []
                    for item in band:
                        rec = score_lookup.get((season, item["athlete_id"]))
                        if rec is not None and rec.get(col) is not None:
                            scored_band.append((rec[col], item))
                    scored_band.sort(key=lambda pair: (-pair[0], pair[1]["market_rank"]))
                    it = iter(item for _, item in scored_band)
                    ordered = [
                        next(it) if (season, item["athlete_id"]) in score_lookup else item
                        for item in band
                    ]
                for rank, item in enumerate(ordered, start=first):
                    if rank <= BOARD_SIZE:
                        is_target = (rank - first) < TARGETS_PER_BAND
                        placements.append((rank, item, is_target))
                        board_rankings.append({
                            "split": split_label,
                            "draft_season": season,
                            "model": model,
                            "board_rank": rank,
                            "band": f"{first}-{last}",
                            "band_target": is_target,
                            "player": item["player"],
                            "adp": item["adp"],
                            "market_rank": item["market_rank"],
                            "actual_rank": item["actual_rank"],
                            "actual_fppg": item["actual_fppg"],
                            "beat_adp": item["beat_adp"],
                            "fppg_above_adp": item["fppg_above_adp"],
                        })
                        if is_target:
                            season_targets.append(item)

            t_margins = [x["fppg_above_adp"] for x in season_targets if x["fppg_above_adp"] is not None]
            m_margins = [x["fppg_above_adp"] for x in market_targets if x["fppg_above_adp"] is not None]
            if t_margins and m_margins:
                season_target_lifts.append(mean(t_margins) - mean(m_margins))

        ranked = [(rank, x) for rank, x, _ in placements if x["actual_rank"] is not None]
        targets = [x for _, x, t in placements if t]
        t_margins = [x["fppg_above_adp"] for x in targets if x["fppg_above_adp"] is not None]
        entrants = [x for _, x, _ in placements if x["market_rank"] > BOARD_SIZE]
        _, ci_lo, ci_hi, p_val = (
            paired_bootstrap_ci(season_target_lifts) if model != "market" else (None, None, None, None)
        )

        board_summary.append({
            "split": split_label,
            "model": model,
            "seasons": len(seasons),
            "board_slots": len(placements),
            "actual_top_150_captured": sum(x["actual_rank"] <= BOARD_SIZE for _, x in ranked),
            "top_150_capture_pct": round(100 * sum(x["actual_rank"] <= BOARD_SIZE for _, x in ranked) / len(placements), 2) if placements else None,
            "mean_abs_rank_error": round(mean(abs(rank - x["actual_rank"]) for rank, x in ranked), 2) if ranked else None,
            "late_entrants": len(entrants),
            "late_entrants_top_150": sum(x["actual_rank"] is not None and x["actual_rank"] <= BOARD_SIZE for x in entrants),
            "band_targets": len(targets),
            "targets_beat_adp": sum(x["beat_adp"] is True for x in targets),
            "targets_hit_rate_pct": round(100 * sum(x["beat_adp"] is True for x in targets) / len(t_margins), 2) if t_margins else None,
            "targets_avg_fppg_above_adp": round(mean(t_margins), 3) if t_margins else None,
            "vs_market_target_lift_ci95_low": round(ci_lo, 3) if ci_lo is not None else None,
            "vs_market_target_lift_ci95_high": round(ci_hi, 3) if ci_hi is not None else None,
            "vs_market_target_p_value": round(p_val, 4) if p_val is not None else None,
        })

    return board_summary, board_rankings


def evaluate_overall(rows, split_label, top_pct):
    summary = []
    for model in ("overall_pool", *MODEL_NAMES):
        if model == "overall_pool":
            picks = rows
        else:
            picks = [r for r in rows if r[f"{model}_selected"]]
        hits, ranked = adp_hit_counts(picks)
        margins = [r["fppg_above_adp"] for r in picks if r["fppg_above_adp"] is not None]
        pm = price_band_metrics(rows, picks)
        summary.append({
            "split": split_label,
            "model": model,
            "selected_players": len(picks),
            "selected_ranked": ranked,
            "selected_unranked": len(picks) - ranked,
            "selected_hits": hits,
            "hit_rate_pct": round(100 * hits / ranked, 2) if ranked else None,
            "avg_adp": round(mean(r["adp"] for r in picks), 1) if picks else None,
            "avg_fppg_above_adp": round(mean(margins), 3) if margins else None,
            "price_matched_fppg_lift": pm["price_matched_fppg_lift"],
            "price_matched_hit_rate_lift_pp": pm["price_matched_hit_rate_lift_pp"],
        })
    return summary


def run_backtest(data, holdout_seasons=2, cv_folds=6, top_pct=25.0, max_adp=175.0):
    rows, market_pools = build_dataset(data)
    seasons = sorted({r["draft_season"] for r in rows})
    if len(seasons) <= holdout_seasons:
        raise ValueError(f"need more completed draft seasons than --holdout-seasons ({len(seasons)} found)")

    development, holdout = seasons[:-holdout_seasons], seasons[-holdout_seasons:]
    eligible_rows = [r for r in rows if r["adp"] <= max_adp]
    dev_train_all = [
        r for r in eligible_rows
        if r["draft_season"] in development and r["beat_adp"] is not None and r["fppg_above_adp"] is not None
    ]
    if len({r["beat_adp"] for r in dev_train_all}) < 2:
        raise ValueError("development seasons need both ADP hits and misses")

    # 1. Expanding-window chronological CV across development seasons
    cv_scored = []
    for season in development[1:][-cv_folds:]:
        earlier = [r for r in dev_train_all if r["draft_season"] < season]
        candidates = [r.copy() for r in eligible_rows if r["draft_season"] == season]
        if len(earlier) < 40 or len({r["beat_adp"] for r in earlier}) < 2 or not candidates:
            continue
        model_scores = fit_and_score_models(earlier, candidates)
        for i, row in enumerate(candidates):
            row["split"] = "cv"
            row["training_through"] = max(r["draft_season"] for r in earlier)
            for name, vals in model_scores.items():
                row[f"{name}_score"] = round(float(vals[i]), 6)
        cv_scored.extend(candidates)

    if cv_scored:
        attach_selection_flags(cv_scored, top_pct)

    # 2. Final holdout evaluation trained on all development seasons
    holdout_scored = [r.copy() for r in eligible_rows if r["draft_season"] in holdout]
    holdout_scores = fit_and_score_models(dev_train_all, holdout_scored)
    for i, row in enumerate(holdout_scored):
        row["split"] = "holdout"
        row["training_through"] = development[-1]
        for name, vals in holdout_scores.items():
            row[f"{name}_score"] = round(float(vals[i]), 6)
    attach_selection_flags(holdout_scored, top_pct)

    overall_summary, band_summary, board_summary, board_rankings = [], [], [], []
    for split_label, split_rows in (("cv", cv_scored), ("holdout", holdout_scored)):
        if not split_rows:
            continue
        overall_summary.extend(evaluate_overall(split_rows, split_label, top_pct))
        band_summary.extend(evaluate_band_picks(split_rows, split_label, top_pct))
        b_sum, b_rank = evaluate_draft_boards(split_rows, market_pools, split_label)
        board_summary.extend(b_sum)
        board_rankings.extend(b_rank)

    write_csv(data / "backtest_residual_summary.csv", overall_summary)
    write_csv(data / "backtest_residual_band_summary.csv", band_summary)
    write_csv(data / "backtest_residual_board_summary.csv", board_summary)
    write_csv(data / "backtest_residual_board_rankings.csv", board_rankings)
    write_csv(data / "backtest_residual_results.csv", holdout_scored)
    if cv_scored:
        write_csv(data / "backtest_residual_cv_results.csv", cv_scored)

    print(f"\nDevelopment drafts: {', '.join(development)}")
    print(f"Holdout drafts:     {', '.join(holdout)}")

    for split_label in ("cv", "holdout"):
        b_rows = [r for r in board_summary if r["split"] == split_label]
        if not b_rows:
            continue
        print(f"\n=== Top-150 Draft Board ({split_label.upper()}, {b_rows[0]['seasons']} seasons) ===")
        for r in b_rows:
            pval = f", p={r['vs_market_target_p_value']:.3f}" if r["vs_market_target_p_value"] is not None else ""
            print(
                f"  {r['model']:15s} | Top-150: {r['actual_top_150_captured']:4d}/{r['board_slots']} "
                f"({r['top_150_capture_pct']:5.2f}%) | MAE rank: {r['mean_abs_rank_error']:5.2f} | "
                f"Band targets: {r['targets_hit_rate_pct']:5.1f}% beat ADP, "
                f"{r['targets_avg_fppg_above_adp']:+.3f} FPPG vs ADP{pval}"
            )

        print(f"\n=== Price-Matched FPPG Lift by ADP Band ({split_label.upper()}, top {top_pct:g}%) ===")
        bands = sorted({r["adp_band"] for r in band_summary if r["split"] == split_label}, key=lambda s: int(s.split("-")[0]))
        header = f"  {'ADP Band':9s} | " + " | ".join(f"{m:14s}" for m in MODEL_NAMES)
        print(header)
        print("  " + "-" * (len(header) - 2))
        lookup = {(r["adp_band"], r["model"]): r for r in band_summary if r["split"] == split_label}
        for band in bands:
            cells = []
            for m in MODEL_NAMES:
                rec = lookup.get((band, m))
                lift = rec["price_matched_fppg_lift"] if rec else None
                cells.append(f"{lift:+14.2f}" if lift is not None else f"{'n/a':>14s}")
            print(f"  {band:9s} | " + " | ".join(cells))

    print(f"\nSaved outputs under {data}/backtest_residual_*.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"), help="download folder (default: data)")
    parser.add_argument("--holdout-seasons", type=int, default=2, help="latest draft seasons held out (default: 2)")
    parser.add_argument("--cv-folds", type=int, default=6, help="max chronological CV folds (default: 6)")
    parser.add_argument("--top-pct", type=float, default=25.0, help="selection pct for band/overall checks (default: 25)")
    parser.add_argument("--max-adp", type=float, default=175.0, help="max ADP included in model training/eval (default: 175)")
    args = parser.parse_args()
    if args.holdout_seasons < 1 or args.cv_folds < 1 or not 0 < args.top_pct <= 100:
        parser.error("use valid positive holdout-seasons, cv-folds, and 0 < top-pct <= 100")
    try:
        run_backtest(**vars(args))
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(1, f"Residual board backtest failed: {error}\n")


if __name__ == "__main__":
    main()