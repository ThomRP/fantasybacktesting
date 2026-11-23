"""Prune dead features, ensemble top models, and evaluate banded + global Top-150 draft boards."""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, HuberRegressor, LogisticRegression, RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

from backtest_common import (
    adp_hit_counts,
    adp_slot_points,
    draft_outcomes,
    load_birth_dates,
    name_key,
    read_adp,
    read_season,
    write_csv,
)
from backtest_residual_board import (
    BASE_FEATURES,
    active_features,
    extract_rich_season_features,
    paired_bootstrap_ci,
    read_adp_platform_stats,
    to_matrix,
)
from backtest_value_benchmark import price_band_metrics


BOARD_SIZE = 150
BANDS = ((1, 25), (26, 50), (51, 75), (76, 100), (101, 125), (126, 175))
TARGETS_PER_BAND = 5
RIDGE_ALPHAS = np.logspace(-1, 3.5, 25)
LAMBDA_GRID = (0.25, 0.50, 0.75, 1.00, 1.25)
RESIDUAL_CLIP = 8.0

FEATURE_GROUPS = {
    "core_market": ("log_adp", "mkt_expected_fppg", "prior_minus_mkt_fppg"),
    "aging": ("age_at_prior_break", "age_under_24", "age_over_30"),
    "role_volume": ("prior_games", "prior_mpg", "prior_fppm", "prior_start_pct"),
    "recency_splits": (
        "shrunk_post_break_mpg_change",
        "shrunk_post_break_fppm_change",
        "shrunk_last20_mpg_delta",
        "shrunk_last20_start_delta",
    ),
    "skill_efficiency": (
        "prior_ts_pct",
        "prior_stocks_per36",
        "prior_ast_per36",
        "prior_reb_per36",
        "prior_tov_per36",
        "prior_shooting_fpts_share",
    ),
    "team_context": (
        "prior_fga_team_share",
        "prior_ast_team_share",
        "prior_reb_team_share",
        "prior_team_pace",
    ),
    "multi_year": (
        "yoy_fppg_delta",
        "yoy_mpg_delta",
        "two_yr_weighted_fppg",
        "two_yr_games_avg",
        "no_prev2_season",
    ),
    "adp_platforms": ("adp_platform_std", "espn_vs_avg"),
}

RESIDUAL_FEATURES = tuple(f for feats in FEATURE_GROUPS.values() for f in feats)
FEATURE_TO_GROUP = {f: grp for grp, feats in FEATURE_GROUPS.items() for f in feats}

MPG_FEATURES = (
    "log_adp", "mkt_expected_fppg", "age_at_prior_break", "age_under_24", "age_over_30",
    "prior_games", "prior_mpg", "prior_start_pct", "shrunk_post_break_mpg_change",
    "shrunk_last20_mpg_delta", "shrunk_last20_start_delta", "yoy_mpg_delta",
    "two_yr_games_avg", "no_prev2_season",
)

FPPM_FEATURES = (
    "log_adp", "mkt_expected_fppg", "age_at_prior_break", "age_under_24", "age_over_30",
    "prior_fppm", "prior_mpg", "shrunk_post_break_fppm_change", "prior_ts_pct",
    "prior_stocks_per36", "prior_ast_per36", "prior_reb_per36", "prior_tov_per36",
    "prior_shooting_fpts_share", "prior_fga_team_share", "prior_ast_team_share",
    "prior_reb_team_share", "prior_team_pace", "yoy_fppg_delta",
    "two_yr_weighted_fppg", "no_prev2_season",
)

MODEL_NAMES = (
    "logistic_base",
    "huber_full",
    "huber_pruned",
    "decomposed_eb",
    "ensemble_equal",
    "ensemble_stacked",
)


def build_dataset(data, max_pool_adp=225.0):
    """Build multi-season player-draft rows and expanded market pools (zero leakage)."""
    seasons = {int(p.name[:4]): p for p in data.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}", p.name)}
    season_stats, rich_cache = {}, {}

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

    rows, market_pools = [], {}
    for year in sorted(seasons):
        following = seasons.get(year + 1)
        if following is None or not (following / "adp.csv").is_file():
            continue
        prior_res, next_res = get_season(year), get_season(year + 1)
        if prior_res is None or next_res is None:
            continue

        all_star_date, prior_players = prior_res
        _, next_players = next_res
        prior_rich, next_rich = get_rich(year), get_rich(year + 1)
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
            birth = births.get(aid)
            age = None
            if birth is not None:
                age = all_star_date.year - birth.year - ((all_star_date.month, all_star_date.day) < (birth.month, birth.day))

            adp = outcome["adp"]
            mkt_exp = adp_slot_points(prior_leaderboard, adp)
            pstats = platform_stats.get(name_key(name), {})

            if p2 is not None:
                yoy_fppg, yoy_mpg = pr["fppg"] - p2["fppg"], pr["mpg"] - p2["mpg"]
                two_yr_fppg, two_yr_games, no_prev2 = 0.65 * pr["fppg"] + 0.35 * p2["fppg"], 0.5 * (pr["games"] + p2["games"]), 0.0
            else:
                yoy_fppg, yoy_mpg = None, None
                two_yr_fppg, two_yr_games = pr["fppg"], float(pr["games"])
                no_prev2 = 1.0 if prev2_rich is not None else None

            rows.append({
                "prior_season": seasons[year].name,
                "draft_season": following.name,
                "player": name,
                "athlete_id": aid,
                "adp": adp,
                "age_at_prior_break": age,
                "prior_games": pr["games"],
                "prior_fppg": pr["fppg"],
                "prior_post_break_fppg_change": pr["raw_post_fppg_chg"],
                "prior_start_pct": pr["start_pct"],
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
                "next_mpg": nr["mpg"] if nr else None,
                "next_fppm": nr["fppm"] if nr else None,
                **outcome,
            })

        market = {}
        for name, adp in sorted(read_adp(following / "adp.csv"), key=lambda item: (item[1], item[0].casefold())):
            market.setdefault(name_key(name), (name, adp))
        pool = []
        for market_rank, (key, (name, adp)) in enumerate(market.items(), start=1):
            if market_rank > int(max_pool_adp) and adp > max_pool_adp:
                continue
            out = outcomes_all_by_key.get(key, {})
            pool.append({
                "draft_season": following.name,
                "player": name,
                "athlete_id": prior_id_by_key.get(key),
                "adp": adp,
                "market_rank": market_rank,
                "mkt_expected_fppg": adp_slot_points(prior_leaderboard, adp),
                "actual_fppg": out.get("next_fppg"),
                "actual_rank": out.get("next_rank"),
                "beat_adp": out.get("beat_adp"),
                "fppg_above_adp": out.get("fppg_above_adp"),
            })
        if len(pool) >= BOARD_SIZE:
            market_pools[following.name] = pool

    return rows, market_pools


def _make_spline_ridge(feat_list):
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


def _fit_predict_decomposed(training, evaluated):
    decomp_rows = [r for r in training if r.get("next_mpg") is not None and r.get("next_fppm") is not None]
    mpg_feats = active_features(decomp_rows, MPG_FEATURES)
    fppm_feats = active_features(decomp_rows, FPPM_FEATURES)

    X_mpg_tr = to_matrix(decomp_rows, mpg_feats)
    y_mpg_tr = np.array([float(r["next_mpg"]) for r in decomp_rows], dtype=float)
    X_fppm_tr = to_matrix(decomp_rows, fppm_feats)
    y_fppm_tr = np.array([float(r["next_fppm"]) for r in decomp_rows], dtype=float)

    groups = np.array([r["draft_season"] for r in decomp_rows])
    n_seasons = len(set(groups))
    if n_seasons >= 2:
        gkf = GroupKFold(n_splits=min(n_seasons, 5))
        tr_struct_fppg = (
            cross_val_predict(_make_spline_ridge(mpg_feats), X_mpg_tr, y_mpg_tr, cv=gkf, groups=groups)
            * cross_val_predict(_make_spline_ridge(fppm_feats), X_fppm_tr, y_fppm_tr, cv=gkf, groups=groups)
        )
    else:
        tr_struct_fppg = (
            _make_spline_ridge(mpg_feats).fit(X_mpg_tr, y_mpg_tr).predict(X_mpg_tr)
            * _make_spline_ridge(fppm_feats).fit(X_fppm_tr, y_fppm_tr).predict(X_fppm_tr)
        )

    mpg_model = _make_spline_ridge(mpg_feats).fit(X_mpg_tr, y_mpg_tr)
    fppm_model = _make_spline_ridge(fppm_feats).fit(X_fppm_tr, y_fppm_tr)
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
        tr_struct_fppg - tr_mkt, (tr_struct_fppg - tr_mkt) * tr_log_adp,
        tr_log_adp, np.minimum(tr_games, 70.0) / 70.0,
    ])
    meta_ev = np.column_stack([
        ev_struct_fppg - ev_mkt, (ev_struct_fppg - ev_mkt) * ev_log_adp,
        ev_log_adp, np.minimum(ev_games, 70.0) / 70.0,
    ])
    meta = make_pipeline(StandardScaler(), HuberRegressor(alpha=10.0, max_iter=1000))
    meta.fit(meta_tr, [float(r["fppg_above_adp"]) for r in decomp_rows])
    return meta.predict(meta_ev)


def select_stable_features(training, candidate_features, min_keep=8):
    """Leak-free pruning strictly inside `training`: ElasticNetCV sparsity + inner-fold sign stability."""
    feats = active_features(training, candidate_features)
    if len(feats) <= min_keep:
        return feats, {f: {"enet_coef": None, "stability": None, "sign_consistency": None} for f in feats}

    X = to_matrix(training, feats)
    y = np.array([float(r["fppg_above_adp"]) for r in training], dtype=float)
    groups = np.array([r["draft_season"] for r in training])
    n_seasons = len(set(groups))

    Xs = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()).fit_transform(X)
    splits = list(GroupKFold(n_splits=min(n_seasons, 5)).split(Xs, y, groups)) if n_seasons >= 2 else 3

    enet = ElasticNetCV(l1_ratio=[0.15, 0.4, 0.7, 0.9], cv=splits, max_iter=5000, random_state=42).fit(Xs, y)
    enet_coefs = enet.coef_

    fold_coefs = []
    if isinstance(splits, list):
        for tr_idx, _ in splits:
            fold_coefs.append(HuberRegressor(alpha=25.0, epsilon=1.35, max_iter=1000).fit(Xs[tr_idx], y[tr_idx]).coef_)
    else:
        fold_coefs.append(HuberRegressor(alpha=25.0, epsilon=1.35, max_iter=1000).fit(Xs, y).coef_)
    fold_mat = np.vstack(fold_coefs)
    mean_w, std_w = fold_mat.mean(axis=0), fold_mat.std(axis=0)
    sign_cons = np.maximum((fold_mat > 0).mean(axis=0), (fold_mat < 0).mean(axis=0))
    stability = np.abs(mean_w) / (std_w + 0.05)

    diagnostics, kept = {}, []
    for i, f in enumerate(feats):
        diagnostics[f] = {
            "huber_mean_coef": float(mean_w[i]),
            "huber_std_coef": float(std_w[i]),
            "sign_consistency": float(sign_cons[i]),
            "stability": float(stability[i]),
            "enet_coef": float(enet_coefs[i]),
        }
        if abs(enet_coefs[i]) > 0.02 and (sign_cons[i] >= 0.75 or stability[i] >= 0.85):
            kept.append(f)

    if len(kept) < min_keep:
        ranked = sorted(range(len(feats)), key=lambda i: (0 if abs(enet_coefs[i]) > 0.01 else 1, -float(stability[i])))
        kept = [feats[i] for i in ranked[:min_keep]]

    return kept, diagnostics


def _fit_huber(training, evaluated, feats, alpha=25.0):
    use_feats = active_features(training, feats)
    pipe = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        HuberRegressor(alpha=alpha, epsilon=1.35, max_iter=1000),
    )
    pipe.fit(to_matrix(training, use_feats), [float(r["fppg_above_adp"]) for r in training])
    return pipe.predict(to_matrix(evaluated, use_feats)), pipe.named_steps["huberregressor"].coef_, use_feats


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


def _fit_simplex_weights(oof_matrix, y_true, min_weight=0.10, shrink_l2=0.15):
    grid = np.linspace(min_weight, 1.0 - 2 * min_weight, 17)
    best_w, best_obj = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), float("inf")
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < min_weight - 1e-9:
                continue
            pred = w1 * oof_matrix[:, 0] + w2 * oof_matrix[:, 1] + w3 * oof_matrix[:, 2]
            pen = shrink_l2 * ((w1 - 1 / 3) ** 2 + (w2 - 1 / 3) ** 2 + (w3 - 1 / 3) ** 2)
            obj = _huber_loss_1d(y_true - pred) + pen
            if obj < best_obj:
                best_obj, best_w = obj, (float(w1), float(w2), float(w3))
    return best_w


def fit_and_score_models(training, evaluated):
    """Fit base, pruned, decomposed, and ensemble models strictly on `training`."""
    y_res = np.array([float(r["fppg_above_adp"]) for r in training], dtype=float)
    groups = np.array([r["draft_season"] for r in training])
    unique_seasons = sorted(set(groups))

    kept_feats, feat_diag = select_stable_features(training, RESIDUAL_FEATURES, min_keep=8)

    oof_logit = np.zeros(len(training), dtype=float)
    oof_pruned = np.zeros(len(training), dtype=float)
    oof_decomp = np.zeros(len(training), dtype=float)

    if len(unique_seasons) >= 2:
        gkf = GroupKFold(n_splits=min(len(unique_seasons), 5))
        for tr_idx, va_idx in gkf.split(training, y_res, groups):
            tr_sub = [training[i] for i in tr_idx]
            va_sub = [training[i] for i in va_idx]
            if len({r["beat_adp"] for r in tr_sub}) >= 2:
                _, oof_logit[va_idx] = _fit_logistic_logits(tr_sub, va_sub)
            oof_pruned[va_idx], _, _ = _fit_huber(tr_sub, va_sub, kept_feats)
            oof_decomp[va_idx] = _fit_predict_decomposed(tr_sub, va_sub)
    else:
        _, oof_logit = _fit_logistic_logits(training, training)
        oof_pruned, _, _ = _fit_huber(training, training, kept_feats)
        oof_decomp = _fit_predict_decomposed(training, training)

    _, ev_logit = _fit_logistic_logits(training, evaluated)
    ev_huber_full, full_coefs, full_feats = _fit_huber(training, evaluated, RESIDUAL_FEATURES)
    ev_huber_pruned, _, _ = _fit_huber(training, evaluated, kept_feats)
    ev_decomp = _fit_predict_decomposed(training, evaluated)

    # Calibrate logistic_base logits -> FPPG residual units strictly on OOF training predictions
    cal = HuberRegressor(alpha=5.0, max_iter=500).fit(oof_logit.reshape(-1, 1), y_res)
    if cal.coef_[0] < 0.05:
        cal.coef_[0] = 0.05
    oof_log_fppg = cal.predict(oof_logit.reshape(-1, 1))
    ev_log_fppg = cal.predict(ev_logit.reshape(-1, 1))

    ev_ens_equal = (ev_log_fppg + ev_huber_pruned + ev_decomp) / 3.0
    weights = _fit_simplex_weights(np.column_stack([oof_log_fppg, oof_pruned, oof_decomp]), y_res)
    ev_ens_stacked = weights[0] * ev_log_fppg + weights[1] * ev_huber_pruned + weights[2] * ev_decomp

    scores = {
        "logistic_base": ev_log_fppg,
        "huber_full": ev_huber_full,
        "huber_pruned": ev_huber_pruned,
        "decomposed_eb": ev_decomp,
        "ensemble_equal": ev_ens_equal,
        "ensemble_stacked": ev_ens_stacked,
    }
    meta = {
        "kept_features": kept_feats,
        "feature_diagnostics": feat_diag,
        "full_coefs": dict(zip(full_feats, map(float, full_coefs))),
        "stack_weights": weights,
    }
    return scores, meta


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


def spearman_corr(x, y):
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(np.asarray(x, dtype=float))).astype(float)
    ry = np.argsort(np.argsort(np.asarray(y, dtype=float))).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else None


def run_cv_group_ablation(dev_train_all, eval_pool, cv_seasons, top_pct):
    """Leave-one-group-out ablation of `huber_full` strictly across development CV folds."""
    def eval_subset(feats):
        all_eval = []
        for season in cv_seasons:
            earlier = [r for r in dev_train_all if r["draft_season"] < season]
            cands = [r.copy() for r in eval_pool if r["draft_season"] == season and r["adp"] <= 150]
            if len(earlier) < 40 or not cands:
                continue
            preds, _, _ = _fit_huber(earlier, cands, feats)
            for row, p in zip(cands, preds):
                row["score"] = float(p)
            all_eval.extend(cands)
        band_lifts = []
        for lo, hi in ((0, 25), (25, 50), (50, 75), (75, 100), (100, 125), (125, 150)):
            pool = [r for r in all_eval if lo < r["adp"] <= hi]
            if not pool:
                continue
            picks = [pool[i] for i in sorted(select_top_pct_indices(pool, "score", top_pct))]
            pm = price_band_metrics(pool, picks)
            if pm["price_matched_fppg_lift"] is not None:
                band_lifts.append(pm["price_matched_fppg_lift"])
        errs = [abs(r["score"] - r["fppg_above_adp"]) for r in all_eval if r["fppg_above_adp"] is not None]
        return (mean(band_lifts) if band_lifts else None), (mean(errs) if errs else None)

    base_lift, base_mae = eval_subset(RESIDUAL_FEATURES)
    ablation = [{
        "dropped_group": "NONE (all 31 features)",
        "features_in_group": len(RESIDUAL_FEATURES),
        "cv_avg_band_fppg_lift": round(base_lift, 3) if base_lift is not None else None,
        "lift_change_if_dropped": 0.0,
        "cv_residual_mae": round(base_mae, 3) if base_mae is not None else None,
        "verdict": "baseline",
    }]
    for grp, grp_feats in FEATURE_GROUPS.items():
        lift, mae = eval_subset(tuple(f for f in RESIDUAL_FEATURES if f not in grp_feats))
        delta = (lift - base_lift) if (lift is not None and base_lift is not None) else None
        verdict = "n/a" if delta is None else ("HURTS (drop)" if delta > 0.05 else ("HELPS (keep)" if delta < -0.05 else "neutral"))
        ablation.append({
            "dropped_group": grp,
            "features_in_group": len(grp_feats),
            "cv_avg_band_fppg_lift": round(lift, 3) if lift is not None else None,
            "lift_change_if_dropped": round(delta, 3) if delta is not None else None,
            "cv_residual_mae": round(mae, 3) if mae is not None else None,
            "verdict": verdict,
        })
    return ablation


def evaluate_band_picks(rows, split_label, top_pct):
    summary = []
    for lo, hi in ((0, 25), (25, 50), (50, 75), (75, 100), (100, 125), (125, 150)):
        label = f"{lo + 1}-{hi}"
        pool = [r for r in rows if lo < r["adp"] <= hi]
        if not pool:
            continue
        seasons = sorted({r["draft_season"] for r in pool})
        for model in ("market_pool", *MODEL_NAMES):
            picks = pool if model == "market_pool" else [pool[i] for i in sorted(select_top_pct_indices(pool, f"{model}_score", top_pct))]
            hits, ranked = adp_hit_counts(picks)
            margins = [r["fppg_above_adp"] for r in picks if r["fppg_above_adp"] is not None]
            pm = price_band_metrics(pool, picks)
            season_lifts = []
            if model != "market_pool":
                for s in seasons:
                    s_pm = price_band_metrics([r for r in pool if r["draft_season"] == s], [r for r in picks if r["draft_season"] == s])
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


def evaluate_banded_boards(scored_rows, market_pools, split_label):
    score_lookup = {(r["draft_season"], r["athlete_id"]): r for r in scored_rows}
    seasons = sorted({r["draft_season"] for r in scored_rows} & set(market_pools))
    board_summary = []
    for model in ("market", *MODEL_NAMES):
        placements, season_target_lifts = [], []
        for season in seasons:
            pool = market_pools[season]
            season_targets, market_targets = [], []
            for first, last in BANDS:
                band = pool[first - 1:last]
                market_targets.extend(band[:TARGETS_PER_BAND])
                if model == "market":
                    ordered = band
                else:
                    col = f"{model}_score"
                    scored_band = [
                        (score_lookup[(season, item["athlete_id"])][col], item)
                        for item in band
                        if (season, item["athlete_id"]) in score_lookup
                        and score_lookup[(season, item["athlete_id"])].get(col) is not None
                    ]
                    scored_band.sort(key=lambda pair: (-pair[0], pair[1]["market_rank"]))
                    it = iter(item for _, item in scored_band)
                    ordered = [next(it) if (season, item["athlete_id"]) in score_lookup else item for item in band]
                for rank, item in enumerate(ordered, start=first):
                    if rank <= BOARD_SIZE:
                        is_target = (rank - first) < TARGETS_PER_BAND
                        placements.append((rank, item, is_target))
                        if is_target:
                            season_targets.append(item)
            t_m = [x["fppg_above_adp"] for x in season_targets if x["fppg_above_adp"] is not None]
            m_m = [x["fppg_above_adp"] for x in market_targets if x["fppg_above_adp"] is not None]
            if t_m and m_m:
                season_target_lifts.append(mean(t_m) - mean(m_m))

        ranked = [(rank, x) for rank, x, _ in placements if x["actual_rank"] is not None]
        targets = [x for _, x, t in placements if t]
        t_margins = [x["fppg_above_adp"] for x in targets if x["fppg_above_adp"] is not None]
        _, _, _, p_val = paired_bootstrap_ci(season_target_lifts) if model != "market" else (None, None, None, None)
        board_summary.append({
            "split": split_label,
            "model": model,
            "seasons": len(seasons),
            "board_slots": len(placements),
            "actual_top_150_captured": sum(x["actual_rank"] <= BOARD_SIZE for _, x in ranked),
            "top_150_capture_pct": round(100 * sum(x["actual_rank"] <= BOARD_SIZE for _, x in ranked) / len(placements), 2) if placements else None,
            "mean_abs_rank_error": round(mean(abs(rank - x["actual_rank"]) for rank, x in ranked), 2) if ranked else None,
            "targets_hit_rate_pct": round(100 * sum(x["beat_adp"] is True for x in targets) / len(t_margins), 2) if t_margins else None,
            "targets_avg_fppg_above_adp": round(mean(t_margins), 3) if t_margins else None,
            "vs_market_p_value": round(p_val, 4) if p_val is not None else None,
        })
    return board_summary


def build_global_board(pool, score_lookup, season, model, lam):
    scored = []
    for item in pool:
        adj = 0.0
        if model != "market":
            rec = score_lookup.get((season, item["athlete_id"]))
            if rec is not None and rec.get(f"{model}_score") is not None:
                adj = lam * max(-RESIDUAL_CLIP, min(RESIDUAL_CLIP, float(rec[f"{model}_score"])))
        scored.append((item["mkt_expected_fppg"] + adj, adj, item))
    scored.sort(key=lambda t: (-t[0], t[2]["market_rank"]))
    return scored


def tune_global_lambdas(cv_rows, market_pools):
    score_lookup = {(r["draft_season"], r["athlete_id"]): r for r in cv_rows}
    seasons = sorted({r["draft_season"] for r in cv_rows} & set(market_pools))
    best_lambdas = {"market": 0.0}
    for model in MODEL_NAMES:
        best_lam, best_score = 0.5, (-1e9, -1e9)
        for lam in LAMBDA_GRID:
            top150_fppg, top150_hits = [], 0
            for season in seasons:
                for _, _, item in build_global_board(market_pools[season], score_lookup, season, model, lam)[:BOARD_SIZE]:
                    if item["actual_fppg"] is not None:
                        top150_fppg.append(item["actual_fppg"])
                    if item["actual_rank"] is not None and item["actual_rank"] <= BOARD_SIZE:
                        top150_hits += 1
            cand = (mean(top150_fppg) if top150_fppg else -1e9, top150_hits)
            if cand > best_score:
                best_score, best_lam = cand, lam
        best_lambdas[model] = best_lam
    return best_lambdas


def evaluate_global_boards(scored_rows, market_pools, split_label, lambdas):
    score_lookup = {(r["draft_season"], r["athlete_id"]): r for r in scored_rows}
    seasons = sorted({r["draft_season"] for r in scored_rows} & set(market_pools))
    market_fppg = {
        s: mean([x["actual_fppg"] for x in market_pools[s][:BOARD_SIZE] if x["actual_fppg"] is not None])
        for s in seasons
    }
    summary, rankings = [], []

    for model in ("market", *MODEL_NAMES):
        lam = lambdas.get(model, 0.5)
        board_items, added, dropped, risers, fallers, season_lifts, spearmans = [], [], [], [], [], [], []
        for season in seasons:
            ordered = build_global_board(market_pools[season], score_lookup, season, model, lam)
            top150, outside = ordered[:BOARD_SIZE], ordered[BOARD_SIZE:]
            s_fppg = [item["actual_fppg"] for _, _, item in top150 if item["actual_fppg"] is not None]
            if s_fppg:
                season_lifts.append(mean(s_fppg) - market_fppg[season])

            r_ranks, a_ranks = [], []
            for rank, (board_fppg, adj, item) in enumerate(top150, start=1):
                board_items.append((rank, item))
                delta = item["market_rank"] - rank
                if item["actual_rank"] is not None:
                    r_ranks.append(rank)
                    a_ranks.append(item["actual_rank"])
                if item["market_rank"] > BOARD_SIZE:
                    added.append(item)
                if delta >= 10:
                    risers.append(item)
                elif delta <= -10:
                    fallers.append(item)
                rankings.append({
                    "split": split_label, "draft_season": season, "model": model, "lambda": lam,
                    "board_rank": rank, "market_rank": item["market_rank"], "slot_move": delta,
                    "board_proj_fppg": round(board_fppg, 3), "model_fppg_adj": round(adj, 3),
                    "player": item["player"], "adp": item["adp"], "actual_rank": item["actual_rank"],
                    "actual_fppg": item["actual_fppg"], "beat_adp": item["beat_adp"], "fppg_above_adp": item["fppg_above_adp"],
                })
            for rank, (_, _, item) in enumerate(outside, start=BOARD_SIZE + 1):
                if item["market_rank"] <= BOARD_SIZE:
                    dropped.append(item)
                    if (item["market_rank"] - rank) <= -10:
                        fallers.append(item)
            sp = spearman_corr(r_ranks, a_ranks)
            if sp is not None:
                spearmans.append(sp)

        ranked = [(rank, x) for rank, x in board_items if x["actual_rank"] is not None]
        all_fppg = [x["actual_fppg"] for _, x in board_items if x["actual_fppg"] is not None]
        add_fppg = [x["actual_fppg"] for x in added if x["actual_fppg"] is not None]
        drp_fppg = [x["actual_fppg"] for x in dropped if x["actual_fppg"] is not None]
        riser_m = [x["fppg_above_adp"] for x in risers if x["fppg_above_adp"] is not None]
        faller_m = [x["fppg_above_adp"] for x in fallers if x["fppg_above_adp"] is not None]
        _, _, _, p_val = paired_bootstrap_ci(season_lifts) if model != "market" else (None, None, None, None)

        summary.append({
            "split": split_label,
            "model": model,
            "lambda": lam,
            "seasons": len(seasons),
            "board_slots": len(board_items),
            "unranked_on_board": len(board_items) - len(ranked),
            "actual_top_150_captured": sum(x["actual_rank"] <= BOARD_SIZE for _, x in ranked),
            "top_150_capture_pct": round(100 * sum(x["actual_rank"] <= BOARD_SIZE for _, x in ranked) / len(board_items), 2) if board_items else None,
            "board_mean_actual_fppg": round(mean(all_fppg), 3) if all_fppg else None,
            "mean_abs_rank_error": round(mean(abs(rank - x["actual_rank"]) for rank, x in ranked), 2) if ranked else None,
            "spearman_rho": round(mean(spearmans), 4) if spearmans else None,
            "swaps_count": len(added),
            "added_top150_pct": round(100 * sum(x["actual_rank"] is not None and x["actual_rank"] <= BOARD_SIZE for x in added) / len(added), 1) if added else None,
            "dropped_top150_pct": round(100 * sum(x["actual_rank"] is not None and x["actual_rank"] <= BOARD_SIZE for x in dropped) / len(dropped), 1) if dropped else None,
            "added_minus_dropped_fppg": round(mean(add_fppg) - mean(drp_fppg), 2) if (add_fppg and drp_fppg) else None,
            "risers_10plus": len(risers),
            "risers_avg_fppg_vs_adp": round(mean(riser_m), 2) if riser_m else None,
            "fallers_10plus": len(fallers),
            "fallers_avg_fppg_vs_adp": round(mean(faller_m), 2) if faller_m else None,
            "vs_market_board_fppg_p": round(p_val, 4) if p_val is not None else None,
        })
    return summary, rankings


def run_backtest(data, holdout_seasons=2, cv_folds=6, top_pct=25.0, max_train_adp=175.0, max_pool_adp=225.0):
    rows, market_pools = build_dataset(data, max_pool_adp=max_pool_adp)
    seasons = sorted({r["draft_season"] for r in rows})
    if len(seasons) <= holdout_seasons:
        raise ValueError(f"need more completed draft seasons than --holdout-seasons ({len(seasons)} found)")

    development, holdout = seasons[:-holdout_seasons], seasons[-holdout_seasons:]
    train_pool = [
        r for r in rows
        if r["draft_season"] in development and r["adp"] <= max_train_adp
        and r["beat_adp"] is not None and r["fppg_above_adp"] is not None
    ]
    eval_pool = [r for r in rows if r["adp"] <= max_pool_adp]
    if len({r["beat_adp"] for r in train_pool}) < 2:
        raise ValueError("development seasons need both ADP hits and misses")

    cv_seasons = development[1:][-cv_folds:]
    cv_scored, cv_keep_counts, cv_coef_history, cv_weights = [], defaultdict(int), defaultdict(list), []
    for season in cv_seasons:
        earlier = [r for r in train_pool if r["draft_season"] < season]
        candidates = [r.copy() for r in eval_pool if r["draft_season"] == season]
        if len(earlier) < 40 or len({r["beat_adp"] for r in earlier}) < 2 or not candidates:
            continue
        model_scores, meta = fit_and_score_models(earlier, candidates)
        cv_weights.append((season, meta["stack_weights"]))
        for f in meta["kept_features"]:
            cv_keep_counts[f] += 1
        for f, w in meta["full_coefs"].items():
            cv_coef_history[f].append(w)
        for i, row in enumerate(candidates):
            row["split"] = "cv"
            row["training_through"] = max(r["draft_season"] for r in earlier)
            for name, vals in model_scores.items():
                row[f"{name}_score"] = round(float(vals[i]), 6)
        cv_scored.extend(candidates)

    if cv_scored:
        attach_selection_flags(cv_scored, top_pct)

    ablation_rows = run_cv_group_ablation(train_pool, eval_pool, cv_seasons, top_pct)
    best_lambdas = tune_global_lambdas(cv_scored, market_pools)

    holdout_scored = [r.copy() for r in eval_pool if r["draft_season"] in holdout]
    holdout_scores, holdout_meta = fit_and_score_models(train_pool, holdout_scored)
    for i, row in enumerate(holdout_scored):
        row["split"] = "holdout"
        row["training_through"] = development[-1]
        for name, vals in holdout_scores.items():
            row[f"{name}_score"] = round(float(vals[i]), 6)
    attach_selection_flags(holdout_scored, top_pct)

    n_cv_used = max(len(cv_weights), 1)
    kept_final = set(holdout_meta["kept_features"])
    feature_audit = []
    for f in RESIDUAL_FEATURES:
        hist = cv_coef_history.get(f, [])
        diag = holdout_meta["feature_diagnostics"].get(f, {})
        sign_cons = max(sum(w > 0 for w in hist) / len(hist), sum(w < 0 for w in hist) / len(hist)) if hist else None
        feature_audit.append({
            "feature": f,
            "group": FEATURE_TO_GROUP[f],
            "cv_mean_huber_weight": round(mean(hist), 4) if hist else None,
            "cv_sign_consistency": round(sign_cons, 2) if sign_cons is not None else None,
            "cv_kept_folds": f"{cv_keep_counts[f]}/{n_cv_used}",
            "final_enet_weight": round(diag["enet_coef"], 4) if diag.get("enet_coef") is not None else None,
            "kept_in_final_model": f in kept_final,
        })
    feature_audit.sort(key=lambda r: (not r["kept_in_final_model"], -abs(r["final_enet_weight"] or 0.0)))

    band_summary, banded_summary, global_summary, global_rankings = [], [], [], []
    for split_label, split_rows in (("cv", cv_scored), ("holdout", holdout_scored)):
        if not split_rows:
            continue
        band_summary.extend(evaluate_band_picks(split_rows, split_label, top_pct))
        banded_summary.extend(evaluate_banded_boards(split_rows, market_pools, split_label))
        g_sum, g_rank = evaluate_global_boards(split_rows, market_pools, split_label, best_lambdas)
        global_summary.extend(g_sum)
        global_rankings.extend(g_rank)

    write_csv(data / "backtest_ensemble_feature_audit.csv", feature_audit)
    write_csv(data / "backtest_ensemble_group_ablation.csv", ablation_rows)
    write_csv(data / "backtest_ensemble_band_summary.csv", band_summary)
    write_csv(data / "backtest_ensemble_banded_board_summary.csv", banded_summary)
    write_csv(data / "backtest_ensemble_global_board_summary.csv", global_summary)
    write_csv(data / "backtest_ensemble_global_board_rankings.csv", global_rankings)
    write_csv(data / "backtest_ensemble_results.csv", holdout_scored)
    if cv_scored:
        write_csv(data / "backtest_ensemble_cv_results.csv", cv_scored)

    w = holdout_meta["stack_weights"]
    print(f"\nDevelopment drafts: {', '.join(development)}")
    print(f"Holdout drafts:     {', '.join(holdout)}")
    print(f"Final stacked weights (logistic / huber_pruned / decomposed_eb): {w[0]:.2f} / {w[1]:.2f} / {w[2]:.2f}")
    print(f"Pruned Huber kept {len(kept_final)}/{len(RESIDUAL_FEATURES)} features: {', '.join(holdout_meta['kept_features'])}")

    print("\n=== Feature Group Leave-One-Out Ablation (CV, Huber Residual) ===")
    for r in ablation_rows:
        print(
            f"  {r['dropped_group']:24s} ({r['features_in_group']:2d} feats) | "
            f"CV avg band lift: {r['cv_avg_band_fppg_lift']:+.3f} "
            f"(delta: {r['lift_change_if_dropped']:+.3f}) | MAE: {r['cv_residual_mae']:.3f} | {r['verdict']}"
        )

    for split_label in ("cv", "holdout"):
        b_rows = [r for r in banded_summary if r["split"] == split_label]
        if b_rows:
            print(f"\n=== BANDED Top-150 Board ({split_label.upper()}, {b_rows[0]['seasons']} seasons) ===")
            for r in b_rows:
                pval = f", p={r['vs_market_p_value']:.3f}" if r["vs_market_p_value"] is not None else ""
                print(
                    f"  {r['model']:16s} | Top150: {r['actual_top_150_captured']:3d}/{r['board_slots']} "
                    f"({r['top_150_capture_pct']:5.2f}%) | MAE: {r['mean_abs_rank_error']:5.2f} | "
                    f"Band targets: {r['targets_hit_rate_pct']:5.1f}% beat ADP, "
                    f"{r['targets_avg_fppg_above_adp']:+.3f} FPPG vs ADP{pval}"
                )

        g_rows = [r for r in global_summary if r["split"] == split_label]
        if not g_rows:
            continue
        print(f"\n=== GLOBAL UNBANDED Top-150 Board ({split_label.upper()}, {g_rows[0]['seasons']} seasons) ===")
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
                f"({r['top_150_capture_pct']:5.2f}%) | AvgFPPG: {r['board_mean_actual_fppg']:5.2f}{pval} | "
                f"MAE: {r['mean_abs_rank_error']:5.2f} | {swap_str}"
            )

        print(f"\n=== Big Movers on Global Board ({split_label.upper()}, >=10 slots vs ADP) ===")
        for r in g_rows:
            if r["model"] == "market":
                continue
            r_val = f"{r['risers_avg_fppg_vs_adp']:+.2f}" if r["risers_avg_fppg_vs_adp"] is not None else "  n/a"
            f_val = f"{r['fallers_avg_fppg_vs_adp']:+.2f}" if r["fallers_avg_fppg_vs_adp"] is not None else "  n/a"
            print(
                f"  {r['model']:16s} | Risers ({r['risers_10plus']:3d}): {r_val} FPPG vs ADP | "
                f"Fallers ({r['fallers_10plus']:3d}): {f_val} FPPG vs ADP"
            )

        print(f"\n=== Price-Matched FPPG Lift by ADP Band ({split_label.upper()}, top {top_pct:g}%) ===")
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

    print(f"\nSaved outputs under {data}/backtest_ensemble_*.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"), help="download folder (default: data)")
    parser.add_argument("--holdout-seasons", type=int, default=2, help="latest draft seasons held out (default: 2)")
    parser.add_argument("--cv-folds", type=int, default=6, help="max chronological CV folds (default: 6)")
    parser.add_argument("--top-pct", type=float, default=25.0, help="selection pct for band checks (default: 25)")
    parser.add_argument("--max-train-adp", type=float, default=175.0, help="max ADP used to fit models (default: 175)")
    parser.add_argument("--max-pool-adp", type=float, default=225.0, help="max ADP allowed into global Top-150 pool (default: 225)")
    args = parser.parse_args()
    if args.holdout_seasons < 1 or args.cv_folds < 1 or not 0 < args.top_pct <= 100:
        parser.error("use valid positive holdout-seasons, cv-folds, and 0 < top-pct <= 100")
    try:
        run_backtest(**vars(args))
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(1, f"Ensemble board backtest failed: {error}\n")


if __name__ == "__main__":
    main()