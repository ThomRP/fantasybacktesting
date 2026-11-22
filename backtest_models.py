"""Compare supervised models for beating ADP by fantasy points per game."""

import argparse
import csv
import json
import math
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from backtest_clusters import FEATURES, draft_rows, feature_matrix
from backtest_common import adp_hit_counts, value_metrics, write_csv


def validation_splits(training, fold_limit):
    """Validate on a whole draft season using only earlier drafts for each fit."""
    years = sorted({row["draft_season"] for row in training})
    splits = []
    for year in years[1:]:
        train = [i for i, row in enumerate(training) if row["draft_season"] < year]
        validate = [i for i, row in enumerate(training) if row["draft_season"] == year]
        if len({training[i]["beat_adp"] for i in train}) == len({training[i]["beat_adp"] for i in validate}) == 2:
            splits.append((train, validate))
    return splits[-fold_limit:]


def model_specs():
    adp_only = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), LogisticRegression(max_iter=1000),
    )
    logistic = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        PolynomialFeatures(degree=1, include_bias=False),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    trees = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.05, max_depth=2, early_stopping=False, random_state=42,
        ),
    )

    return (
        ("adp_only", ("adp",), adp_only, {"logisticregression__C": [0.1, 1.0, 10.0]}),
        ("logistic", FEATURES, logistic, {
            "logisticregression__C": [0.03, 0.1, 1.0, 10.0],
            "polynomialfeatures__degree": [1, 2],
        }),
        ("boosted_trees", FEATURES, trees, {
            "histgradientboostingclassifier__max_depth": [2, 3],
            "histgradientboostingclassifier__min_samples_leaf": [20, 40],
            "histgradientboostingclassifier__l2_regularization": [0.0, 1.0],
        }),
    )


def select_top_picks(rows, probabilities, top_pct):
    picks = set()
    for year in sorted({row["draft_season"] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row["draft_season"] == year]
        indices.sort(key=lambda i: (-probabilities[i], rows[i]["adp"], rows[i]["athlete_id"]))
        picks.update(indices[:math.ceil(len(indices) * top_pct / 100)])
    return picks


def summary_row(name, selected, baseline, *, targets=(), probabilities=(), params=None):
    hits, ranked = adp_hit_counts(selected)
    rate = 100 * hits / ranked if ranked else None
    metrics = value_metrics(selected, per_game=True)
    has_both = len(set(targets)) == 2
    return {
        "model": name,
        "selected_players": len(selected),
        "selected_ranked": ranked,
        "selected_unranked": len(selected) - ranked,
        "selected_hits": hits,
        "selected_hit_rate_pct": round(rate, 2) if rate is not None else None,
        "selected_lift_pp": round(rate - baseline, 2) if rate is not None and baseline is not None else None,
        "selected_avg_fppg_above_adp": metrics["avg_fppg_above_adp"],
        "selected_median_fppg_above_adp": metrics["median_fppg_above_adp"],
        "holdout_average_precision": round(average_precision_score(targets, probabilities), 4) if has_both else None,
        "holdout_roc_auc": round(roc_auc_score(targets, probabilities), 4) if has_both else None,
        "validation_parameters": json.dumps(params, sort_keys=True) if params else None,
    }


def evaluate_split(training, evaluated, top_pct, fold_limit, *, include_grid=False):
    targets = [row["beat_adp"] for row in training]
    if len(set(targets)) < 2:
        raise ValueError("training drafts need at least one ranked hit and one ranked miss")

    splits = validation_splits(training, fold_limit)
    tuning_drafts = [training[validation[0]]["draft_season"] for _, validation in splits]
    ranked_indices = [i for i, row in enumerate(evaluated) if row["beat_adp"] is not None]
    holdout_targets = [evaluated[i]["beat_adp"] for i in ranked_indices]
    baseline = 100 * sum(holdout_targets) / len(holdout_targets) if holdout_targets else None
    summary = [summary_row("overall_holdout", evaluated, baseline)]
    grid_report = []

    for name, features, model, grid in model_specs():
        matrix = feature_matrix(training, features)
        params = None
        if splits:
            search = GridSearchCV(model, grid, cv=splits, scoring="average_precision", error_score="raise")
            search.fit(matrix, targets)
            model, params = search.best_estimator_, search.best_params_
            if include_grid:
                results = search.cv_results_
                grid_report.extend({
                    "model": name,
                    "parameters": json.dumps(candidate, sort_keys=True),
                    "mean_validation_ap": round(float(results["mean_test_score"][index]), 6),
                    "std_validation_ap": round(float(results["std_test_score"][index]), 6),
                    "ap_rank_within_model": int(results["rank_test_score"][index]),
                    "selected": index == search.best_index_,
                    "validation_drafts": ", ".join(tuning_drafts),
                } for index, candidate in enumerate(results["params"]))
        else:
            model.fit(matrix, targets)
        probabilities = model.predict_proba(feature_matrix(evaluated, features))[:, 1]
        selected = select_top_picks(evaluated, probabilities, top_pct)
        summary.append(summary_row(
            name, [row for i, row in enumerate(evaluated) if i in selected], baseline,
            targets=holdout_targets, probabilities=probabilities[ranked_indices], params=params,
        ))
        for i, (row, probability) in enumerate(zip(evaluated, probabilities)):
            row[f"{name}_score"] = round(float(probability), 6)
            row[f"{name}_selected"] = i in selected

    return summary, grid_report, tuning_drafts


def cross_validate(training, development, top_pct, fold_limit):
    folds = []
    for year in sorted(development)[1:][-fold_limit:]:
        earlier = [row for row in training if row["draft_season"] < year]
        if len({row["beat_adp"] for row in earlier}) < 2:
            print(f"Skipping CV for {year}: earlier drafts need both ranked hits and misses")
            continue
        print(f"Cross-validation: evaluating {year} with earlier drafts", flush=True)
        evaluated = [row.copy() for row in development[year]]
        summary, _, tuning_drafts = evaluate_split(earlier, evaluated, top_pct, fold_limit)
        folds.extend({
            "evaluation_draft": year,
            "training_through": max(row["draft_season"] for row in earlier),
            "training_ranked": len(earlier),
            "tuning_drafts": ", ".join(tuning_drafts),
            **row,
        } for row in summary)
    return folds


def run_backtest(data, holdout_seasons=2, top_pct=25, cv_folds=5):
    rows = list(draft_rows(data))
    seasons = sorted({row["draft_season"] for row in rows})
    if len(seasons) <= holdout_seasons:
        raise ValueError("need more completed draft seasons than --holdout-seasons")
    train_years, test_years = seasons[:-holdout_seasons], seasons[-holdout_seasons:]
    development = {year: [row for row in rows if row["draft_season"] == year] for year in train_years}
    training = [row for row in rows if row["draft_season"] in development and row["beat_adp"] is not None]
    evaluated = [row for row in rows if row["draft_season"] in test_years]

    cv = cross_validate(training, development, top_pct, cv_folds)
    print(f"Final holdout: evaluating {', '.join(test_years)}", flush=True)
    summary, grid_report, tuning_drafts = evaluate_split(training, evaluated, top_pct, cv_folds, include_grid=True)
    write_csv(data / "backtest_model_summary.csv", summary)
    write_csv(data / "backtest_model_results.csv", evaluated)
    write_csv(data / "backtest_model_cv.csv", cv, fieldnames=[
        "evaluation_draft", "training_through", "training_ranked", "tuning_drafts", *summary[0],
    ])
    write_csv(data / "backtest_model_parameter_search.csv", grid_report, fieldnames=[
        "model", "parameters", "mean_validation_ap", "std_validation_ap", "ap_rank_within_model", "selected",
        "validation_drafts",
    ])

    print(f"Completed {len({row['evaluation_draft'] for row in cv})} out-of-sample development folds")
    print(f"Training drafts: {', '.join(train_years)}")
    print(f"Holdout drafts: {', '.join(test_years)}")
    if tuning_drafts:
        print(f"Parameter validation drafts: {', '.join(tuning_drafts)}")
    else:
        print("Using default parameters because no earlier validation splits had both outcome classes")
    print(f"Each model selects the top {top_pct:g}% of each holdout draft by predicted score")
    for row in summary:
        if row["selected_ranked"]:
            print(f"{row['model']}: {row['selected_hits']}/{row['selected_ranked']} beat ADP "
                  f"({row['selected_hit_rate_pct']:.2f}%, {row['selected_lift_pp']:+.2f} pp vs overall), "
                  f"{row['selected_unranked']} unranked, {row['selected_avg_fppg_above_adp']:+.1f} avg FPPG vs ADP")
        else:
            print(f"{row['model']}: {row['selected_unranked']} selected players without FPPG ranks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"), help="download folder (default: data)")
    parser.add_argument("--holdout-seasons", type=int, default=2, help="latest draft seasons to evaluate (default: 2)")
    parser.add_argument("--top-pct", type=float, default=25, help="percentage selected from each holdout draft (default: 25)")
    parser.add_argument("--cv-folds", type=int, default=5, help="maximum chronological folds for validation and tuning (default: 5)")
    args = parser.parse_args()
    if args.holdout_seasons < 1 or args.cv_folds < 1 or not 0 < args.top_pct <= 100:
        parser.error("use at least one holdout season and CV fold, and a selection percentage greater than 0 and at most 100")
    try:
        run_backtest(args.data, args.holdout_seasons, args.top_pct, args.cv_folds)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(1, f"Backtest failed: {error}\n")


if __name__ == "__main__":
    main()
