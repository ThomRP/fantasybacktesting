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
from sklearn.preprocessing import StandardScaler

from backtest_clusters import FEATURES, draft_rows, feature_matrix
from backtest_common import adp_hit_counts, value_metrics, write_csv


def validation_splits(training):
    """Validate on a whole draft season using only earlier drafts for each fit."""
    years = sorted({row["draft_season"] for row in training})
    splits = []
    for year in years[1:]:
        train = [i for i, row in enumerate(training) if row["draft_season"] < year]
        validate = [i for i, row in enumerate(training) if row["draft_season"] == year]
        if len({training[i]["beat_adp"] for i in train}) == len({training[i]["beat_adp"] for i in validate}) == 2:
            splits.append((train, validate))
    return splits[-3:]


def model_specs():
    def logistic():
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), LogisticRegression(max_iter=1000),
        )

    return (
        ("adp_only", ("adp",), logistic(), {"logisticregression__C": [0.1, 1.0, 10.0]}),
        ("logistic", FEATURES, logistic(), {"logisticregression__C": [0.1, 1.0, 10.0]}),
        (
            "boosted_trees", FEATURES,
            make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), HistGradientBoostingClassifier(
                max_iter=120, learning_rate=0.05, max_depth=2, early_stopping=False, random_state=42,
            )),
            {"histgradientboostingclassifier__max_depth": [2, 3],
             "histgradientboostingclassifier__min_samples_leaf": [20, 40]},
        ),
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


def run_backtest(data, holdout_seasons=2, top_pct=25):
    rows = list(draft_rows(data))
    seasons = sorted({row["draft_season"] for row in rows})
    if len(seasons) <= holdout_seasons:
        raise ValueError("need more completed draft seasons than --holdout-seasons")
    train_years, test_years = seasons[:-holdout_seasons], seasons[-holdout_seasons:]
    training = [row for row in rows if row["draft_season"] in train_years and row["beat_adp"] is not None]
    evaluated = [row for row in rows if row["draft_season"] in test_years]
    targets = [row["beat_adp"] for row in training]
    if len(set(targets)) < 2:
        raise ValueError("training drafts need at least one ranked hit and one ranked miss")

    splits = validation_splits(training)
    ranked_indices = [i for i, row in enumerate(evaluated) if row["beat_adp"] is not None]
    holdout_targets = [evaluated[i]["beat_adp"] for i in ranked_indices]
    baseline = 100 * sum(holdout_targets) / len(holdout_targets) if holdout_targets else None
    summary = [summary_row("overall_holdout", evaluated, baseline)]

    for name, features, model, grid in model_specs():
        matrix = feature_matrix(training, features)
        params = None
        if splits:
            search = GridSearchCV(model, grid, cv=splits, scoring="average_precision", error_score="raise")
            search.fit(matrix, targets)
            model, params = search.best_estimator_, search.best_params_
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

    write_csv(data / "backtest_model_summary.csv", summary)
    write_csv(data / "backtest_model_results.csv", evaluated)
    print(f"Training drafts: {', '.join(train_years)}")
    print(f"Holdout drafts: {', '.join(test_years)}")
    if splits:
        validation_years = [training[validation[0]]["draft_season"] for _, validation in splits]
        print(f"Parameter validation drafts: {', '.join(validation_years)}")
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
    args = parser.parse_args()
    if args.holdout_seasons < 1 or not 0 < args.top_pct <= 100:
        parser.error("use at least one holdout season and a selection percentage greater than 0 and at most 100")
    try:
        run_backtest(args.data, args.holdout_seasons, args.top_pct)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(1, f"Backtest failed: {error}\n")


if __name__ == "__main__":
    main()
