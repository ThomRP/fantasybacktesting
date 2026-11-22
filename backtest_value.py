"""Test opportunity features and direct FPPG value predictions against ADP."""

import argparse
import csv
from pathlib import Path
from statistics import mean

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from backtest_clusters import FEATURES, draft_rows, feature_matrix
from backtest_common import adp_hit_counts, write_csv
from backtest_models import select_top_picks, summary_row
from backtest_opportunity import OPPORTUNITY_FEATURES, load_opportunity_features
from backtest_value_benchmark import price_band_metrics


def model_specs():
    for objective in ("logistic", "value"):
        for feature_set, features in (("base", FEATURES), ("opportunity", FEATURES + OPPORTUNITY_FEATURES)):
            classifier = objective == "logistic"
            estimator = LogisticRegression(C=0.1, max_iter=1000) if classifier else Ridge(alpha=10.0)
            model = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                PolynomialFeatures(degree=2, include_bias=False),
                StandardScaler(),
                estimator,
            )
            yield f"{objective}_{feature_set}", features, model, classifier


def selection_summary(name, selected, evaluated, baseline, **metrics):
    report = summary_row(name, selected, baseline, **metrics)
    del report["validation_parameters"]  # This experiment uses fixed parameters for every draft.
    margins = [row["fppg_above_adp"] for row in selected if row["fppg_above_adp"] is not None]
    report["selected_avg_fppg_above_adp"] = round(mean(margins), 3) if margins else None
    report["selected_avg_adp"] = round(mean(row["adp"] for row in selected), 1) if selected else None
    report.update(price_band_metrics(evaluated, selected))
    return report


def evaluate_split(training, evaluated, top_pct):
    hits, ranked_count = adp_hit_counts(evaluated)
    baseline = 100 * hits / ranked_count if ranked_count else None
    report = [selection_summary("overall_pool", evaluated, evaluated, baseline)]
    ranked = [i for i, row in enumerate(evaluated) if row["beat_adp"] is not None]
    actual = [evaluated[i]["beat_adp"] for i in ranked]

    for name, features, model, classifier in model_specs():
        target = "beat_adp" if classifier else "fppg_above_adp"
        model.fit(feature_matrix(training, features), [row[target] for row in training])
        candidates = feature_matrix(evaluated, features)
        scores = model.predict_proba(candidates)[:, 1] if classifier else model.predict(candidates)
        picks = select_top_picks(evaluated, scores, top_pct)
        selected = [row for i, row in enumerate(evaluated) if i in picks]
        report.append(selection_summary(name, selected, evaluated, baseline, targets=actual, probabilities=scores[ranked]))
        for i, (row, score) in enumerate(zip(evaluated, scores)):
            row[f"{name}_score"] = round(float(score), 6)
            row[f"{name}_selected"] = i in picks
    return report


def run_backtest(data, *, holdout_seasons=2, cv_folds=5, top_pct=25, min_adp=0, max_adp=float("inf"),
                 save_cv_results=False):
    rows = list(draft_rows(data, extra_prior_features=load_opportunity_features))
    for row in rows:
        for feature in OPPORTUNITY_FEATURES:
            row.setdefault(feature, None)
    seasons = sorted({row["draft_season"] for row in rows})
    if len(seasons) <= holdout_seasons:
        raise ValueError("need more completed draft seasons than --holdout-seasons")

    development, holdout = seasons[:-holdout_seasons], seasons[-holdout_seasons:]
    training = [row for row in rows if row["draft_season"] in development and row["beat_adp"] is not None]
    if len({row["beat_adp"] for row in training}) < 2:
        raise ValueError("training drafts need both ranked hits and misses")
    eligible = [row for row in rows if min_adp <= row["adp"] <= max_adp]
    evaluated = [row for row in eligible if row["draft_season"] in holdout]
    if not evaluated:
        raise ValueError("no holdout candidates in the requested ADP range")

    folds = []
    cv_results = []
    for year in development[1:][-cv_folds:]:
        earlier = [row for row in training if row["draft_season"] < year]
        candidates = [row.copy() for row in eligible if row["draft_season"] == year]
        if len({row["beat_adp"] for row in earlier}) < 2 or not candidates:
            continue
        report = evaluate_split(earlier, candidates, top_pct)
        if save_cv_results:
            cv_results.extend(candidates)
        folds.extend({"evaluation_draft": year, "training_through": max(r["draft_season"] for r in earlier), **r}
                     for r in report)

    summary = evaluate_split(training, evaluated, top_pct)
    write_csv(data / "backtest_value_cv.csv", folds, fieldnames=["evaluation_draft", "training_through", *summary[0]])
    if save_cv_results:
        write_csv(data / "backtest_value_cv_results.csv", cv_results, fieldnames=list(evaluated[0]))
    write_csv(data / "backtest_value_summary.csv", summary)
    write_csv(data / "backtest_value_results.csv", evaluated)
    print(f"Training drafts: {', '.join(development)}")
    print(f"Holdout drafts: {', '.join(holdout)}")
    print(f"Selected top {top_pct:g}% per draft in the requested ADP range")
    for row in summary:
        if row["selected_ranked"]:
            print(f"{row['model']}: {row['selected_hits']}/{row['selected_ranked']} beat ADP "
                  f"({row['selected_hit_rate_pct']:.2f}%), {row['selected_unranked']} unranked, "
                  f"{row['selected_avg_fppg_above_adp']:+.2f} avg FPPG vs ADP, "
                  f"{row['price_matched_fppg_lift']:+.2f} FPPG vs same-price picks")
        else:
            print(f"{row['model']}: {row['selected_unranked']} selected players without FPPG ranks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"), help="download folder (default: data)")
    parser.add_argument("--holdout-seasons", type=int, default=2, help="latest draft seasons to evaluate (default: 2)")
    parser.add_argument("--cv-folds", type=int, default=5, help="max earlier chronological evaluation folds (default: 5)")
    parser.add_argument("--save-cv-results", action="store_true", help="save player-level predictions from earlier folds")
    parser.add_argument("--top-pct", type=float, default=25, help="percentage selected each draft (default: 25)")
    parser.add_argument("--min-adp", type=float, default=0, help="minimum candidate ADP for evaluation")
    parser.add_argument("--max-adp", type=float, default=float("inf"), help="maximum candidate ADP for evaluation")
    args = parser.parse_args()
    if args.holdout_seasons < 1 or args.cv_folds < 1 or not 0 < args.top_pct <= 100:
        parser.error("use at least one holdout season and CV fold, and a selection percentage greater than 0 and at most 100")
    if not 0 <= args.min_adp <= args.max_adp:
        parser.error("provide a nonnegative ADP range in ascending order")
    try:
        run_backtest(**vars(args))
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(1, f"Backtest failed: {error}\n")


if __name__ == "__main__":
    main()
