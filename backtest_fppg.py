"""Forecast next-season fantasy points per game from prior box scores."""

import argparse
import csv
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from backtest_clusters import feature_matrix
from backtest_common import BOX_SCORE_STATS, adp_hit_counts, draft_outcomes, season_pairs, write_csv
from backtest_value_benchmark import price_band_metrics


FPPG_FEATURES = ("prior_fppg", "prior_games", "prior_minutes")
BOX_FEATURES = FPPG_FEATURES + tuple(f"prior_{stat}" for stat in BOX_SCORE_STATS if stat != "minutes")
MODELS = ("carry_forward", "ridge_fppg", "ridge_box_score")


def player_seasons(data):
    for _, prior, following, (_, earlier), (_, later) in season_pairs(
        data, split_at_all_star=False, include_box_scores=True, require_adp=False,
    ):
        draft = {}
        if (following / "adp.csv").is_file():
            try:
                outcomes, _ = draft_outcomes(following, earlier, later, match_prior_only=True, rank_per_game=True)
            except (OSError, ValueError, csv.Error) as error:
                print(f"{following.name}: skipping ADP comparison ({error})")
            else:
                draft = {athlete_id: outcome for _, athlete_id, outcome in outcomes}

        for athlete_id, player in earlier.items():
            after = later.get(athlete_id)
            outcome = draft.get(athlete_id, {})
            yield {
                "prior_season": prior.name,
                "draft_season": following.name,
                "athlete_id": athlete_id,
                "player": min(player.names),
                "prior_games": player.total_games,
                "prior_fppg": player.total_points / player.total_games,
                **{f"prior_{stat}": total / player.total_games if total is not None else None
                   for stat, total in player.box_score_totals.items()},
                "next_games": after.total_games if after else 0,
                "next_fppg": after.total_points / after.total_games if after else None,
                "adp": outcome.get("adp"),
                "next_rank": outcome.get("next_rank"),
                "beat_adp": outcome.get("beat_adp"),
                "fppg_above_adp": outcome.get("fppg_above_adp"),
            }


def forecast(training, by_season):
    targets = [row["next_fppg"] for row in training]
    evaluated = [row for rows in by_season.values() for row in rows]
    for row in evaluated:
        row["carry_forward_prediction"] = row["prior_fppg"]
    for model, features in (("ridge_fppg", FPPG_FEATURES), ("ridge_box_score", BOX_FEATURES)):
        estimator = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            Ridge(alpha=10.0),
        )
        estimator.fit(feature_matrix(training, features), targets)
        predictions = estimator.predict(feature_matrix(evaluated, features))
        for row, prediction in zip(evaluated, predictions):
            row[f"{model}_prediction"] = float(prediction)

    for rows in by_season.values():
        pool = [row for row in rows if row["adp"] is not None and 50 < row["adp"] <= 75]
        for model in MODELS:
            ordered = sorted(pool, key=lambda row: (-row[f"{model}_prediction"], row["adp"], row["athlete_id"]))
            picks = {row["athlete_id"] for row in ordered[:2]}
            for row in rows:
                row[f"{model}_draft_target"] = row["athlete_id"] in picks


def summarize(rows, split, draft_season):
    pool = [row for row in rows if row["adp"] is not None and 50 < row["adp"] <= 75]
    through = {row["training_through"] for row in rows}
    for model in MODELS:
        errors = [row[f"{model}_prediction"] - row["next_fppg"] for row in rows if row["next_fppg"] is not None]
        selected = [row for row in pool if row[f"{model}_draft_target"]]
        hits, ranked = adp_hit_counts(selected)
        yield {
            "split": split,
            "draft_season": draft_season,
            "training_through": next(iter(through)) if len(through) == 1 else None,
            "model": model,
            "players_forecast": len(rows),
            "players_scored": len(errors),
            "players_without_appearances": len(rows) - len(errors),
            "mae_fppg": round(mean(map(abs, errors)), 3) if errors else None,
            "rmse_fppg": round(sqrt(mean(error**2 for error in errors)), 3) if errors else None,
            "adp_51_to_75_candidates": len(pool),
            "draft_targets": len(selected),
            "draft_hits": hits,
            "draft_unranked": len(selected) - ranked,
            "draft_price_matched_fppg_lift": price_band_metrics(pool, selected)["price_matched_fppg_lift"],
        }


def run_backtest(data, holdout_seasons, cv_folds):
    by_season = defaultdict(list)
    for row in player_seasons(data):
        by_season[row["draft_season"]].append(row)
    seasons = sorted(by_season)
    if len(seasons) <= holdout_seasons:
        raise ValueError("need more complete season pairs than --holdout-seasons")
    development, holdout = seasons[:-holdout_seasons], seasons[-holdout_seasons:]

    training = [row for season in development for row in by_season[season] if row["next_fppg"] is not None]
    if not training:
        raise ValueError("no training players with appearances in consecutive seasons")
    cv = development[1:][-cv_folds:]
    for season in cv:
        earlier = [row for row in training if row["draft_season"] < season]
        if not earlier:
            continue
        forecast(earlier, {season: by_season[season]})
        for row in by_season[season]:
            row.update(split="cv", training_through=earlier[-1]["draft_season"])

    forecast(training, {season: by_season[season] for season in holdout})
    for season in holdout:
        for row in by_season[season]:
            row.update(split="holdout", training_through=training[-1]["draft_season"])

    predictions, summary = [], []
    for split, drafts in (("cv", cv), ("holdout", holdout)):
        rows = [row for season in drafts for row in by_season[season] if row.get("split") == split]
        if not rows:
            continue
        predictions.extend(rows)
        summary.extend(summarize(rows, split, "Overall"))
        for season in sorted({row["draft_season"] for row in rows}):
            summary.extend(summarize(by_season[season], split, season))

    summary_path = data / "backtest_fppg_summary.csv"
    predictions_path = data / "backtest_fppg_predictions.csv"
    write_csv(summary_path, summary)
    write_csv(predictions_path, predictions)
    print(f"Training seasons ended through {training[-1]['draft_season']}")
    print(f"Holdout seasons: {', '.join(holdout)}")
    for row in summary:
        if row["draft_season"] != "Overall":
            continue
        print(f"{row['split']} | {row['model']}: {row['mae_fppg']} MAE, {row['rmse_fppg']} RMSE "
              f"across {row['players_scored']} player-seasons")
        if row["draft_targets"]:
            print(f"  ADP 51–75: {row['draft_hits']}/{row['draft_targets']} targets beat ADP, "
                  f"{row['draft_unranked']} unranked, {row['draft_price_matched_fppg_lift']} FPPG lift")
    print(f"Saved report to {summary_path} and predictions to {predictions_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"), help="download folder (default: data)")
    parser.add_argument("--holdout-seasons", type=int, default=2, help="latest outcomes held out (default: 2)")
    parser.add_argument("--cv-folds", type=int, default=5, help="maximum earlier chronological folds (default: 5)")
    args = parser.parse_args()
    if args.holdout_seasons < 1 or args.cv_folds < 1:
        parser.error("use at least one holdout season and one CV fold")
    try:
        run_backtest(**vars(args))
    except (OSError, KeyError, ValueError, csv.Error) as error:
        parser.exit(1, f"FPPG backtest failed: {error}\n")


if __name__ == "__main__":
    main()
