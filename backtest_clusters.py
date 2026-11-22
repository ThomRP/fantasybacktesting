"""Find pre-draft player profiles with K-means and describe them with PCA."""

import argparse
import csv
from pathlib import Path
from statistics import mean

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from backtest_common import draft_outcomes, format_value_metrics, load_birth_dates, season_pairs, value_metrics, write_csv

FEATURES = (
    "adp",
    "age_at_prior_break",
    "prior_games",
    "prior_fppg",
    "prior_post_break_fppg_change",
    "prior_start_pct",
)


def draft_rows(data):
    for year, prior, following, prior_stats, following_stats in season_pairs(data):
        all_star_date, prior_players = prior_stats
        _, following_players = following_stats
        outcomes, total_picks = draft_outcomes(following, prior_players, following_players, match_prior_only=True)
        print(f"{prior.name} to {following.name}: matched {len(outcomes)}/{total_picks} ADP entries to prior-season players")
        if not outcomes:
            continue

        try:
            births = load_birth_dates(data, year + 1)
        except (OSError, ValueError, KeyError) as error:
            print(f"{prior.name}: birth dates unavailable ({error})")
            births = {}

        for name, athlete_id, outcome in outcomes:
            player = prior_players[athlete_id]
            age = None
            birth = births.get(athlete_id)
            if birth is not None:
                age = all_star_date.year - birth.year
                if (all_star_date.month, all_star_date.day) < (birth.month, birth.day):
                    age -= 1
            fppg_change = None
            if player.pre_games and player.post_games:
                fppg_change = player.post_points / player.post_games - player.pre_points / player.pre_games
            start_pct = None
            if not player.games_without_start_status:
                start_pct = 100 * player.starts / player.total_games

            yield {
                "prior_season": prior.name,
                "draft_season": following.name,
                "player": name,
                "athlete_id": athlete_id,
                "age_at_prior_break": age,
                "prior_games": player.total_games,
                "prior_fppg": player.total_points / player.total_games,
                "prior_post_break_fppg_change": fppg_change,
                "prior_start_pct": start_pct,
                **outcome,
            }


def feature_matrix(rows, features):
    return [[row[name] if row[name] is not None else float("nan") for name in features] for row in rows]


def summary_row(cluster, train_count, evaluated):
    hits = sum(row["beat_adp"] for row in evaluated)
    count = len(evaluated)
    return {
        "cluster": cluster,
        "training_players": train_count,
        "holdout_players": count,
        "holdout_beat_adp": hits,
        "holdout_beat_adp_pct": round(100 * hits / count, 1) if count else None,
        "holdout_avg_adp": round(mean(row["adp"] for row in evaluated), 1) if count else None,
        **value_metrics(evaluated),
    }


def run_backtest(data, clusters=4, holdout_seasons=2):
    rows = list(draft_rows(data))
    seasons = sorted({row["draft_season"] for row in rows})
    if len(seasons) <= holdout_seasons:
        raise ValueError("need more completed draft seasons than --holdout-seasons. Each draft needs its prior season and adp.csv")
    holdout = set(seasons[-holdout_seasons:])
    training = [row for row in rows if row["draft_season"] not in holdout]
    evaluated = [row for row in rows if row["draft_season"] in holdout]

    # Use only features with observed variation in the training seasons.
    features = [name for name in FEATURES if len({row[name] for row in training if row[name] is not None}) > 1]
    if len(features) < 2:
        raise ValueError("need at least two varying training features for PCA")
    omitted = [name for name in FEATURES if name not in features]
    if omitted:
        print(f"Features without training variation: {', '.join(omitted)}")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_x = scaler.fit_transform(imputer.fit_transform(feature_matrix(training, features)))
    test_x = scaler.transform(imputer.transform(feature_matrix(evaluated, features)))
    if len({tuple(row) for row in train_x}) < clusters:
        raise ValueError("fewer distinct training profiles than clusters. Try a smaller --clusters value")

    model = KMeans(n_clusters=clusters, random_state=42, n_init=20).fit(train_x)
    pca = PCA(n_components=2, svd_solver="full").fit(train_x)
    test_labels = model.predict(test_x)
    for row, cluster, (pc1, pc2) in zip(evaluated, test_labels, pca.transform(test_x)):
        row.update(cluster=int(cluster) + 1, pc1=round(float(pc1), 4), pc2=round(float(pc2), 4))

    centers = scaler.inverse_transform(model.cluster_centers_)
    summary = [{**summary_row("Overall", len(training), evaluated), **{f"center_{name}": None for name in features}}]
    for index, center in enumerate(centers):
        group = [row for row in evaluated if row["cluster"] == index + 1]
        summary.append({
            **summary_row(index + 1, int(sum(model.labels_ == index)), group),
            **{f"center_{name}": round(float(value), 3) for name, value in zip(features, center)},
        })
    components = [
        {
            "component": f"PC{index}",
            "training_variance_pct": round(100 * float(variance), 2),
            **{name: round(float(weight), 4) for name, weight in zip(features, weights)},
        }
        for index, (weights, variance) in enumerate(zip(pca.components_, pca.explained_variance_ratio_), 1)
    ]

    write_csv(data / "backtest_cluster_results.csv", evaluated)
    write_csv(data / "backtest_cluster_summary.csv", summary)
    write_csv(data / "backtest_cluster_pca.csv", components)
    print(f"Training drafts: {', '.join(seasons[:-holdout_seasons])}")
    print(f"Holdout drafts: {', '.join(sorted(holdout))}")
    for row in summary:
        label = "Overall" if row["cluster"] == "Overall" else f"Cluster {row['cluster']}"
        count = row["holdout_players"]
        if count:
            print(f"{label}: {row['holdout_beat_adp']}/{count} beat ADP ({row['holdout_beat_adp_pct']:.1f}%), "
                  f"average ADP {row['holdout_avg_adp']:.1f}, {format_value_metrics(row)}")
        else:
            print(f"{label}: no holdout players")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"), help="download folder (default: data)")
    parser.add_argument("--clusters", type=int, default=4, help="number of K-means clusters (default: 4)")
    parser.add_argument("--holdout-seasons", type=int, default=2, help="latest draft seasons to evaluate (default: 2)")
    args = parser.parse_args()
    if args.clusters < 2 or args.holdout_seasons < 1:
        parser.error("use at least two clusters and at least one holdout season")
    try:
        run_backtest(args.data, args.clusters, args.holdout_seasons)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(1, f"Backtest failed: {error}\n")


if __name__ == "__main__":
    main()
