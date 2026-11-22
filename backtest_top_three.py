"""Compare three-player draft shortlists based on FPPG or projected value."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from backtest_adp_bands import summarize
from backtest_common import adp_slot_points, read_csv, write_csv
from backtest_fppg import MODELS


MIN_ADP, MAX_ADP = 50, 75
ADP_BAND = f"{MIN_ADP} < ADP <= {MAX_ADP}"
PICKS_PER_DRAFT = 3


def logistic_scores(data):
    scores = {}
    for split, name in (("holdout", "backtest_value_results.csv"), ("cv", "backtest_value_cv_results.csv")):
        path = data / name
        if path.is_file():
            for row in read_csv(path):
                score = float(row["logistic_base_score"])
                if not math.isfinite(score):
                    raise ValueError(f"{path}: invalid logistic score for {row['player']}")
                scores[(split, row["draft_season"], row["athlete_id"])] = (float(row["adp"]), score)
    return scores


def draft_candidates(rows):
    # The forecast export includes all prior-season players, including those
    # with no appearances in the evaluated season.
    prior_leaderboard = sorted((float(row["prior_fppg"]) for row in rows), reverse=True)
    pool = []
    for row in rows:
        adp = float(row["adp"]) if row["adp"] else None
        if adp is None or not MIN_ADP < adp <= MAX_ADP:
            continue
        rank = int(row["next_rank"]) if row["next_rank"] else None
        pool.append({
            **row,
            "adp": adp,
            "next_rank": rank,
            "next_games": int(row["next_games"]),
            "fppg_above_adp": float(row["fppg_above_adp"]) if row["fppg_above_adp"] else None,
            "adp_slots_gained": adp - rank if rank is not None else None,
            "beat_adp": rank < adp if rank is not None else None,
            "prior_slot_fppg": adp_slot_points(prior_leaderboard, adp),
        })
    return pool


def draft_scores(pool, scores):
    forecasts = {model: [float(row[f"{model}_prediction"]) for row in pool] for model in MODELS}
    yield from forecasts.items()
    for model in ("ridge_fppg", "ridge_box_score"):
        yield f"{model}_value", [prediction - row["prior_slot_fppg"]
                                 for prediction, row in zip(forecasts[model], pool)]

    logistic = [scores.get((row["split"], row["draft_season"], row["athlete_id"])) for row in pool]
    if all(score is not None and score[0] == row["adp"] for score, row in zip(logistic, pool)):
        yield "logistic_base", [score[1] for score in logistic]
    elif any(score is not None for score in logistic):
        print(f"{pool[0]['draft_season']}: skipping logistic comparison with incomplete or different ADP entries")


def summarize_targets(split, season, model, pools, shortlists):
    pool = [row for draft in pools for row in draft]
    picks = [row for shortlist in shortlists for row in shortlist]
    thirds = [shortlist[PICKS_PER_DRAFT - 1] for shortlist in shortlists if len(shortlist) == PICKS_PER_DRAFT]
    third_margins = [row["fppg_above_adp"] for row in thirds if row["fppg_above_adp"] is not None]
    return {
        "split": split,
        **summarize(season, ADP_BAND, model, pool, picks),
        "drafts_evaluated": len(pools),
        "third_picks": len(thirds),
        "third_pick_hits": sum(row["beat_adp"] is True for row in thirds),
        "third_pick_unranked": sum(row["next_rank"] is None for row in thirds),
        "third_pick_avg_fppg_above_adp": round(mean(third_margins), 3) if third_margins else None,
    }


def run_backtest(data):
    by_draft = defaultdict(list)
    for row in read_csv(data / "backtest_fppg_predictions.csv"):
        by_draft[row["split"], row["draft_season"]].append(row)
    logistic = logistic_scores(data)

    results, pick_details, summaries = defaultdict(list), [], []
    for (split, season), rows in sorted(by_draft.items()):
        pool = draft_candidates(rows)
        if not pool:
            continue
        for model, scores in draft_scores(pool, logistic):
            if any(not math.isfinite(score) for score in scores):
                raise ValueError(f"{season}: invalid score for {model}")
            ordered = sorted(zip(pool, scores), key=lambda item: (-item[1], item[0]["adp"], item[0]["athlete_id"]))
            shortlist = ordered[:PICKS_PER_DRAFT]
            picks = [row for row, _ in shortlist]
            results[split, model].append((pool, picks))
            summaries.append(summarize_targets(split, season, model, [pool], [picks]))
            for pick_number, (row, score) in enumerate(shortlist, start=1):
                pick_details.append({
                    "split": split,
                    "draft_season": season,
                    "model": model,
                    "pick_number": pick_number,
                    "player": row["player"],
                    "athlete_id": row["athlete_id"],
                    "adp": row["adp"],
                    "prior_slot_fppg": round(row["prior_slot_fppg"], 3),
                    "selection_score": round(score, 6),
                    "next_rank": row["next_rank"],
                    "next_games": row["next_games"],
                    "next_fppg": row["next_fppg"],
                    "beat_adp": row["beat_adp"],
                    "fppg_above_adp": row["fppg_above_adp"],
                })

    if not results:
        raise ValueError(f"no forecast candidates with {ADP_BAND}; generate predictions with ADP files present")
    overall = []
    for (split, model), drafts in results.items():
        pools, shortlists = zip(*drafts)
        overall.append(summarize_targets(split, "Overall", model, pools, shortlists))
    summary_path = data / "backtest_top_three_summary.csv"
    picks_path = data / "backtest_top_three_picks.csv"
    write_csv(summary_path, overall + summaries)
    write_csv(picks_path, pick_details)
    for row in overall:
        lift = row["price_matched_fppg_lift"]
        lift_text = f"{lift:+.2f} FPPG vs same-price picks" if lift is not None else "no ranked FPPG comparison"
        print(f"{row['split']} | {row['model']} (drafts evaluated: {row['drafts_evaluated']}): "
              f"{row['selected_hits']}/{row['selected_players']} targets beat ADP, "
              f"{row['selected_unranked']} unranked, {lift_text}. "
              f"Third pick: {row['third_pick_hits']}/{row['third_picks']} beat ADP")
    print(f"Saved report to {summary_path} and picks to {picks_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"), help="folder with forecast and value model results")
    args = parser.parse_args()
    try:
        run_backtest(args.data)
    except (OSError, KeyError, TypeError, ValueError, csv.Error) as error:
        parser.exit(1, f"Top-three backtest failed: {error}\n")


if __name__ == "__main__":
    main()
