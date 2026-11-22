"""Test short draft target lists and screen late targets for availability."""

import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from backtest_adp_bands import summarize
from backtest_common import write_csv
from backtest_rank_band import load_results


TESTS = ((50, 75, "logistic_base"), (125, 150, "value_base"))


def player_key(row):
    return row["draft_season"], row["athlete_id"]


def read_availability(path):
    """Use the latest documented status available at each draft cutoff."""
    drafts, latest = {}, {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = {"draft_season", "athlete_id", "reported_at", "draft_at", "status"}
        if not columns.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected columns {', '.join(sorted(columns))}")
        for row in reader:
            draft, athlete = player_key(row)
            reported = datetime.fromisoformat(row["reported_at"].replace("Z", "+00:00"))
            cutoff = datetime.fromisoformat(row["draft_at"].replace("Z", "+00:00"))
            status = row["status"].strip().lower()
            if not draft or not athlete or status not in {"out", "available"}:
                raise ValueError(f"{path}:{reader.line_num}: expected season, athlete ID and status out or available")
            if reported.tzinfo is None or cutoff.tzinfo is None:
                raise ValueError(f"{path}:{reader.line_num}: timestamps need a UTC offset")
            if drafts.setdefault(draft, cutoff) != cutoff:
                raise ValueError(f"{path}:{reader.line_num}: conflicting draft cutoffs for {draft}")
            if reported > cutoff:
                continue
            previous = latest.get((draft, athlete))
            if previous is None or reported > previous[0]:
                latest[(draft, athlete)] = reported, status
            elif reported == previous[0] and status != previous[1]:
                raise ValueError(f"{path}:{reader.line_num}: conflicting statuses for {draft}, athlete {athlete}")
    return {key: status for key, (_, status) in latest.items()}


def rank_candidates(candidates, score_column):
    for row in candidates:
        raw_score = row.get(score_column)
        score = float(raw_score) if raw_score not in (None, "") else math.nan
        if not row.get("athlete_id") or not math.isfinite(score):
            raise ValueError(f"missing athlete_id or {score_column} for {row['player']} in {row['draft_season']}")
        row[score_column] = score
        row["prior_games"] = int(row["prior_games"])
        row["next_games"] = int(row["next_games"])
    return sorted(candidates, key=lambda row: (-row[score_column], row["adp"], row["athlete_id"]))


def evaluate_band(pool, band, model, minimum_games, reports):
    by_draft = defaultdict(list)
    for row in pool:
        by_draft[row["draft_season"]].append(row)

    score_column = f"{model}_score"
    strategies = defaultdict(list)
    quarter_keys = set()
    for draft, candidates in sorted(by_draft.items()):
        ranked = rank_candidates(candidates, score_column)
        quota = math.ceil(len(ranked) / 4)
        quarter = ranked[:quota]
        quarter_keys.update(map(player_key, quarter))
        strategies["top_quarter"].append((draft, quota, quarter))
        if model == "logistic_base":
            short_quota = min(2, len(ranked))
            strategies["top_two"].append((draft, short_quota, ranked[:short_quota]))
            prior_fppg_ranked = rank_candidates(candidates, "prior_fppg")
            strategies["prior_fppg_top_two"].append((draft, short_quota, prior_fppg_ranked[:short_quota]))
        else:
            prior_games = [row for row in ranked if row["prior_games"] >= minimum_games]
            strategies["prior_games_screen"].append((draft, quota, prior_games[:quota]))
            if reports is not None:
                eligible = [row for row in ranked if reports.get(player_key(row)) != "out"]
                strategies["draft_availability_screen"].append((draft, quota, eligible[:quota]))

    summary, pick_details = [], []
    for strategy, seasons in strategies.items():
        strategy_model = "prior_fppg" if strategy == "prior_fppg_top_two" else model
        strategy_score = "prior_fppg" if strategy == "prior_fppg_top_two" else score_column
        all_picks = [row for _, _, picks in seasons for row in picks]
        for draft, quota, picks in [("Overall", sum(quota for _, quota, _ in seasons), all_picks), *seasons]:
            candidates = pool if draft == "Overall" else by_draft[draft]
            metrics = summarize(draft, band, strategy_model, candidates, picks)
            n_reported = sum(player_key(row) in reports for row in candidates) if reports is not None else None
            summary.append({
                **metrics,
                "strategy": strategy,
                "target_picks": quota,
                "unfilled_picks": quota - len(picks),
                "picks_outside_top_quarter": sum(player_key(row) not in quarter_keys for row in picks),
                "selected_1_to_10_games": sum(1 <= row["next_games"] <= 10 for row in picks),
                "hit_rate_per_target_pct": round(100 * metrics["selected_hits"] / quota, 2),
                "prior_games_minimum": minimum_games if strategy == "prior_games_screen" else None,
                "candidates_with_draft_status": n_reported,
            })
        for row in all_picks:
            key = player_key(row)
            pick_details.append({
                "draft_season": row["draft_season"],
                "adp_band": band,
                "model": strategy_model,
                "strategy": strategy,
                "player": row["player"],
                "athlete_id": row["athlete_id"],
                "adp": row["adp"],
                "model_score": row[strategy_score],
                "in_top_quarter": key in quarter_keys,
                "prior_games": row["prior_games"],
                "draft_status": reports.get(key, "unknown") if reports is not None else None,
                "next_games": row["next_games"],
                "next_rank": row["next_rank"],
                "beat_adp": row["beat_adp"],
                "fppg_above_adp": row["fppg_above_adp"],
            })
    return summary, pick_details


def run_backtest(results, min_prior_games, availability=None):
    rows, _ = load_results(results)
    reports = read_availability(availability) if availability is not None else None
    summary, pick_details = [], []
    for lower, upper, model in TESTS:
        band = f"{lower} < ADP <= {upper}"
        pool = [row for row in rows if lower < row["adp"] <= upper]
        if not pool:
            raise ValueError(f"no candidates for {band}; use results generated without ADP limits")
        band_summary, band_picks = evaluate_band(pool, band, model, min_prior_games, reports)
        summary.extend(band_summary)
        pick_details.extend(band_picks)

    prefix = results.stem.removesuffix("_results")
    summary_path = results.with_name(f"{prefix}_draft_targets_summary.csv")
    picks_path = results.with_name(f"{prefix}_draft_targets_picks.csv")
    write_csv(summary_path, summary)
    write_csv(picks_path, pick_details)
    for row in summary:
        if row["draft_season"] != "Overall":
            continue
        lift = row["price_matched_fppg_lift"]
        lift_text = f"{lift:+.2f} FPPG vs same-price picks" if lift is not None else "no ranked FPPG comparison"
        print(f"{row['adp_band']} | {row['strategy']}: {row['selected_hits']}/{row['target_picks']} targets beat ADP, "
              f"{row['selected_unranked']} unranked, {row['unfilled_picks']} unfilled, {lift_text}")
    print(f"Saved breakdown to {summary_path} and picks to {picks_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("data/backtest_value_results.csv"),
                        help="full value model predictions (default: data/backtest_value_results.csv)")
    parser.add_argument("--min-prior-games", type=int, default=50,
                        help="prior-season games required by the participation screen (default: 50)")
    parser.add_argument("--availability", type=Path, help="optional CSV of dated draft-time player statuses")
    args = parser.parse_args()
    if args.min_prior_games < 0:
        parser.error("--min-prior-games must be nonnegative")
    try:
        run_backtest(**vars(args))
    except (OSError, TypeError, ValueError, KeyError, csv.Error) as error:
        parser.exit(1, f"Draft-target backtest failed: {error}\n")


if __name__ == "__main__":
    main()
