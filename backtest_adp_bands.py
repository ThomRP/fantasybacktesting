"""Compare equal-sized model selections within pre-draft ADP bands."""

import argparse
import csv
import math
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from statistics import mean

from backtest_common import adp_hit_counts, write_csv
from backtest_models import select_top_picks
from backtest_rank_band import load_results
from backtest_value_benchmark import price_band_metrics


DEFAULT_ENDS = (25, 50, 75, 100, 125, 150)


def summarize(draft, band, model, pool, picks):
    hits, ranked = adp_hit_counts(picks)
    margins = [row["fppg_above_adp"] for row in picks if row["fppg_above_adp"] is not None]
    return {
        "draft_season": draft,
        "adp_band": band,
        "model": model,
        "pool_players": len(pool),
        "selected_players": len(picks),
        "selected_ranked": ranked,
        "selected_unranked": len(picks) - ranked,
        "selected_hits": hits,
        "selected_hit_rate_pct": round(100 * hits / ranked, 2) if ranked else None,
        "selected_avg_adp": round(mean(row["adp"] for row in picks), 2),
        "selected_avg_fppg_above_adp": round(mean(margins), 3) if margins else None,
        "selected_top_50": sum(row["next_rank"] <= 50 for row in picks if row["next_rank"] is not None),
        "selected_top_100": sum(row["next_rank"] <= 100 for row in picks if row["next_rank"] is not None),
        "selected_lost_25_slots": sum(
            row["adp_slots_gained"] <= -25 for row in picks if row["adp_slots_gained"] is not None
        ),
        **price_band_metrics(pool, picks),
    }


def run_backtest(results, band_ends, top_pct):
    rows, flags = load_results(results)
    models = [flag.removesuffix("_selected") for flag in flags]
    for row in rows:
        if not row.get("athlete_id"):
            raise ValueError("input must include athlete_id for each candidate")
        for model in models:
            column = f"{model}_score"
            score = float(row.get(column) or "nan")
            if not math.isfinite(score):
                raise ValueError(f"missing or invalid {column} for {row['player']} in {row['draft_season']}")
            row[column] = score

    bands = defaultdict(list)
    for row in rows:
        bands[bisect_left(band_ends, row["adp"])].append(row)

    summary, picks_csv = [], []
    for index, pool in sorted(bands.items()):
        lower = band_ends[index - 1] if index else 0
        band = f"{lower} < ADP <= {band_ends[index]}" if index < len(band_ends) else f"ADP > {lower}"
        years = sorted({row["draft_season"] for row in pool})
        selections = {"overall_pool": pool}
        for model in models:
            scores = [row[f"{model}_score"] for row in pool]
            indices = select_top_picks(pool, scores, top_pct)
            selections[model] = [pool[i] for i in sorted(indices)]
            for row in selections[model]:
                picks_csv.append({
                    "draft_season": row["draft_season"],
                    "adp_band": band,
                    "model": model,
                    "player": row["player"],
                    "athlete_id": row["athlete_id"],
                    "model_score": row[f"{model}_score"],
                    **{key: row.get(key) for key in (
                        "adp", "next_rank", "next_games", "next_fppg", "beat_adp",
                        "adp_slots_gained", "fppg_above_adp",
                    )},
                })
        for draft in ("Overall", *years):
            candidates = pool if draft == "Overall" else [row for row in pool if row["draft_season"] == draft]
            for model, selected in selections.items():
                picks = selected if draft == "Overall" else [row for row in selected if row["draft_season"] == draft]
                summary.append(summarize(draft, band, model, candidates, picks))

    prefix = results.stem.removesuffix("_results")
    summary_path = results.with_name(f"{prefix}_adp_band_summary.csv")
    picks_path = results.with_name(f"{prefix}_adp_band_picks.csv")
    write_csv(summary_path, summary)
    write_csv(picks_path, picks_csv)
    print(f"Selected the top {top_pct:g}% by model score per draft and ADP band")
    for row in summary:
        if row["draft_season"] != "Overall":
            continue
        lift = row["price_matched_fppg_lift"]
        lift_text = f"{lift:+.2f} FPPG vs same-price picks" if lift is not None else "no ranked FPPG comparison"
        print(f"{row['adp_band']} | {row['model']}: {row['selected_hits']}/{row['selected_ranked']} beat ADP, "
              f"{row['selected_unranked']} unranked, {lift_text}")
    print(f"Saved breakdown to {summary_path} and picks to {picks_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("data/backtest_value_results.csv"),
                        help="player-level model results (default: data/backtest_value_results.csv)")
    parser.add_argument("--band-ends", nargs="+", type=int, default=DEFAULT_ENDS,
                        help="inclusive upper ADP for each band (default: 25 50 75 100 125 150)")
    parser.add_argument("--top-pct", type=float, default=25,
                        help="percentage selected within each band per draft (default: 25)")
    args = parser.parse_args()
    if min(args.band_ends) < 1 or sorted(set(args.band_ends)) != list(args.band_ends):
        parser.error("--band-ends must be unique positive ADPs in ascending order")
    if not 0 < args.top_pct <= 100:
        parser.error("--top-pct must be greater than 0 and at most 100")
    try:
        run_backtest(**vars(args))
    except (OSError, TypeError, ValueError, csv.Error) as error:
        parser.exit(1, f"ADP-band backtest failed: {error}\n")


if __name__ == "__main__":
    main()
