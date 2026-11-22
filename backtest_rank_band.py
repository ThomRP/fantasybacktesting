"""Audit saved model picks within a realized FPPG finish-rank band."""

import argparse
import csv
import math
from pathlib import Path
from statistics import mean

from backtest_common import write_csv
from backtest_value_benchmark import price_band, price_band_metrics


def load_results(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames or []
        selections = [name for name in fields if name.endswith("_selected")]
        required = {"draft_season", "player", "adp", "next_rank", "next_fppg", "fppg_above_adp"}
        if not required.issubset(fields) or not selections:
            raise ValueError("input must be an FPPG model results CSV with selection columns")
        rows = []
        for row in reader:
            row["adp"] = float(row["adp"])
            rank = int(row["next_rank"]) if row["next_rank"] else None
            if not math.isfinite(row["adp"]) or row["adp"] <= 0 or (rank is not None and rank <= 0):
                raise ValueError(f"invalid ADP or finish rank on line {reader.line_num}")
            row["next_rank"] = rank
            row["beat_adp"] = rank < row["adp"] if rank is not None else None
            row["adp_slots_gained"] = round(row["adp"] - rank, 3) if rank is not None else None
            row["fppg_above_adp"] = float(row["fppg_above_adp"]) if row["fppg_above_adp"] else None
            for flag in selections:
                value = (row[flag] or "").strip().lower()
                if value not in ("true", "false"):
                    raise ValueError(f"invalid {flag} flag on line {reader.line_num}")
                row[flag] = value == "true"
            rows.append(row)
    if not rows:
        raise ValueError("the results file has no player rows")
    return rows, selections


def summary_row(draft, scope, adp_band, candidates, finishers, flag):
    model = flag.removesuffix("_selected") if flag else "overall_pool"
    selected = [row for row in finishers if flag is None or row[flag]]
    selected_before = sum(flag is None or row[flag] for row in candidates)
    pool_hits = sum(row["beat_adp"] for row in finishers)
    hits = sum(row["beat_adp"] for row in selected)
    margins = [row["fppg_above_adp"] for row in selected if row["fppg_above_adp"] is not None]
    benchmark = price_band_metrics(finishers, selected)
    return {
        "draft_season": draft,
        "scope": scope,
        "adp_band": adp_band,
        "model": model,
        "pool_in_finish_band": len(finishers),
        "pool_hits": pool_hits,
        "selected_before_finish_filter": selected_before,
        "selected_in_finish_band": len(selected),
        "selected_hits": hits,
        "selected_hit_rate_pct": round(100 * hits / len(selected), 2) if selected else None,
        "pool_winners_captured_pct": round(100 * hits / pool_hits, 2) if pool_hits else None,
        "selected_avg_slots_gained": round(mean(row["adp_slots_gained"] for row in selected), 2) if selected else None,
        "selected_avg_fppg_above_adp": round(mean(margins), 3) if margins else None,
        "price_matched_hit_rate_pct": benchmark["price_matched_hit_rate_pct"],
        "price_matched_hit_rate_lift_pp": benchmark["price_matched_hit_rate_lift_pp"],
        "price_matched_fppg_lift": benchmark["price_matched_fppg_lift"],
    }


def audit_rank_band(results, min_rank, max_rank):
    rows, selections = load_results(results)
    years = sorted({row["draft_season"] for row in rows})
    finishers = [row for row in rows if row["next_rank"] is not None and min_rank <= row["next_rank"] <= max_rank]
    summary = []
    for draft in ("Overall", *years):
        candidates = [row for row in rows if draft == "Overall" or row["draft_season"] == draft]
        scopes = [
            ("all_adp", "All ADPs", candidates),
            ("adp_in_finish_band", f"{min_rank} < ADP <= {max_rank}",
             [row for row in candidates if min_rank < row["adp"] <= max_rank]),
        ]
        for band in sorted({price_band(row)[1] for row in candidates}):
            scopes.append(("adp_band", f"{25 * (band - 1)} < ADP <= {25 * band}",
                           [row for row in candidates if price_band(row)[1] == band]))
        for scope, label, pool in scopes:
            group = [row for row in pool if row["next_rank"] is not None and min_rank <= row["next_rank"] <= max_rank]
            summary.extend({"min_finish_rank": min_rank, "max_finish_rank": max_rank,
                            **summary_row(draft, scope, label, pool, group, flag)} for flag in (None, *selections))

    prefix = results.stem.removesuffix("_results")
    report = results.with_name(f"{prefix}_rank_band_summary.csv")
    picks_csv = results.with_name(f"{prefix}_rank_band_picks.csv")
    picks = [row for row in finishers if any(row[flag] for flag in selections)]
    picks.sort(key=lambda row: (row["draft_season"], row["adp"], row["player"]))
    write_csv(report, summary)
    write_csv(picks_csv, picks, fieldnames=list(rows[0]))
    print(f"Evaluating original selections with FPPG finishes {min_rank}-{max_rank} inclusive")
    for row in summary:
        if row["draft_season"] != "Overall" or row["scope"] == "adp_band":
            continue
        count = row["selected_in_finish_band"]
        label = f"{row['adp_band']} | {row['model']}"
        if count:
            print(f"{label}: {row['selected_hits']}/{count} beat ADP ({row['selected_hit_rate_pct']:.2f}%), "
                  f"{count} of {row['selected_before_finish_filter']} original picks in the finish band")
        else:
            print(f"{label}: 0 of {row['selected_before_finish_filter']} original picks in the finish band")
    print(f"Saved breakdown to {report} and picks to {picks_csv}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("data/backtest_value_results.csv"),
                        help="FPPG results CSV with model selection flags (default: data/backtest_value_results.csv)")
    parser.add_argument("--min-rank", type=int, default=50, help="best included FPPG finish rank (default: 50)")
    parser.add_argument("--max-rank", type=int, default=150, help="last included FPPG finish rank (default: 150)")
    args = parser.parse_args()
    if not 1 <= args.min_rank <= args.max_rank:
        parser.error("finish ranks must be positive and in ascending order")
    try:
        audit_rank_band(args.results, args.min_rank, args.max_rank)
    except (OSError, TypeError, ValueError, csv.Error) as error:
        parser.exit(1, f"Rank-band audit failed: {error}\n")


if __name__ == "__main__":
    main()
