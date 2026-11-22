"""Benchmark draft value at the same season and draft price mix."""

from collections import Counter, defaultdict
from math import ceil, fsum
from statistics import mean

from backtest_common import adp_hit_counts


def price_band(row):
    return row["draft_season"], ceil(row["adp"] / 25)


def ranked_fppg_margins(rows):
    return [row["fppg_above_adp"] for row in rows
            if row["beat_adp"] is not None and row["fppg_above_adp"] is not None]


def price_band_metrics(evaluated, selected):
    """Compare actual results with expected totals from drawing the same price mix.

    The benchmark draws from the full eligible pool within each draft and price
    band. All selections determine the draw counts. Hit rates and FPPG margins
    divide expected outcome totals by their expected scored counts.
    """
    selected = list(selected)
    draws_by_band = Counter(price_band(row) for row in selected)
    pool_by_band = defaultdict(list)
    for row in evaluated:
        pool_by_band[price_band(row)].append(row)

    expected_hits = expected_ranked = expected_scored = expected_margin_sum = 0.0
    for band, draw_count in sorted(draws_by_band.items()):
        pool = pool_by_band[band]
        if draw_count > len(pool):
            raise ValueError("price-matched selections must come from the eligible draft pool")
        draw_fraction = draw_count / len(pool)
        hits, ranked = adp_hit_counts(pool)
        margins = ranked_fppg_margins(pool)
        expected_hits += draw_fraction * hits
        expected_ranked += draw_fraction * ranked
        expected_scored += draw_fraction * len(margins)
        expected_margin_sum += draw_fraction * fsum(margins)

    matched_rate = 100 * expected_hits / expected_ranked if expected_ranked else None
    matched_margin = expected_margin_sum / expected_scored if expected_scored else None
    selected_hits, selected_ranked = adp_hit_counts(selected)
    selected_margins = ranked_fppg_margins(selected)
    hit_rate_lift = (100 * selected_hits / selected_ranked - matched_rate
                     if selected_ranked and matched_rate is not None else None)
    fppg_lift = (mean(selected_margins) - matched_margin
                 if selected_margins and matched_margin is not None else None)

    return {
        "price_matched_expected_ranked": round(expected_ranked, 2),
        "price_matched_expected_unranked": round(len(selected) - expected_ranked, 2),
        "price_matched_expected_hits": round(expected_hits, 2),
        "price_matched_hit_rate_pct": round(matched_rate, 2) if matched_rate is not None else None,
        "price_matched_avg_fppg_above_adp": round(matched_margin, 2) if matched_margin is not None else None,
        "price_matched_hit_rate_lift_pp": round(hit_rate_lift, 2) if hit_rate_lift is not None else None,
        "price_matched_fppg_lift": round(fppg_lift, 2) if fppg_lift is not None else None,
    }
