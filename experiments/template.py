"""A small point-in-time strategy experiment, without a simulator or UI.

Run ``python -m experiments.template --demo`` for a synthetic offline example.
Run ``python -m experiments.template --season 2026`` against processed data, or
pass ``--cohort signals.csv`` with season/canonical_player_id signal keys.
The fixed example signal is prior-season PPG in the top quartile. It uses no
upcoming-season outcomes, position metadata, or holdout-driven tuning.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import PROCESSED_DIR, SCORING, SEASON_OPENERS, SEASONS
from src.grader import evaluate_cohort
from src.normalize import calculate_fantasy_points, filter_game_logs


def make_cohort(game_logs: pd.DataFrame, adp: pd.DataFrame, season: int) -> pd.DataFrame:
    """Select from prior-season data available strictly before season opening."""
    if season not in SEASONS:
        raise ValueError("season is outside the configured research window")
    available = filter_game_logs(game_logs, as_of=SEASON_OPENERS[season])
    prior = available.loc[available["season"].eq(season - 1)]
    keys = ["season", "canonical_player_id"]
    if prior.empty:
        return pd.DataFrame(columns=keys)
    ppg = prior.groupby("canonical_player_id")["fantasy_points"].mean()
    selected = ppg[ppg >= ppg.quantile(0.75)].rename("prior_ppg").reset_index()
    selected["season"] = season
    current_market = adp.loc[adp["season"].eq(season), keys + ["adp_rank"]]
    # Keep missing market matches visible as grader exclusions; do not pretend
    # that archive coverage is complete or silently grade only a lucky subset.
    signals = selected.merge(current_market, on=keys, how="left", validate="one_to_one")
    signals = signals[signals["adp_rank"].isna() | signals["adp_rank"].between(1, 200)]
    return signals[keys].reset_index(drop=True)


def demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deliberately invented miniature box scores; never written to data/."""
    rows, prices = [], []
    for season in (2021, 2022, 2023):
        for i, rank in enumerate((1, 40, 80, 160), start=1):
            player = f"demo:{i}"
            prices.append({"season": season, "canonical_player_id": player,
                           "adp": float(rank), "adp_rank": rank, "position_rank": i})
            for day in (3, 8):
                stats = dict.fromkeys(SCORING, 0.0)
                stats.update(pts=32.0 - 4 * i, fgm=12.0 - i, fga=22.0 - i,
                             fg3m=3.0, reb=10.0 - i, ast=8.0 - i,
                             stl=1.0, blk=1.0, tov=2.0, ftm=3.0, fta=4.0)
                rows.append({"season": season, "canonical_player_id": player,
                             "game_date": pd.Timestamp(season, 1, day).date(),
                             "minutes_or_snaps": 32.0, **stats})
    games = pd.DataFrame(rows)
    games["fantasy_points"] = calculate_fantasy_points(games)
    return games, pd.DataFrame(prices)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="use explicitly synthetic offline data")
    parser.add_argument("--season", type=int, choices=SEASONS, default=None)
    parser.add_argument("--cohort", type=Path, help="CSV with season,canonical_player_id signal keys")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    if args.demo:
        print("SYNTHETIC DEMO: invented data, not evidence of strategy performance.")
        games, adp = demo_data()
        season = args.season or 2023
    else:
        print("REAL DATA: exclusions and absent training history are reported explicitly.")
        games = pd.read_parquet(args.processed_dir / "game_logs.parquet")
        adp = pd.read_parquet(args.processed_dir / "adp.parquet")
        coverage = json.loads((args.processed_dir / "coverage.json").read_text())
        # Incomplete seasons cannot supply complete prior-season PPG features.
        games = games[games["season"].isin(coverage["completed_seasons"])]
        season = args.season or max(SEASONS)
    signals = (pd.read_csv(args.cohort, dtype={"canonical_player_id": str})
               if args.cohort else make_cohort(games, adp, season))
    print(f"Cohort: {len(signals)} season/player signals")
    if signals.empty:
        print("No eligible signals; check prior-season coverage and the selected season.")
    if args.demo:
        report = evaluate_cohort(signals, game_logs=games, adp=adp)
    else:
        # Let the grader validate disk coverage/scoring itself; no bypass through
        # the direct-DataFrame API is needed for real processed data.
        report = evaluate_cohort(signals, processed_dir=args.processed_dir)
    print(json.dumps(_json_safe(report["summary"]), indent=2, allow_nan=False))
    for key in ("era_breakdown", "split_breakdown"):
        if not report[key].empty:
            print(report[key].to_string(index=False))
    if not report["exclusions"].empty:
        print("Excluded signals (first 20):")
        print(report["exclusions"][["season", "canonical_player_id", "status"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
