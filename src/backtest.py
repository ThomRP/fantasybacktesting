"""Did last season's post-All-Star surges disappoint at the next draft's ADP?

Signals use only season S. Outcomes and draft prices belong to S+1. Rank loss
(actual fantasy finish rank minus ADP rank) is positive for underperformance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ALL_STAR_DATES, SEASON_OPENERS, PROCESSED_DIR, SCORING, SEASONS

KEYS = ["season", "canonical_player_id"]


def _validate(frame: pd.DataFrame, *, prices: bool = False) -> pd.DataFrame:
    required = set(KEYS + (["adp", "adp_rank", "available_at", "timing_status"] if prices else ["game_date", "fantasy_points"]))
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing {'ADP' if prices else 'game log'} columns: {sorted(missing)}")
    frame = frame.copy()
    years = pd.to_numeric(frame.season, errors="raise")
    if years.isna().any() or not np.isfinite(years).all() or (years % 1 != 0).any() or not years.isin(SEASONS).all():
        raise ValueError("season must be a configured integer ending year")
    frame["season"] = years.astype(int)
    if "full_name" not in frame:
        frame["full_name"] = frame.canonical_player_id
    known = frame.canonical_player_id.notna()
    if not frame.loc[known, "canonical_player_id"].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("canonical_player_id must contain nonempty strings")
    if prices:
        for column in ["adp", "adp_rank"]:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if frame[column].isna().any() or not np.isfinite(frame[column]).all() or frame[column].le(0).any():
                raise ValueError(f"{column} must be finite and positive")
        if (frame.adp_rank % 1 != 0).any():
            raise ValueError("adp_rank must contain integers")
        frame["adp_rank"] = frame.adp_rank.astype(int)
        if frame.loc[known].duplicated(KEYS).any() or frame.duplicated(["season", "adp_rank"]).any():
            raise ValueError("Duplicate ADP player or draft-rank keys")
        if not frame.timing_status.isin(["preseason", "opening_day", "after_opener", "unknown"]).all():
            raise ValueError("Unrecognized ADP timing_status")
        frame["available_at"] = pd.to_datetime(frame.available_at, errors="raise", format="mixed").dt.date
        frame["match_status"] = frame.get("match_status", pd.Series("matched", index=frame.index))
    else:
        if not known.all():
            raise ValueError("Game logs contain unresolved player IDs")
        dates = pd.to_datetime(frame.game_date, errors="raise", format="mixed")
        if dates.isna().any():
            raise ValueError("Every game must have an exact game_date")
        frame["game_date"] = dates.dt.date
        if ((dates.dt.year != frame.season) & (dates.dt.year != frame.season - 1)).any():
            raise ValueError("game_date does not belong to its season")
        if frame.duplicated(KEYS + ["game_date"]).any():
            raise ValueError("Duplicate season/date/player game keys")
        frame["fantasy_points"] = pd.to_numeric(frame.fantasy_points, errors="raise")
        if frame.fantasy_points.isna().any() or not np.isfinite(frame.fantasy_points).all():
            raise ValueError("fantasy_points must be finite and non-null")
    return frame


def _signals(logs: pd.DataFrame, min_games: int, threshold: float) -> pd.DataFrame:
    """Classify a completed season without reading its following season."""
    frame = logs.copy()
    frame["part"] = np.where(frame.game_date.le(frame.season.map(ALL_STAR_DATES)), "pre", "post")
    parts = frame.groupby(KEYS + ["part"]).fantasy_points.agg(["size", "mean"]).unstack("part")
    signal = frame.groupby(KEYS).full_name.last().to_frame()
    for period in ["pre", "post"]:
        signal[f"{period}_games"] = parts.get(("size", period), pd.Series(index=parts.index, dtype=float)).fillna(0).astype(int)
        signal[f"{period}_ppg"] = parts.get(("mean", period), pd.Series(index=parts.index, dtype=float))
    signal = signal.reset_index().rename(columns={"season": "signal_season"})
    signal["season"] = signal.signal_season + 1
    signal = signal[signal.season.isin(SEASONS)].copy()
    eligible = signal.pre_games.ge(min_games) & signal.post_games.ge(min_games) & signal.pre_ppg.gt(0)
    signal["surge_pct"] = np.where(signal.pre_ppg.gt(0), signal.post_ppg / signal.pre_ppg - 1, np.nan)
    signal["is_surge"] = pd.Series(pd.NA, index=signal.index, dtype="boolean")
    target = signal.pre_ppg * (1 + threshold)
    signal.loc[eligible, "is_surge"] = (signal.post_ppg.ge(target) | np.isclose(signal.post_ppg, target, rtol=1e-12, atol=0))[eligible]
    return signal


def _outcomes(logs: pd.DataFrame, prices: pd.DataFrame, min_games: int) -> pd.DataFrame:
    # All normalized log rows are actual appearances, even rounded zero minutes.
    actual = logs.groupby(KEYS, as_index=False).agg(actual_total=("fantasy_points", "sum"),
        actual_ppg=("fantasy_points", "mean"), games_played=("fantasy_points", "size"))
    universe = pd.concat([actual[KEYS], prices[KEYS]]).drop_duplicates()
    actual = universe.merge(actual, on=KEYS, how="left", validate="one_to_one")
    complete = actual.season.isin(set(logs.season))
    no_games = complete & actual.games_played.isna()
    actual.loc[no_games, ["actual_total", "games_played"]] = 0
    actual["total_rank"] = actual.groupby("season").actual_total.rank(method="min", ascending=False)
    qualified = actual.games_played.ge(min_games)
    actual["ppg_rank"] = np.nan
    actual.loc[qualified, "ppg_rank"] = actual.loc[qualified].groupby("season").actual_ppg.rank(method="min", ascending=False)
    return actual


def _number(value):
    return None if pd.isna(value) or not math.isfinite(float(value)) else float(value)


def _metrics(rows: pd.DataFrame) -> dict:
    graded = rows[rows.status.eq("graded")]
    ppg = graded[graded.ppg_rank_delta.notna()]
    return {"n_candidates": len(rows), "n_graded": len(graded), "n_ppg": len(ppg),
            "mean_total_rank_delta": _number(graded.total_rank_delta.mean()),
            "median_total_rank_delta": _number(graded.total_rank_delta.median()),
            "underperform_total_rate": _number(100 * graded.total_rank_delta.gt(0).mean()) if len(graded) else None,
            "mean_ppg_rank_delta": _number(ppg.ppg_rank_delta.mean()),
            "median_ppg_rank_delta": _number(ppg.ppg_rank_delta.median()),
            "underperform_ppg_rate": _number(100 * ppg.ppg_rank_delta.gt(0).mean()) if len(ppg) else None}


def run_backtest(game_logs: pd.DataFrame, adp: pd.DataFrame, *, surge_threshold: float = .2,
                 min_games: int = 10, include_opening_day: bool = False) -> dict:
    """Compare prior-season surges and non-surges against following-season ADP.

    Caller-supplied logs must contain complete normalized regular seasons; a
    rolling window must never be passed as a season's final outcomes. ADP with
    unresolved identity is reported, never mistaken for a known zero-game NBA
    player. PPG ranks require at least min_games appearances in the target year.
    This is a descriptive association, not a causal estimate or tuned strategy.
    """
    if not isinstance(min_games, int) or isinstance(min_games, bool) or min_games < 1:
        raise ValueError("min_games must be a positive integer")
    if not np.isfinite(surge_threshold) or surge_threshold < 0:
        raise ValueError("surge_threshold must be finite and nonnegative")
    logs, prices = _validate(game_logs), _validate(adp, prices=True)
    matched = prices.canonical_player_id.notna() & prices.match_status.eq("matched")
    unresolved = prices.loc[~matched].copy()
    unresolved["status"] = "unresolved_adp"
    prices = prices.loc[matched].copy()
    signals = _signals(logs, min_games, surge_threshold)
    # Outer join preserves both signal players lacking a market match and
    # drafted players with no prior signal, including the first dataset year.
    result = signals.merge(prices.drop(columns="full_name"), on=KEYS, how="outer", validate="one_to_one", indicator=True)
    names = prices.set_index(KEYS).full_name
    missing_names = result.full_name.isna()
    result.loc[missing_names, "full_name"] = [names.get((s, p), p) for s, p in result.loc[missing_names, KEYS].itertuples(index=False, name=None)]
    result["signal_season"] = result.season - 1
    result["status"] = "graded"
    missing_prior = ~result.signal_season.isin(set(logs.season))
    insufficient = result.pre_games.fillna(0).lt(min_games) | result.post_games.fillna(0).lt(min_games)
    result.loc[insufficient, "status"] = "insufficient_prior_games"
    result.loc[result.pre_ppg.le(0), "status"] = "nonpositive_pre_ppg"
    result.loc[missing_prior, "status"] = "missing_prior_season"
    # Validate the actual date as well as its label; a claimed 'preseason'
    # status must never grant access to an opening-day or future observation.
    dates = pd.to_datetime(result.available_at, errors="raise", format="mixed")
    openers = pd.to_datetime(result.season.map(SEASON_OPENERS))
    safe = result.timing_status.eq("preseason") & dates.lt(openers)
    if include_opening_day:
        safe |= result.timing_status.eq("opening_day") & dates.eq(openers)
    target = result.status.eq("graded")
    result.loc[target & ~safe.fillna(False), "status"] = "unsafe_adp"
    result.loc[result._merge.eq("left_only"), "status"] = "missing_adp"
    result.loc[result.status.eq("graded") & ~result.season.isin(set(logs.season)), "status"] = "missing_outcomes"
    result = result.drop(columns="_merge").merge(_outcomes(logs, prices, min_games), on=KEYS, how="left", validate="one_to_one")
    graded = result.status.eq("graded")
    result["total_rank_delta"] = (result.total_rank - result.adp_rank).where(graded)
    result["ppg_rank_delta"] = (result.ppg_rank - result.adp_rank).where(graded)
    for rank, output in [("total_rank_delta", "underperformed_total"), ("ppg_rank_delta", "underperformed_ppg")]:
        result[output] = result[rank].gt(0).astype("boolean").where(result[rank].notna() & result.status.eq("graded"))
    # No unqualified/unknown signal can accidentally become a reference player.
    result["is_surge"] = result.is_surge.astype("boolean")
    summary = {"surge_threshold": surge_threshold, "min_games_per_split": min_games,
               "timing_policy": "preseason_plus_opening_day_sensitivity" if include_opening_day else "strict_preseason",
               "n_candidates": len(result), "n_graded": int(result.status.eq("graded").sum()),
               "n_unresolved_adp": len(unresolved),
               "excluded_counts": {str(k): int(v) for k, v in result.loc[~result.status.eq("graded"), "status"].value_counts().items()}}
    seasonal = []
    for cohort, flag in [("surge", True), ("reference", False)]:
        subset = result[result.is_surge.eq(flag).fillna(False)]
        metrics = _metrics(subset)
        summary[f"n_{cohort}"] = metrics["n_graded"]
        for key, value in metrics.items():
            summary[f"{key}_{cohort}"] = value
        for season, rows in subset.groupby("season", sort=True):
            seasonal.append({"season": int(season), "cohort": cohort, **_metrics(rows)})
    summary["underperform_total_rate_gap"] = (
        summary["underperform_total_rate_surge"] - summary["underperform_total_rate_reference"]
        if summary["underperform_total_rate_surge"] is not None and summary["underperform_total_rate_reference"] is not None else None)
    exclusions = pd.concat([result.loc[~result.status.eq("graded")], unresolved], ignore_index=True)
    return {"summary": summary, "players": result.sort_values(KEYS).reset_index(drop=True),
            "season_summary": pd.DataFrame(seasonal, columns=["season", "cohort", *_metrics(result)]),
            "exclusions": exclusions}


def load_data(processed_dir: Path = PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Only load complete, scored data written by the data command."""
    location = Path(processed_dir)
    metadata = json.loads((location / "metadata.json").read_text())
    if metadata.get("scoring") != SCORING:
        raise ValueError("Stored scoring differs from config; rerun python -m src.data")
    files = metadata.get("files", {})
    for name in ["game_logs", "adp"]:
        path = location / f"{name}.parquet"
        expected = files.get(name, files.get(path.name))
        if isinstance(expected, dict):
            expected = expected.get("sha256")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if not expected or actual != expected:
            raise ValueError(f"Stored {name} checksum mismatch; rerun python -m src.data")
    games, prices = pd.read_parquet(location / "game_logs.parquet"), pd.read_parquet(location / "adp.parquet")
    complete = metadata.get("seasons")
    if not isinstance(complete, list) or not set(games.season).issubset(set(complete)):
        raise ValueError("Metadata does not identify complete source seasons")
    return games, prices


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--surge-threshold", type=float, default=.2)
    parser.add_argument("--min-games", type=int, default=10)
    parser.add_argument("--include-opening-day", action="store_true", help="Sensitivity analysis including prices of unknown intraday availability on opening day")
    args = parser.parse_args(argv)
    games, prices = load_data(args.processed_dir)
    report = run_backtest(games, prices, surge_threshold=args.surge_threshold,
                          min_games=args.min_games, include_opening_day=args.include_opening_day)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    surges = report["players"][report["players"].is_surge.eq(True).fillna(False) & report["players"].status.eq("graded")]
    surges.to_csv(args.output_dir / "surge_players.csv", index=False)
    report["season_summary"].to_csv(args.output_dir / "season_summary.csv", index=False)
    report["exclusions"].to_csv(args.output_dir / "exclusions.csv", index=False)
    encoded = json.dumps(report["summary"], indent=2, allow_nan=False)
    (args.output_dir / "summary.json").write_text(encoded + "\n")
    print(encoded)
    print(f"Reports written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
