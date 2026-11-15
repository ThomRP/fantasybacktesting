"""End-of-season outcomes and strictly chronological ADP expectations.

``season`` is the NBA season's ending year.  A fitted curve is descriptive of
its training sample; use :func:`fit_chronological_curve` for backtesting.
"""
from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .config import HOLDOUT_SEASONS, SEASONS

KEYS = ["season", "canonical_player_id"]
OUTCOMES = ["actual_total_points", "actual_ppg", "games_played"]


def validate_keys(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Validate entity/season keys without silently coercing invalid IDs."""
    missing = set(KEYS) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    result = frame.copy()
    ids = result["canonical_player_id"]
    if ids.isna().any() or not ids.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError(f"{label} canonical_player_id must be nonempty strings")
    seasons = pd.to_numeric(result["season"], errors="raise")
    if seasons.isna().any() or not np.isfinite(seasons).all() or (seasons % 1 != 0).any():
        raise ValueError(f"{label} season must be a finite integer ending year")
    result["season"] = seasons.astype("int64")
    if not result["season"].isin(SEASONS).all():
        raise ValueError(f"{label} contains unsupported seasons; supported: {min(SEASONS)}–{max(SEASONS)}")
    return result


def validate_adp(adp: pd.DataFrame) -> pd.DataFrame:
    result = validate_keys(adp, "adp")
    if result.duplicated(KEYS).any():
        raise ValueError("adp has duplicate season/player keys")
    if "adp_rank" not in result:
        raise ValueError("adp is missing adp_rank")
    rank = pd.to_numeric(result["adp_rank"], errors="raise")
    if rank.isna().any() or not np.isfinite(rank).all() or (rank <= 0).any() or (rank % 1 != 0).any():
        raise ValueError("adp_rank must contain positive finite integers")
    result["adp_rank"] = rank.astype("int64")
    if result.duplicated(["season", "adp_rank"]).any():
        raise ValueError("adp has duplicate season/adp_rank slots")
    if "adp" in result:
        prices = pd.to_numeric(result["adp"], errors="raise")
        if prices.isna().any() or not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError("adp must contain positive finite prices")
        result["adp"] = prices.astype(float)
        ordered = result.sort_values(["season", "adp_rank"])
        if ordered.groupby("season")["adp"].diff().lt(0).any():
            raise ValueError("adp_rank ordering is inconsistent with ascending ADP prices")
    return result


def aggregate_outcomes(
    game_logs: pd.DataFrame,
    adp: pd.DataFrame,
    *,
    completed_seasons: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Join market prices to outcomes, retaining drafted players with no games.

    If *no* logs exist for an entire season, its outcomes remain unknown rather
    than becoming zeros. Present seasons must come from complete regular-season
    ingestion; incomplete user-supplied frames cannot establish season coverage.
    Every input row represents an appearance. Rounded zero-minute appearances
    count; callers must omit DNP placeholders rather than inferring them from MIN.
    Pass ``coverage.json["completed_seasons"]`` when using processed data so
    quarantined/incomplete seasons remain unknown. If omitted, the caller is
    responsible for supplying only complete, validated regular seasons.
    """
    prices = validate_adp(adp)
    logs = validate_keys(game_logs, "game_logs")
    required = {"game_date", "fantasy_points", "minutes_or_snaps"}
    if not required.issubset(logs):
        raise ValueError(f"game_logs is missing columns: {sorted(required - set(logs))}")
    dates = pd.to_datetime(logs["game_date"], errors="raise", utc=True)
    if dates.isna().any():
        raise ValueError("game_logs game_date cannot be null")
    if ((dates.dt.year != logs["season"]) & (dates.dt.year != logs["season"] - 1)).any():
        raise ValueError("game_date year does not belong to its season")
    logs["game_date"] = dates.dt.date
    if logs.duplicated(KEYS + ["game_date"]).any():
        raise ValueError("game_logs has duplicate season/date/player keys")
    for col in ["fantasy_points", "minutes_or_snaps"]:
        logs[col] = pd.to_numeric(logs[col], errors="raise")
        if logs[col].isna().any() or not np.isfinite(logs[col]).all():
            raise ValueError(f"game_logs {col} must be finite and non-null")
    if (logs["minutes_or_snaps"] < 0).any():
        raise ValueError("minutes_or_snaps cannot be negative")
    if completed_seasons is not None:
        completed = list(completed_seasons)
        if any(not isinstance(s, Integral) or isinstance(s, bool) or s not in SEASONS for s in completed):
            raise ValueError("completed_seasons must contain supported integer season ending years")
        logs = logs[logs["season"].isin(completed)]
    covered = set(logs["season"])
    actual = logs.groupby(KEYS, as_index=False).agg(
        actual_total_points=("fantasy_points", "sum"),
        actual_ppg=("fantasy_points", "mean"),
        games_played=("fantasy_points", "size"),
    )
    result = prices.merge(actual, on=KEYS, how="left", validate="one_to_one")
    result["season_available"] = result["season"].isin(covered)
    no_games = result["season_available"] & result["games_played"].isna()
    result.loc[no_games, ["actual_total_points", "games_played"]] = 0
    return result


def fit_expected_curve(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Fit a descriptive decreasing curve for draft ranks 1–200.

    Callers explicitly choose the training sample. Zero-game players contribute
    zero totals but no PPG observation. Ranks beyond 200 are outside this model.
    Edge ranks outside the observed support use the nearest fitted expectation.
    """
    data = validate_adp(outcomes)
    for col in ["actual_total_points", "actual_ppg"]:
        if col not in data:
            raise ValueError(f"outcomes is missing {col}")
        data[col] = pd.to_numeric(data[col], errors="raise")
        if np.isinf(data[col]).any():
            raise ValueError(f"{col} cannot be infinite")
    data = data[data["adp_rank"].between(1, 200)]
    curve = pd.DataFrame({"adp_rank": np.arange(1, 201, dtype="int64")})
    for observed, expected in [("actual_total_points", "expected_total_points"), ("actual_ppg", "expected_ppg")]:
        usable = data.dropna(subset=[observed])
        if usable.empty:
            raise ValueError(f"No training observations for {observed}")
        model = IsotonicRegression(increasing=False, out_of_bounds="clip")
        model.fit(usable["adp_rank"].to_numpy(), usable[observed].to_numpy())
        curve[expected] = model.predict(curve["adp_rank"].to_numpy())
    curve.attrs["training_seasons"] = sorted(data["season"].unique().tolist())
    curve.attrs["n_training_rows"] = len(data.dropna(subset=["actual_total_points"]))
    curve.attrs["n_training_ppg"] = len(data.dropna(subset=["actual_ppg"]))
    curve.attrs["observed_rank_min"] = int(data["adp_rank"].min())
    curve.attrs["observed_rank_max"] = int(data["adp_rank"].max())
    return curve


def fit_chronological_curve(
    game_logs: pd.DataFrame,
    adp: pd.DataFrame,
    evaluation_season: int,
    *,
    completed_seasons: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Use earlier seasons only; never train on the four-season holdout.

    For Parquet inputs pass the manifest's ``completed_seasons``. Without it,
    callers must supply only complete, validated seasons, as with the explicit
    DataFrame grader API. This function cannot infer source completeness.
    """
    if evaluation_season not in SEASONS:
        raise ValueError("evaluation_season is outside the configured window")
    outcomes = aggregate_outcomes(game_logs, adp, completed_seasons=completed_seasons)
    train = outcomes[
        (outcomes["season"] < evaluation_season)
        & ~outcomes["season"].isin(HOLDOUT_SEASONS)
        & outcomes["season_available"]
    ]
    if train.empty:
        raise ValueError(f"No earlier training seasons for {evaluation_season}")
    return fit_expected_curve(train)
