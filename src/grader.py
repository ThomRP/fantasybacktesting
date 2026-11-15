"""Standardized cohort grading with chronological baselines and a frozen holdout."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import KEYS, aggregate_outcomes, fit_expected_curve, validate_keys
from .config import HOLDOUT_SEASONS, PROCESSED_DIR, SCORING, SEASONS


def _check_disk_coverage(
    location: Path, game_logs: pd.DataFrame, *, disk_games: bool, disk_adp: bool
) -> set[int]:
    path = location / "coverage.json"
    if not path.is_file():
        raise ValueError("Processed data requires coverage.json; run python -m src.normalize first")
    try:
        coverage = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot read processed coverage.json") from exc
    if not isinstance(coverage, dict):
        raise ValueError("coverage.json must contain a metadata object")
    completed = coverage.get("completed_seasons")
    if not isinstance(completed, list) or any(type(s) is not int or s not in SEASONS for s in completed):
        raise ValueError("coverage.json requires a valid completed_seasons list from complete source downloads")
    if coverage.get("scoring") != SCORING:
        raise ValueError("Processed scoring differs from src.config.SCORING; rerun normalization")
    if disk_games and coverage.get("game_logs") != len(game_logs):
        raise ValueError("coverage.json game_logs count differs from Parquet; rerun normalization")
    hashes = coverage.get("processed_sha256")
    if hashes is not None:
        if not isinstance(hashes, dict):
            raise ValueError("coverage.json processed_sha256 must be a hash mapping")
        for stem, from_disk in [("game_logs", disk_games), ("adp", disk_adp)]:
            if not from_disk:
                continue
            with (location / f"{stem}.parquet").open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if hashes.get(stem) != digest:
                raise ValueError(f"Processed {stem}.parquet SHA256 differs from coverage.json; rerun normalization")
    return set(completed)


def _metrics(frame: pd.DataFrame) -> dict:
    graded = frame[frame["status"] == "graded"]
    ppg = graded["alpha_ppg"].dropna()
    return {
        "n_cohort": int(len(frame)),
        "n_graded": int(len(graded)),
        "n_ppg": int(len(ppg)),
        "n_excluded": int(len(frame) - len(graded)),
        "mean_alpha_total": float(graded["alpha_total"].mean()),
        "median_alpha_total": float(graded["alpha_total"].median()),
        "mean_alpha_ppg": float(ppg.mean()),
        "median_alpha_ppg": float(ppg.median()),
        "beat_adp_rate": float(100 * (graded["alpha_total"] > 0).mean()) if len(graded) else float("nan"),
    }


def evaluate_cohort(
    cohort_df: pd.DataFrame,
    *,
    game_logs: pd.DataFrame | None = None,
    adp: pd.DataFrame | None = None,
    processed_dir: str | Path | None = None,
) -> dict:
    """Grade season/player signals using only prior, non-holdout seasons.

    Returns summary metrics, five-year era and development/holdout breakdowns,
    per-player results, and explicit exclusions. ``beat_adp_rate`` is a percent
    of graded *total-points* observations; ties are not wins. A zero-game player
    has zero total and unknown PPG. Callers must construct signals from data
    available before each season; outcome grading cannot audit strategy code.

    All supplied logs must cover complete regular seasons. Missing entire
    seasons are excluded; partial seasons must not be supplied as completed.
    Every row represents a played appearance, including rounded zero minutes;
    explicit DNP placeholder rows must be omitted by the caller.
    Disk loading additionally requires a normalization coverage manifest with
    verified complete seasons and scoring identical to the current config.
    """
    cohort = validate_keys(cohort_df, "cohort")
    if cohort.duplicated(KEYS).any():
        raise ValueError("cohort has duplicate season/player keys")
    # Extra strategy columns cannot overwrite outcomes or model metadata.
    cohort = cohort[KEYS]
    location = Path(processed_dir) if processed_dir is not None else PROCESSED_DIR
    disk_games = game_logs is None
    disk_adp = adp is None
    if game_logs is None:
        game_logs = pd.read_parquet(location / "game_logs.parquet")
    if adp is None:
        adp = pd.read_parquet(location / "adp.parquet")
    incomplete_seasons = set()
    if disk_games or disk_adp:
        complete = _check_disk_coverage(location, game_logs, disk_games=disk_games, disk_adp=disk_adp)
        game_logs = validate_keys(game_logs, "game_logs")
        incomplete_seasons = set(game_logs["season"]) - complete
        # Quarantine makes a season unsuitable for totals or PPG training. Keep
        # other completed seasons usable and expose these exclusions explicitly.
        game_logs = game_logs[game_logs["season"].isin(complete)]
    outcomes = aggregate_outcomes(game_logs, adp)
    # Keep an ADP-match indicator separate from a missing outcome.
    result = cohort.merge(outcomes, on=KEYS, how="left", validate="one_to_one", indicator=True)
    result["status"] = "graded"
    result.loc[result["_merge"] == "left_only", "status"] = "missing_adp"
    matched = result["_merge"] == "both"
    result.loc[matched & ~result["season_available"].fillna(False).astype(bool), "status"] = "missing_season"
    result.loc[matched & result["season"].isin(incomplete_seasons), "status"] = "incomplete_season"
    result.loc[matched & ~result["adp_rank"].between(1, 200), "status"] = "outside_draft_range"
    result = result.drop(columns="_merge")
    for col in ["expected_total_points", "expected_ppg", "alpha_total", "alpha_ppg"]:
        result[col] = np.nan
    result["training_seasons"] = ""
    result["n_training_rows"] = 0
    result["n_training_ppg"] = 0
    for season in sorted(result["season"].unique()):
        target = (result["season"] == season) & (result["status"] == "graded")
        if not target.any():
            continue
        train = outcomes[
            (outcomes["season"] < season)
            & ~outcomes["season"].isin(HOLDOUT_SEASONS)
            & outcomes["season_available"]
        ]
        if train.empty or train.loc[train["adp_rank"].between(1, 200), "actual_ppg"].notna().sum() == 0:
            result.loc[target, "status"] = "no_training"
            continue
        curve = fit_expected_curve(train).set_index("adp_rank")
        for col in ["expected_total_points", "expected_ppg"]:
            result.loc[target, col] = result.loc[target, "adp_rank"].map(curve[col])
        result.loc[target, "training_seasons"] = ",".join(str(x) for x in curve.attrs["training_seasons"])
        result.loc[target, "n_training_rows"] = curve.attrs["n_training_rows"]
        result.loc[target, "n_training_ppg"] = curve.attrs["n_training_ppg"]
    result["alpha_total"] = result["actual_total_points"] - result["expected_total_points"]
    result["alpha_ppg"] = result["actual_ppg"] - result["expected_ppg"]
    result["split"] = np.where(result["season"].isin(HOLDOUT_SEASONS), "holdout", "development")
    era_start = 2006 + ((result["season"] - 2006) // 5) * 5
    result["era"] = era_start.astype(str) + "–" + (era_start + 4).astype(str)
    eras = [{"era": label, **_metrics(rows)} for label, rows in result.groupby("era", sort=True)]
    splits = [{"split": label, **_metrics(rows)} for label, rows in result.groupby("split", sort=True)]
    summary = _metrics(result)
    summary["holdout_seasons"] = list(HOLDOUT_SEASONS)
    summary["exclusion_counts"] = {str(k): int(v) for k, v in result.loc[result["status"] != "graded", "status"].value_counts().items()}
    return {
        "summary": summary,
        "era_breakdown": pd.DataFrame(eras),
        "split_breakdown": pd.DataFrame(splits),
        "player_results": result,
        "exclusions": result.loc[result["status"] != "graded"].copy(),
    }
