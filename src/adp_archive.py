"""Import a small, immutable, preseason ESPN ADP archive.

The public mirror committed this CSV on 2024-10-19, before the 2024-25
opening game. This is one partial preseason snapshot, not a historical ADP
database. The original download is retained unchanged; the adapter output in
raw/adp is a local staging CSV consumed by the normalizer. Name resolution is
deliberately left to the normalizer's strict master-player resolver.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .config import RAW_DIR
from .ingest import CachedDownloader, CacheIntegrityError

LOGGER = logging.getLogger(__name__)
ARCHIVE_COMMIT = "8e501a78fa9989edaf532e24580f0a7365c077e1"
ARCHIVE_URL = (
    "https://raw.githubusercontent.com/toddehalexander/"
    f"NBA_ADP_Player_Selector/{ARCHIVE_COMMIT}/nba-adp-All.csv"
)
ARCHIVE_HISTORY_URL = (
    "https://github.com/toddehalexander/NBA_ADP_Player_Selector/"
    f"commit/{ARCHIVE_COMMIT}"
)
ARCHIVE_SHA256 = "0348239132c36cc6269d017744a525ff2945d079abf4e8017ea61f0d9059beaa"
AVAILABLE_AT = "2024-10-19"  # Git commit: 2024-10-19T15:01:34-07:00.


def adapt_espn_2025(payload: bytes) -> pd.DataFrame:
    """Keep priced ESPN rows, preserving fractional ADP and preseason position.

    Global ranks use ascending ESPN ADP, with full name as deterministic tie
    breaker. Position ranks use the first original preseason position token
    (G/F/C in this archive), never a player's later/current roster position.
    Missing ESPN values are excluded; expert rankings and other providers'
    values never substitute for an ESPN market price.
    """
    frame = pd.read_csv(io.BytesIO(payload), keep_default_na=False)
    if not {"Name", "ESPN", "Pos"}.issubset(frame.columns):
        raise ValueError("Archive requires Name, ESPN and Pos columns")
    raw_prices = frame["ESPN"].astype(str).str.strip()
    missing = raw_prices.isin(["", "-"])
    prices = pd.to_numeric(raw_prices.mask(missing), errors="coerce")
    if prices[~missing].isna().any() or not np.isfinite(prices[~missing]).all():
        raise ValueError("Archive contains invalid ESPN ADP values")
    if (prices.dropna() <= 0).any():
        raise ValueError("ESPN ADP must be positive")
    frame = frame.loc[~missing].copy()
    frame["adp"] = prices.loc[~missing].astype(float)
    frame["full_name"] = frame["Name"].astype(str).str.strip()
    frame["market_position"] = frame["Pos"].astype(str).str.strip().map(
        lambda value: re.split(r"[/,\-]", value)[0].strip().upper()
    )
    if frame.empty or frame["full_name"].eq("").any():
        raise ValueError("Archive has no priced players or an empty player name")
    if frame["full_name"].duplicated().any():
        raise ValueError("Duplicate priced names need explicit identity review")
    if not frame["market_position"].isin(["PG", "SG", "SF", "PF", "C", "G", "F"]).all():
        raise ValueError("Archive contains an unknown preseason position")
    frame = frame.sort_values(["adp", "full_name"], kind="stable").reset_index(drop=True)
    frame["season"] = 2025
    frame["adp_rank"] = np.arange(1, len(frame) + 1, dtype=np.int64)
    frame["position_rank"] = frame.groupby("market_position", sort=False).cumcount() + 1
    frame["available_at"] = AVAILABLE_AT
    frame["source"] = ARCHIVE_URL
    return frame[["season", "full_name", "adp", "adp_rank", "position_rank",
                  "available_at", "source", "market_position"]]


def fetch_pinned_2025_espn_adp(*, raw_dir: Path = RAW_DIR,
                                downloader: CachedDownloader | None = None) -> Path:
    """Download once and stage a documented, strictly preseason ADP input."""
    def validate(payload: bytes) -> None:
        if hashlib.sha256(payload).hexdigest() != ARCHIVE_SHA256:
            raise CacheIntegrityError("Pinned ADP archive checksum mismatch")
        adapt_espn_2025(payload)

    downloader = downloader or CachedDownloader()
    original = downloader.get(
        ARCHIVE_URL,
        Path(raw_dir) / "archive_sources" / f"nba-adp-All-{ARCHIVE_COMMIT}.csv",
        validator=validate,
    )
    frame = adapt_espn_2025(original.read_bytes())
    target = Path(raw_dir) / "adp" / "espn_2025.csv"
    encoded = frame.to_csv(index=False, lineterminator="\n").encode()
    if target.exists() and target.read_bytes() != encoded:
        raise CacheIntegrityError(f"Refusing to overwrite a different local ADP file: {target}")
    if not target.exists():
        CachedDownloader._atomic_bytes(target, encoded)
    metadata = {
        "kind": "adapter_output_not_original_download",
        "provider": "ESPN", "season": 2025, "available_at": AVAILABLE_AT,
        "available_at_evidence": ARCHIVE_HISTORY_URL,
        "source": ARCHIVE_URL, "source_sha256": ARCHIVE_SHA256,
        "output_sha256": hashlib.sha256(encoded).hexdigest(),
        "rows": len(frame), "partial_snapshot": True,
        "adp_min": float(frame.adp.min()), "adp_max": float(frame.adp.max()),
        "position_rank_method": "within first original Pos token, sorted by ESPN ADP/name",
        "adp_rank_method": "ascending ESPN ADP/name, contiguous integer ranks",
    }
    CachedDownloader._atomic_bytes(
        target.with_suffix(".csv.adapter.json"),
        json.dumps(metadata, indent=2, sort_keys=True).encode(),
    )
    LOGGER.warning(
        "Staged partial ESPN 2025 snapshot: %d priced players, ADP %.2f–%.2f; "
        "other seasons still require dated ADP CSVs", len(frame), frame.adp.min(), frame.adp.max()
    )
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fetch_pinned_2025_espn_adp(raw_dir=args.raw_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
