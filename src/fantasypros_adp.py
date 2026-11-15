"""Historical ESPN market prices from FantasyPros' year-specific NBA ADP pages.

Dates are source-reported last-update dates, not independently captured Wayback
snapshots. Current team/position/injury labels on historical pages are ignored.
No expert ranks, Yahoo prices, or current rankings fill a missing ESPN price.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import logging
from pathlib import Path
import re
import unicodedata

import pandas as pd
import requests

from .config import RAW_DIR, PROCESSED_DIR, SEASONS, SEASON_OPENERS, season_label
from .ingest import CachedDownloader, CacheIntegrityError
from .normalize import normalize_name, nba_id

LOGGER = logging.getLogger(__name__)


def _exact_name(name: str) -> str:
    """Keep suffixes to distinguish Gary Payton from Gary Payton II."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9\s]", "", text).split())


class ArchiveUnavailable(ValueError):
    """A page cannot establish dated, preseason ESPN ADP for the requested year."""


class _Tables(HTMLParser):
    """Parse only required tables, including the site's omitted closing tr tags."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables, self.table, self.row, self.cell = {}, None, None, None
        self.title, self.in_title = [], False

    def _close_cell(self):
        if self.cell is not None and self.row is not None:
            self.row.append(self.cell)
        self.cell = None

    def _close_row(self):
        self._close_cell()
        if self.row is not None and self.table:
            self.tables[self.table].append(self.row)
        self.row = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "title":
            self.in_title = True
        if tag == "table" and attrs.get("id") in {"data", "experts"}:
            self.table = attrs["id"]
            self.tables[self.table] = []
        if self.table:
            if tag == "tr":
                self._close_row()
                self.row = []
            elif tag in {"td", "th"}:
                self._close_cell()
                self.cell = {"text": "", "name": None}
            elif self.cell is not None and "fp-player-name" in attrs:
                self.cell["name"] = attrs["fp-player-name"]

    def handle_data(self, data):
        if self.in_title:
            self.title.append(data)
        if self.cell is not None:
            self.cell["text"] += data

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        elif tag in {"td", "th"}:
            self._close_cell()
        elif tag == "tr":
            self._close_row()
        elif tag == "table" and self.table:
            self._close_row()
            self.table = None


def archive_url(season: int) -> str:
    return f"https://www.fantasypros.com/nba/adp/overall.php?year={season - 1}"


def parse_archive(payload: bytes, season: int) -> tuple[pd.DataFrame, dict]:
    """Fail closed on wrong years, unknown dates, same-day updates and bad prices."""
    parser = _Tables()
    parser.feed(payload.decode("utf-8-sig"))
    title = " ".join("".join(parser.title).split())
    if season_label(season) not in title or "Average Draft Position (ADP)" not in title:
        raise ArchiveUnavailable(f"Page is not the requested {season_label(season)} NBA ADP archive")
    updates = []
    for row in parser.tables.get("experts", []):
        if any(cell["text"].strip() == "ESPN" for cell in row):
            # The site's hidden numeric sort key is a UTC Unix date, not a
            # trustworthy intraday publication time. Keep only its calendar day.
            text = row[-1]["text"].strip()
            stamp = re.match(r"(\d{10})", text)
            if stamp:
                updates.append(datetime.fromtimestamp(int(stamp.group(1)), timezone.utc).date())
    if len(updates) != 1:
        raise ArchiveUnavailable("No unique source-reported ESPN last-update date")
    available = updates[0]
    if not (datetime(season - 1, 1, 1).date() <= available < SEASON_OPENERS[season]):
        raise ArchiveUnavailable(f"ESPN update {available} is not strictly before opener {SEASON_OPENERS[season]}; need an earlier dated snapshot")
    rows = parser.tables.get("data", [])
    if not rows:
        raise ArchiveUnavailable("No player ADP table")
    headers = [cell["text"].strip() for cell in rows[0]]
    if headers.count("ESPN") != 1 or "Player" not in headers:
        raise ArchiveUnavailable("Player table has no unique ESPN price column")
    price_index, name_index = headers.index("ESPN"), headers.index("Player")
    records = []
    for row in rows[1:]:
        if len(row) != len(headers):
            raise ArchiveUnavailable("Malformed player table row")
        price_text = row[price_index]["text"].strip()
        if price_text in {"", "-", "—", "–", "N/A"}:
            continue
        try:
            price = float(price_text.replace(",", ""))
        except ValueError as exc:
            raise ArchiveUnavailable(f"Non-numeric ESPN price {price_text!r}") from exc
        if not 0 < price < float("inf"):
            raise ArchiveUnavailable("ESPN price must be finite and positive")
        name = row[name_index]["name"]
        if not name or not normalize_name(name):
            raise ArchiveUnavailable("Missing player-name attribute; never parse current team/position labels")
        records.append({"full_name": name.strip(), "adp": price})
    if not records:
        raise ArchiveUnavailable("Archive has no priced ESPN players")
    frame = pd.DataFrame(records)
    if frame.full_name.map(_exact_name).duplicated().any():
        raise ArchiveUnavailable("Duplicate player names need explicit identity review")
    # Consensus order is NOT ESPN order. Assign ranks before any identity gaps
    # are removed so unresolvable players never shift everyone else's slots.
    frame = frame.sort_values(["adp", "full_name"], kind="stable").reset_index(drop=True)
    frame["season"] = season
    frame["adp_rank"] = range(1, len(frame) + 1)
    frame["position_rank"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame["position_rank_status"] = "unavailable_historical_positions"
    frame["available_at"] = available.isoformat()
    frame["source"] = archive_url(season)
    metadata = {
        "provider": "ESPN", "archive_host": "FantasyPros", "season": season,
        "available_at": available.isoformat(), "available_at_evidence": archive_url(season),
        "date_evidence_kind": "source_reported_update_date_not_independent_archive_capture",
        "source": archive_url(season), "source_sha256": hashlib.sha256(payload).hexdigest(),
        "source_priced_rows": len(frame), "partial_snapshot": True,
        "market_format": "not_specified_by_archive_do_not_assume_points_only",
        "position_rank_method": "unavailable; current page positions intentionally ignored",
        "adp_rank_method": "ascending observed ESPN ADP/name; not consensus ranks",
    }
    return frame, metadata


def _name_index(raw_dir: Path, processed_dir: Path) -> dict[str, set[str]]:
    """Use exact historical roster aliases and the master, never fuzzy matching."""
    names: dict[str, set[str]] = {}
    def add(name, player_id):
        if pd.notna(name):
            names.setdefault(normalize_name(name), set()).add(nba_id(player_id))
            names.setdefault("exact:" + _exact_name(name), set()).add(nba_id(player_id))
    players_path = Path(processed_dir) / "players.parquet"
    if players_path.exists():
        coverage_path = Path(processed_dir) / "coverage.json"
        if not coverage_path.exists():
            raise CacheIntegrityError("Master identities require coverage.json; rerun normalization")
        digest = hashlib.sha256(players_path.read_bytes()).hexdigest()
        if json.loads(coverage_path.read_text()).get("processed_sha256", {}).get("players") != digest:
            raise CacheIntegrityError("Master player checksum mismatch; rerun normalization")
        for row in pd.read_parquet(players_path).itertuples():
            add(row.full_name, row.canonical_player_id)
    for pattern, name_column in [("rosters_*.parquet", "player"), ("game_logs_*.parquet", "player_name")]:
        for path in sorted((Path(raw_dir) / "sportsdataverse").glob(pattern)):
            meta_path = path.with_suffix(path.suffix + ".metadata.json")
            if not meta_path.exists() or json.loads(meta_path.read_text()).get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
                raise CacheIntegrityError(f"NBA identity cache checksum missing/mismatched: {path}; rerun ingest to diagnose")
            # Include log-only players before first normalization, so a fresh
            # install and an upgrade resolve the same identity universe.
            # These names are identity aids, never historical strategy features.
            for name, player_id in pd.read_parquet(path, columns=[name_column, "player_id"]).itertuples(index=False, name=None):
                add(name, player_id)
    return names


def fetch_archives(*, raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR,
                   seasons=SEASONS, downloader=None) -> dict:
    """Cache every requested page and stage safe seasons without overwriting inputs.

    Expected coverage gaps are reported instead of aborting other seasons. A
    network/cache failure is a separate status, never described as archive absence.
    """
    raw_dir = Path(raw_dir)
    target_dir = raw_dir / "adp"
    target_dir.mkdir(parents=True, exist_ok=True)
    downloader = downloader or CachedDownloader()
    names = _name_index(raw_dir, Path(processed_dir))
    if not names:
        raise ValueError("No NBA identity data; run src.ingest --source sportsdataverse first")
    existing: dict[int, list[Path]] = {}
    for path in sorted(target_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        if "season" not in frame:
            raise ValueError(f"Existing ADP CSV missing season: {path}")
        for year in frame.season.dropna().unique():
            existing.setdefault(int(year), []).append(path)
    results = []
    for season in seasons:
        target = target_dir / f"fantasypros_espn_{season}.csv"
        others = [p for p in existing.get(season, []) if p != target]
        if others:
            results.append({"season": season, "status": "existing_input", "files": [str(p) for p in others]})
            continue
        try:
            source = downloader.get(archive_url(season), raw_dir / "archive_sources" / f"fantasypros_{season - 1}.html")
            frame, metadata = parse_archive(source.read_bytes(), season)
            resolved, unresolved = [], []
            for row in frame.to_dict("records"):
                matches = names.get("exact:" + _exact_name(row["full_name"]), set())
                if len(matches) != 1:
                    matches = names.get(normalize_name(row["full_name"]), set())
                if len(matches) == 1:
                    row["canonical_player_id"] = next(iter(matches))
                    resolved.append(row)
                else:
                    unresolved.append({"season": season, "full_name": row["full_name"], "adp_rank": row["adp_rank"],
                                       "reason": "ambiguous_name" if matches else "unmapped_name"})
            if not resolved:
                raise ArchiveUnavailable("No unambiguous NBA identities; supply reviewed canonical IDs")
            output = pd.DataFrame(resolved)
            encoded = output.to_csv(index=False, lineterminator="\n").encode()
            if target.exists() and target.read_bytes() != encoded:
                raise CacheIntegrityError(f"Refusing to overwrite changed local ADP file {target}; archive/review it explicitly")
            if not target.exists():
                CachedDownloader._atomic_bytes(target, encoded)
            metadata.update({"kind": "adapter_output_not_original_download", "rows": len(output),
                             "output_sha256": hashlib.sha256(encoded).hexdigest(), "unresolved_players": unresolved})
            CachedDownloader._atomic_bytes(target.with_suffix(".csv.adapter.json"), json.dumps(metadata, indent=2).encode())
            results.append({"season": season, "status": "imported", "rows": len(output),
                            "available_at": metadata["available_at"], "unresolved_players": unresolved,
                            "missing_historical_position_ranks": len(output), "file": str(target)})
        except CacheIntegrityError:
            raise  # Corruption or a user edit is not ordinary missing archive coverage.
        except ArchiveUnavailable as exc:
            results.append({"season": season, "status": "manual_snapshot_needed", "reason": str(exc), "url": archive_url(season)})
            LOGGER.warning("Season %s: %s", season, exc)
        except (requests.RequestException, OSError) as exc:
            results.append({"season": season, "status": "fetch_failed", "reason": str(exc), "url": archive_url(season)})
            LOGGER.warning("Season %s fetch failed: %s", season, exc)
    for row in results:
        row["season_label"] = season_label(row["season"])
        row["snapshot_before"] = SEASON_OPENERS[row["season"]].isoformat()
    report = {"results": results, "manual_requirements": [
        "For rejected/missing years: real ESPN draft ADP captured strictly before the season opener, with the source URL/file and dated evidence; not expert rankings.",
        "For omitted players: verify the NBA canonical ID, then supply a corrected complete season CSV without shifting the other ESPN draft ranks.",
        "For position strategies: original preseason positions or position ranks. Historical pages show current labels, so automatic position_rank is null.",
    ]}
    CachedDownloader._atomic_bytes(target_dir / "coverage_report.json", json.dumps(report, indent=2).encode())
    return report
