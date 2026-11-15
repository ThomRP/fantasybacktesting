"""Resumable, immutable NBA downloads and a deliberately strict local ADP adapter.

Seasons are END years: 2026 means 2025-26. The mirror is an explicit
alternative to stats.nba.com, never a silent fallback. No request needs a key.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Callable
from urllib.parse import urlsplit

import pandas as pd
import requests

from .config import RAW_DIR, SEASONS, season_label

LOGGER = logging.getLogger(__name__)
MIRROR = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
RETRYABLE = {408, 425, 429, 500, 502, 503, 504}


class CacheIntegrityError(ValueError):
    """A cache no longer matches its recorded content or request identity."""


class CachedDownloader:
    """Sequential, TLS-verified HTTP downloads with bounded transient retries.

    An immutable payload and its SHA256/provenance sidecar are committed via
    atomic renames. A pending metadata file lets a run recover if interrupted
    between committing the payload and committing its final metadata. There is
    no force-refresh option: remove/archive a cache explicitly to refresh it.
    """

    def __init__(self, *, session=None, min_interval: float = 1.0,
                 retries: int = 3, timeout: float = 45.0,
                 sleep: Callable[[float], None] = time.sleep):
        if min_interval < 0 or retries < 0 or timeout <= 0:
            raise ValueError("Invalid downloader limits")
        self.session = session if session is not None else requests.Session()
        self.min_interval, self.retries, self.timeout = min_interval, retries, timeout
        self.sleep = sleep
        self._last_request = None

    @staticmethod
    def _metadata_path(path: Path) -> Path:
        return path.with_suffix(path.suffix + ".metadata.json")

    @staticmethod
    def _atomic_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        name = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
                name = handle.name
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, path)
        finally:
            if name is not None and os.path.exists(name):
                os.unlink(name)

    def get(self, url: str, path: Path, *, params: dict | None = None,
            headers: dict | None = None,
            validator: Callable[[bytes], None] | None = None) -> Path:
        """Fetch once; validate both downloaded and reused cache contents."""
        path = Path(path)
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.username or parsed.password:
            raise ValueError("Only credential-free HTTPS source URLs may be cached")
        request_identity = {"url": url, "params": params or {}}
        meta_path = self._metadata_path(path)
        pending_path = meta_path.with_suffix(meta_path.suffix + ".pending")
        if path.exists():
            # Recover only our fully staged metadata, never invent provenance.
            metadata_source = meta_path if meta_path.exists() else pending_path
            if not metadata_source.exists():
                raise CacheIntegrityError(f"Missing cache provenance for {path}")
            metadata = json.loads(metadata_source.read_text())
            payload = path.read_bytes()
            if metadata.get("sha256") != hashlib.sha256(payload).hexdigest():
                raise CacheIntegrityError(f"Cache checksum mismatch: {path}")
            if any(metadata.get(k) != v for k, v in request_identity.items()):
                raise CacheIntegrityError(f"Cache request identity mismatch: {path}")
            if validator:
                validator(payload)
            if metadata_source == pending_path:
                os.replace(pending_path, meta_path)
            LOGGER.info("Cache hit: %s", path)
            return path
        for attempt in range(self.retries + 1):
            if self._last_request is not None:
                self.sleep(max(0.0, self.min_interval - (time.monotonic() - self._last_request)))
            try:
                self._last_request = time.monotonic()
                response = self.session.get(url, params=params, headers=headers,
                                            timeout=self.timeout)
                if response.status_code in RETRYABLE and attempt < self.retries:
                    delay = min(60.0, 2.0 ** attempt)
                    retry_after = response.headers.get("Retry-After", "")
                    if retry_after.isdigit():
                        delay = min(60.0, max(delay, float(retry_after)))
                    LOGGER.warning("Transient HTTP %s; retrying in %.1fs", response.status_code, delay)
                    self.sleep(delay)
                    continue
                response.raise_for_status()
                payload = response.content
                if validator:
                    validator(payload)
                metadata = {**request_identity,
                            "fetched_at": datetime.now(timezone.utc).isoformat(),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "bytes": len(payload)}
                self._atomic_bytes(pending_path, json.dumps(metadata, indent=2, sort_keys=True).encode())
                self._atomic_bytes(path, payload)
                os.replace(pending_path, meta_path)
                LOGGER.info("Cached %s (%d bytes)", path, len(payload))
                return path
            except (requests.Timeout, requests.ConnectionError) as exc:
                # A denied CONNECT is a policy/access problem, not a transient timeout.
                if attempt >= self.retries or "403" in str(exc):
                    raise
                self.sleep(min(60.0, 2.0 ** attempt))
        raise AssertionError("Unreachable retry loop")


def _nba_frame(payload: bytes, index: int = 0) -> pd.DataFrame:
    obj = json.loads(payload)
    result_sets = obj.get("resultSets", obj.get("resultSet"))
    if isinstance(result_sets, dict):
        result_sets = [result_sets]
    if not isinstance(result_sets, list) or len(result_sets) <= index:
        raise ValueError("NBA response is missing result sets")
    table = result_sets[index]
    return pd.DataFrame(table["rowSet"], columns=table["headers"])


def load_nba_table(path: str | Path, index: int = 0) -> pd.DataFrame:
    return _nba_frame(Path(path).read_bytes(), index=index)


def load_game_logs(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else load_nba_table(path)


def load_roster(path: str | Path) -> pd.DataFrame:
    """Read an unmodified roster mirror, or an official NBA biography table."""
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else load_nba_table(path)


def _validate_player_logs(payload: bytes, *, parquet: bool) -> None:
    frame = pd.read_parquet(io.BytesIO(payload)) if parquet else _nba_frame(payload)
    names = {str(c).lower() for c in frame.columns}
    required = {"player_id", "game_date", "matchup", "min", "pts", "reb", "ast"}
    if not required.issubset(names) or frame.empty:
        raise ValueError("Source must contain nonempty PLAYER logs, dates, matchups and box scores; team logs are rejected")


def _fetch_nba(endpoint, path: Path, downloader: CachedDownloader, validator=None) -> Path:
    from nba_api.stats.library.http import NBAStatsHTTP
    url = NBAStatsHTTP.base_url.format(endpoint=endpoint.endpoint)
    return downloader.get(url, path, params=endpoint.parameters,
                          headers=NBAStatsHTTP.headers, validator=validator or _nba_frame)


def fetch_game_logs(season: int, *, source: str = "nba", raw_dir: Path = RAW_DIR,
                    downloader: CachedDownloader | None = None) -> Path:
    downloader = downloader or CachedDownloader()
    if source == "sportsdataverse":
        return downloader.get(
            f"{MIRROR}/nba_stats_player_game_logs/player_game_logs_{season}.parquet",
            Path(raw_dir) / source / f"game_logs_{season}.parquet",
            validator=lambda data: _validate_player_logs(data, parquet=True))
    if source != "nba":
        raise ValueError(f"Unknown game log source: {source}")
    from nba_api.stats.endpoints.leaguegamelog import LeagueGameLog
    endpoint = LeagueGameLog(season=season_label(season),
                            player_or_team_abbreviation="P",
                            season_type_all_star="Regular Season", get_request=False)
    return _fetch_nba(endpoint, Path(raw_dir) / "nba" / f"game_logs_{season}.json", downloader,
                      lambda data: _validate_player_logs(data, parquet=False))


def fetch_player_directory(*, raw_dir: Path = RAW_DIR,
                           downloader: CachedDownloader | None = None) -> Path:
    from nba_api.stats.endpoints.commonallplayers import CommonAllPlayers
    endpoint = CommonAllPlayers(is_only_current_season=0,
                                season=season_label(max(SEASONS)), get_request=False)
    return _fetch_nba(endpoint, Path(raw_dir) / "nba" / "players.json", downloader or CachedDownloader())


def fetch_player_info(player_id: str | int, *, raw_dir: Path = RAW_DIR,
                      downloader: CachedDownloader | None = None) -> Path:
    player_id = str(player_id)
    if not player_id.isdigit():
        raise ValueError("NBA player IDs must contain only digits")
    from nba_api.stats.endpoints.commonplayerinfo import CommonPlayerInfo
    return _fetch_nba(CommonPlayerInfo(player_id=player_id, get_request=False),
                      Path(raw_dir) / "nba" / "players" / f"{player_id}.json",
                      downloader or CachedDownloader())


def fetch_roster(season: int, *, raw_dir: Path = RAW_DIR,
                 downloader: CachedDownloader | None = None) -> Path:
    """The NBA Stats roster mirror supplies names, birth dates and positions."""
    def validate(payload: bytes) -> None:
        frame = pd.read_parquet(io.BytesIO(payload))
        if frame.empty or not {"player_id", "player", "position", "birth_date"}.issubset(frame.columns):
            raise ValueError("Roster mirror schema is not a player biography table")
    return (downloader or CachedDownloader()).get(
        f"{MIRROR}/nba_stats_rosters/rosters_{season}.parquet",
        Path(raw_dir) / "sportsdataverse" / f"rosters_{season}.parquet", validator=validate)


def fetch_adp_csv(url: str, *, filename: str, raw_dir: Path = RAW_DIR,
                  downloader: CachedDownloader | None = None) -> Path:
    """Cache an explicitly selected archive without inventing its snapshot date.

    The normalizer enforces available_at/source and pre-opening-night provenance.
    Current ADP pages cannot establish historical draft prices.
    """
    if Path(filename).name != filename or not filename.endswith(".csv"):
        raise ValueError("ADP filename must be a plain .csv basename")
    def validate(payload: bytes) -> None:
        frame = pd.read_csv(io.BytesIO(payload))
        if frame.empty or not {"season", "adp", "available_at", "source"}.issubset(frame.columns):
            raise ValueError("ADP CSV needs season, adp, available_at and source")
    return (downloader or CachedDownloader()).get(
        url, Path(raw_dir) / "adp" / filename, validator=validate)


def read_adp_csvs(*, raw_dir: Path = RAW_DIR, seasons=SEASONS) -> pd.DataFrame:
    paths = sorted((Path(raw_dir) / "adp").glob("*.csv"))
    frames = []
    for path in paths:
        frame = pd.read_csv(path, dtype={"canonical_player_id": "string", "player_id": "string"})
        if "season" not in frame:
            raise ValueError(f"ADP CSV is missing season: {path}")
        frame["input_file"] = str(path)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    present = set(pd.to_numeric(combined["season"], errors="raise").astype(int)) if frames else set()
    missing = sorted(set(seasons) - present)
    if missing:
        LOGGER.warning("Manual dated ADP CSV drops needed in %s for end-year seasons: %s",
                       Path(raw_dir) / "adp", ", ".join(map(str, missing)))
    if frames:
        combined = combined.loc[pd.to_numeric(combined["season"], errors="raise").isin(seasons)].copy()
    return combined


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["nba", "sportsdataverse"], default="nba")
    parser.add_argument("--seasons", type=int, nargs="+", default=list(SEASONS))
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--skip-players", action="store_true", help="Fetch only game logs (biographies remain required for normalization)")
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    downloader = CachedDownloader(min_interval=args.interval)
    failures = []
    for season in args.seasons:
        try:
            fetch_game_logs(season, source=args.source, raw_dir=args.raw_dir, downloader=downloader)
            if not args.skip_players and args.source == "sportsdataverse":
                fetch_roster(season, raw_dir=args.raw_dir, downloader=downloader)
        except (requests.RequestException, ValueError, OSError) as exc:
            LOGGER.error("Season %s failed: %s", season, exc)
            failures.append(season)
    if not args.skip_players and args.source == "nba":
        try:
            fetch_player_directory(raw_dir=args.raw_dir, downloader=downloader)
            ids = set()
            for season in args.seasons:
                path = args.raw_dir / "nba" / f"game_logs_{season}.json"
                if path.exists():
                    ids.update(load_game_logs(path)["PLAYER_ID"].astype(str))
            for player_id in sorted(ids):
                fetch_player_info(player_id, raw_dir=args.raw_dir, downloader=downloader)
        except (requests.RequestException, ValueError, OSError) as exc:
            LOGGER.error("Player biographies failed: %s", exc)
            failures.append("players")
    read_adp_csvs(raw_dir=args.raw_dir, seasons=args.seasons)
    if failures:
        LOGGER.error("Required downloads incomplete: %s; rerun to resume valid caches", failures)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
