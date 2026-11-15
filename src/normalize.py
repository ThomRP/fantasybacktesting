"""Strict schemas, NBA identity resolution, scoring and as-of date queries.

Mutable master biography fields are identity aids, never historical strategy features.
ADP snapshots require a declared source and a verifiable preseason available_at date.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import hashlib
import logging
import os
from pathlib import Path
import re
import tempfile
import unicodedata

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import (RAW_DIR, PROCESSED_DIR, SEASONS, SEASON_OPENERS,
                        SCORING, STAT_COLUMNS, EXPECTED_GAMES)

LOG = logging.getLogger(__name__)
PLAYER_SCHEMA = pa.schema([
    ("canonical_player_id", pa.string()), ("full_name", pa.string()),
    ("normalized_name", pa.string()), ("primary_position", pa.string()),
    ("birth_date", pa.date32()),
])
GAME_SCHEMA = pa.schema([
    ("season", pa.int64()), ("game_date", pa.date32()),
    ("canonical_player_id", pa.string()), ("team", pa.string()),
    ("opponent", pa.string()), ("minutes_or_snaps", pa.float64()),
    *[(stat, pa.float64()) for stat in STAT_COLUMNS],
    ("fantasy_points", pa.float64()),
])
ADP_SCHEMA = pa.schema([
    ("season", pa.int64()), ("canonical_player_id", pa.string()),
    ("adp", pa.float64()), ("adp_rank", pa.int64()),
    ("position_rank", pa.int64()), ("available_at", pa.date32()),
    ("source", pa.string()),
])


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"(?:\s+(?:jr|sr|ii|iii|iv|v))+$", "", text).strip()


def nba_id(value) -> str:
    text = str(value).removeprefix("nba:")
    if not re.fullmatch(r"\d+(?:\.0)?", text) or int(float(text)) <= 0:
        raise ValueError(f"Invalid NBA player ID: {value!r}")
    return f"nba:{int(float(text))}"


def _required(frame, columns):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _dates(values, label):
    parsed = pd.to_datetime(values, errors="raise", format="mixed")
    if parsed.isna().any():
        raise ValueError(f"{label} cannot be missing")
    return parsed.dt.date


def _integers(values, label):
    values = pd.to_numeric(values, errors="raise")
    if values.isna().any() or not np.isfinite(values).all() or (values % 1 != 0).any():
        raise ValueError(f"{label} must contain finite integers")
    return values.astype("int64")


def build_players(rosters: pd.DataFrame, logs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Consolidate roster seasons by NBA ID; never by team or name alone.

    Select latest known non-null biography for entity resolution. Birth dates must
    agree across seasons. Missing bio remains null/UNKNOWN and is reported.
    """
    rosters = rosters.rename(columns=str.lower).copy()
    _required(rosters, ["player_id", "player", "position", "birth_date", "season"])
    rosters["canonical_player_id"] = rosters.player_id.map(nba_id)
    rosters["birth_date"] = pd.to_datetime(rosters.birth_date, errors="raise", format="mixed").dt.date
    if (rosters.groupby("canonical_player_id").birth_date.nunique() > 1).any():
        raise ValueError("Conflicting birth dates for the same NBA ID; resolve raw source conflict")
    rows = []
    for player_id, group in rosters.sort_values("season").groupby("canonical_player_id", sort=True):
        def last_nonnull(column, default=None):
            values = group[column].dropna()
            values = values[values.astype(str).str.strip().ne("")]
            return values.iloc[-1] if len(values) else default
        rows.append({"canonical_player_id": player_id, "full_name": last_nonnull("player"),
                     "primary_position": last_nonnull("position", "UNKNOWN"),
                     "birth_date": last_nonnull("birth_date")})
    result = pd.DataFrame(rows, columns=["canonical_player_id", "full_name", "primary_position", "birth_date"])
    if logs is not None and len(logs):
        raw = logs.rename(columns=str.lower).copy()
        _required(raw, ["player_id", "player_name"])
        raw["canonical_player_id"] = raw.player_id.map(nba_id)
        missing = raw[~raw.canonical_player_id.isin(result.canonical_player_id)]
        missing = missing.drop_duplicates("canonical_player_id", keep="last")
        if len(missing):
            LOG.warning("%d game-log players lack roster biography; birth_date=null, position=UNKNOWN", len(missing))
            result = pd.concat([result, pd.DataFrame({"canonical_player_id": missing.canonical_player_id,
                "full_name": missing.player_name, "primary_position": "UNKNOWN", "birth_date": None})], ignore_index=True)
    if result.full_name.isna().any() or result.full_name.astype(str).str.strip().eq("").any():
        raise ValueError("Player names cannot be missing")
    result["normalized_name"] = result.full_name.map(normalize_name)
    if result.normalized_name.eq("").any():
        raise ValueError("Normalized player names cannot be empty")
    return result[PLAYER_SCHEMA.names].sort_values("canonical_player_id").reset_index(drop=True)


class PlayerResolver:
    """Exact NBA ID first, explicit aliases second, unambiguous names last.

    aliases is an optional DataFrame(source, source_player_id, canonical_player_id).
    Name collisions raise instead of guessing. A birth date can disambiguate.
    """
    def __init__(self, players: pd.DataFrame, aliases: pd.DataFrame | None = None):
        _required(players, PLAYER_SCHEMA.names)
        if players.canonical_player_id.duplicated().any():
            raise ValueError("Duplicate canonical player IDs")
        self.players = players.copy()
        self.ids = set(players.canonical_player_id)
        self.aliases = {}
        if aliases is not None:
            _required(aliases, ["source", "source_player_id", "canonical_player_id"])
            if aliases.duplicated(["source", "source_player_id"]).any():
                raise ValueError("Duplicate cross-source aliases")
            if not set(aliases.canonical_player_id) <= self.ids:
                raise ValueError("Alias points to an unknown player")
            self.aliases = {(str(r.source), str(r.source_player_id)): r.canonical_player_id for r in aliases.itertuples()}

    def resolve(self, *, canonical_player_id=None, nba_player_id=None, full_name=None,
                birth_date=None, source=None, source_player_id=None):
        candidate = None
        if pd.notna(canonical_player_id):
            candidate = str(canonical_player_id)
        elif pd.notna(nba_player_id):
            candidate = nba_id(nba_player_id)
        elif pd.notna(source_player_id):
            candidate = self.aliases.get((str(source), str(source_player_id)))
            if candidate is None:
                raise ValueError("Unmapped source_player_id; supply an explicit alias")
        if candidate is not None:
            if candidate not in self.ids:
                raise ValueError(f"Unknown player ID {candidate}")
            return candidate
        if pd.isna(full_name):
            raise ValueError("Player requires an ID or unambiguous name")
        matches = self.players[self.players.normalized_name.eq(normalize_name(full_name))]
        if pd.notna(birth_date):
            matches = matches[matches.birth_date.eq(pd.Timestamp(birth_date).date())]
        if len(matches) != 1:
            raise ValueError(f"Unresolved or ambiguous player name {full_name!r}")
        return matches.canonical_player_id.iloc[0]


def calculate_fantasy_points(frame: pd.DataFrame, scoring=None) -> pd.Series:
    scoring = SCORING if scoring is None else scoring
    if not scoring or not all(isinstance(v, (float, int)) and np.isfinite(v) for v in scoring.values()):
        raise ValueError("Scoring weights must be finite numbers")
    _required(frame, scoring)
    stats = frame[list(scoring)].apply(pd.to_numeric, errors="raise")
    if stats.isna().any().any() or not np.isfinite(stats.to_numpy()).all():
        raise ValueError("Scoring stats cannot be missing/nonfinite")
    return stats.mul(pd.Series(scoring)).sum(axis=1).astype(float)


def _minutes(value):
    if isinstance(value, str) and ":" in value:
        mins, secs = value.split(":")
        return float(mins) + float(secs) / 60
    return float(value)


def validate_season_coverage(raw: pd.DataFrame, season: int) -> dict:
    """Reject partial downloads before calling a season complete.

    Checks the actual schedule count (including shortened seasons), two teams
    per game and at least five appearances per team. This cannot independently
    certify every upstream stat but catches truncated tables and team-only data.
    """
    frame = raw.rename(columns=str.lower).copy()
    _required(frame, ["game_id", "player_id", "team_abbreviation"])
    if "season_type" in frame:
        kinds = frame.season_type.astype(str).str.lower().str.replace(" ", "-", regex=False)
        frame = frame.loc[kinds.eq("regular-season")]
    if frame.empty or frame[["game_id", "player_id", "team_abbreviation"]].isna().any().any():
        raise ValueError("Missing season game/player/team keys")
    count = frame.game_id.nunique()
    if count != EXPECTED_GAMES[season]:
        raise ValueError(f"Incomplete season {season}: {count} games, expected {EXPECTED_GAMES[season]}")
    if not frame.groupby("game_id").team_abbreviation.nunique().eq(2).all():
        raise ValueError("Each NBA game must contain exactly two teams")
    if frame.groupby(["game_id", "team_abbreviation"]).player_id.nunique().lt(5).any():
        raise ValueError("Each NBA team game must contain at least five player appearances")
    return {"source_game_count": int(count), "source_player_rows": len(frame)}


def _verify_cache(path: Path):
    meta = path.with_suffix(path.suffix + ".metadata.json")
    if not meta.exists() or json.loads(meta.read_text()).get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError(f"Missing or mismatched download checksum: {path}; rerun ingest to diagnose")


def normalize_game_logs(raw: pd.DataFrame, players: pd.DataFrame, *, scoring=None) -> pd.DataFrame:
    frame = raw.rename(columns=str.lower).copy()
    _required(frame, ["season", "game_date", "player_id", "team_abbreviation", "matchup", "min", *STAT_COLUMNS])
    if "season_type" in frame:
        kinds = frame.season_type.astype(str).str.lower().str.replace(" ", "-", regex=False)
        if not kinds.isin(["regular-season", "playoffs", "pre-season", "preseason", "all-star"]).all():
            raise ValueError("Unrecognized season_type")
        frame = frame.loc[kinds.eq("regular-season")].copy()
    frame["season"] = _integers(frame.season, "season")
    if not set(frame.season) <= set(SEASONS):
        raise ValueError("Games outside configured completed-season universe")
    frame["game_date"] = _dates(frame.game_date, "game_date")
    bounds = frame.apply(lambda r: SEASON_OPENERS[r.season] <= r.game_date <= date(r.season, 10, 15), axis=1)
    if not bounds.all():
        raise ValueError("game_date outside the declared NBA season")
    frame["canonical_player_id"] = frame.player_id.map(nba_id)
    if not set(frame.canonical_player_id) <= set(players.canonical_player_id):
        raise ValueError("Game logs contain unresolved player IDs")
    frame["team"] = frame.team_abbreviation.astype(str)
    matchup = frame.matchup.str.extract(r"^([A-Z]{2,3})\s+(?:vs\.?|@)\s+([A-Z]{2,3})$")
    if matchup.isna().any().any() or not matchup[0].eq(frame.team).all():
        raise ValueError("Malformed team/opponent matchup")
    frame["opponent"] = matchup[1]
    frame["minutes_or_snaps"] = frame["min"].map(_minutes)
    if not np.isfinite(frame.minutes_or_snaps).all() or frame.minutes_or_snaps.lt(0).any():
        raise ValueError("Invalid playing time")
    for stat in STAT_COLUMNS:
        frame[stat] = pd.to_numeric(frame[stat], errors="raise").astype(float)
        if frame[stat].isna().any() or not np.isfinite(frame[stat]).all() or frame[stat].lt(0).any():
            raise ValueError(f"Invalid box-score stat {stat}")
    # NBA player logs contain appearances, not DNP placeholders. MIN can round
    # a real subminute appearance to zero; keep it for unbiased games/PPG.
    frame["fantasy_points"] = calculate_fantasy_points(frame, scoring)
    keys = ["season", "game_date", "canonical_player_id"]
    if "game_id" in frame and frame.groupby(keys).game_id.nunique().gt(1).any():
        raise ValueError("Conflicting duplicate player/date keys span distinct source game IDs")
    result = frame[GAME_SCHEMA.names].drop_duplicates()
    keys = ["season", "game_date", "canonical_player_id"]
    if result.duplicated(keys).any():
        examples = result.loc[result.duplicated(keys, keep=False), keys].head().to_dict("records")
        raise ValueError(f"Conflicting duplicate player/date keys; inspect raw game IDs: {examples}")
    return result.sort_values(keys).reset_index(drop=True)


def filter_game_logs(game_logs: pd.DataFrame, *, as_of, start=None, end=None) -> pd.DataFrame:
    """Knowledge cutoff is exclusive; windows are [start, min(end, as_of)).

    A date-only source cannot promise intraday availability. as_of must be a date
    or midnight timestamp; the latest eligible game is the preceding calendar day.
    """
    _required(game_logs, ["game_date"])
    cutoff = pd.Timestamp(as_of)
    if pd.isna(cutoff) or cutoff != cutoff.normalize() or cutoff.tzinfo is not None:
        raise ValueError("as_of must be an unzoned date/midnight (day-granularity data)")
    dates = _dates(game_logs.game_date, "game_date")
    mask = dates.lt(cutoff.date())
    if start is not None:
        mask &= dates.ge(pd.Timestamp(start).date())
    if end is not None:
        mask &= dates.lt(pd.Timestamp(end).date())
    return game_logs.loc[mask].copy()


def normalize_adp(raw: pd.DataFrame, players: pd.DataFrame, *, aliases=None) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=ADP_SCHEMA.names)
    frame = raw.copy()
    _required(frame, ["season", "adp", "position_rank", "available_at", "source"])
    frame["season"] = _integers(frame.season, "season")
    if not set(frame.season) <= set(SEASONS):
        raise ValueError("ADP outside configured season universe")
    frame["available_at"] = _dates(frame.available_at, "available_at")
    if not frame.apply(lambda r: date(r.season - 1, 1, 1) <= r.available_at < SEASON_OPENERS[r.season], axis=1).all():
        raise ValueError("ADP available_at must precede season opener and belong to the draft cycle")
    if frame.source.isna().any() or frame.source.astype(str).str.strip().eq("").any():
        raise ValueError("ADP source provenance required")
    if (frame.groupby("season").source.nunique() > 1).any():
        raise ValueError("Do not combine different ADP markets within a season")
    # Take an entire latest snapshot, never a player-wise mix of dates.
    latest = frame.groupby("season").available_at.transform("max")
    frame = frame[frame.available_at.eq(latest)].copy()
    resolver = PlayerResolver(players, aliases)
    fields = ["canonical_player_id", "nba_player_id", "full_name", "birth_date", "source", "source_player_id"]
    frame["canonical_player_id"] = [resolver.resolve(**{k: r.get(k) for k in fields}) for r in frame.to_dict("records")]
    if frame.duplicated(["season", "canonical_player_id"]).any():
        raise ValueError("Duplicate ADP player/season keys")
    frame["adp"] = pd.to_numeric(frame.adp, errors="raise").astype(float)
    if not np.isfinite(frame.adp).all() or frame.adp.le(0).any():
        raise ValueError("ADP must be finite and positive")
    if "adp_rank" not in frame:
        frame = frame.sort_values(["season", "adp", "canonical_player_id"])
        frame["adp_rank"] = frame.groupby("season").cumcount() + 1
    for column in ["adp_rank", "position_rank"]:
        frame[column] = _integers(frame[column], column)
        if frame[column].le(0).any():
            raise ValueError(f"{column} must be positive")
    if frame.duplicated(["season", "adp_rank"]).any():
        raise ValueError("ADP ranks must be unique within each season")
    ordered = frame.sort_values(["season", "adp_rank"])
    if ordered.groupby("season").adp.diff().lt(0).any():
        raise ValueError("ADP prices must be nondecreasing in adp_rank order")
    return ordered[ADP_SCHEMA.names].reset_index(drop=True)


def write_parquet(frame: pd.DataFrame, path: Path, schema: pa.Schema):
    """Strict Arrow types, date32 dates, atomic replacement on the same filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame[schema.names], schema=schema, preserve_index=False, safe=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--seasons", type=int, nargs="+", default=SEASONS)
    parser.add_argument("--source", choices=["nba", "sportsdataverse"], default="sportsdataverse")
    parser.add_argument("--enrich-bios", action="store_true", help="Use previously downloaded optional exact-ID ESPN metadata")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    from src.ingest import load_game_logs, load_roster, load_nba_table, read_adp_csvs
    raw_logs, raw_rosters = [], []
    season_coverage = {}
    for season in args.seasons:
        suffix = "parquet" if args.source == "sportsdataverse" else "json"
        log_path = args.raw_dir / args.source / f"game_logs_{season}.{suffix}"
        _verify_cache(log_path)
        raw = load_game_logs(log_path)
        season_coverage[str(season)] = validate_season_coverage(raw, season)
        raw.columns = raw.columns.str.lower()
        raw["season"] = season
        raw_logs.append(raw)
        if args.source == "sportsdataverse":
            roster_path = args.raw_dir / "sportsdataverse" / f"rosters_{season}.parquet"
            _verify_cache(roster_path)
            raw_rosters.append(load_roster(roster_path))
        else:
            bios = []
            for player_id in raw.player_id.unique():
                bio = load_nba_table(args.raw_dir / "nba" / "players" / f"{int(player_id)}.json")
                bio.columns = bio.columns.str.lower()
                for row in bio.to_dict("records"):
                    bios.append({"player_id": row["person_id"], "player": row["display_first_last"],
                                 "position": row["position"], "birth_date": row["birthdate"], "season": season})
            raw_rosters.append(pd.DataFrame(bios))
    raw_logs = pd.concat(raw_logs, ignore_index=True)
    rosters = pd.concat(raw_rosters, ignore_index=True)
    supplemental_count = 0
    if args.enrich_bios:
        from src.player_metadata import load_supplemental_bios
        for season in args.seasons:
            for kind in ["crosswalk", "core"]:
                _verify_cache(args.raw_dir / "player_metadata" / f"{kind}_{season}.parquet")
        supplemental = load_supplemental_bios(args.raw_dir, args.seasons)
        # Preserve all preferred NBA roster values, names and positions. Only
        # fill log identities wholly absent from rosters, with matching names.
        known = set(rosters.player_id.map(nba_id))
        log_keys = raw_logs.assign(_id=raw_logs.player_id.map(nba_id),
                                   _name=raw_logs.player_name.map(normalize_name))
        allowed_names = set(zip(log_keys["_id"], log_keys["_name"]))
        keep = supplemental.apply(lambda r: nba_id(r.player_id) not in known and
            (nba_id(r.player_id), normalize_name(r.player)) in allowed_names, axis=1)
        supplemental = supplemental.loc[keep]
        supplemental_count = supplemental.player_id.nunique()
        rosters = pd.concat([rosters, supplemental], ignore_index=True)
    players = build_players(rosters, raw_logs)
    # This game's original-date box combines the December game and March replay.
    # Quarantine it in full: individual date corrections would fabricate timing.
    quarantine_mask = raw_logs.season.eq(2008) & raw_logs.game_id.astype(str).str.zfill(10).eq("0020700367")
    quarantined = raw_logs.loc[quarantine_mask].copy()
    if len(quarantined):
        LOG.warning("Quarantining %d replay-game rows; 2008 is incomplete for outcome grading", len(quarantined))
    games = normalize_game_logs(raw_logs.loc[~quarantine_mask], players)
    adp = normalize_adp(read_adp_csvs(raw_dir=args.raw_dir, seasons=args.seasons), players)
    report = {"seasons": sorted(games.season.unique().tolist()), "players": len(players), "game_logs": len(games),
              "adp": len(adp), "missing_adp_seasons": sorted(set(args.seasons) - set(adp.season)),
              "missing_birth_date": int(players.birth_date.isna().sum()),
              "unknown_position": int(players.primary_position.eq("UNKNOWN").sum()), "scoring": SCORING,
              "source": args.source, "supplemental_bio_players": int(supplemental_count), "season_coverage": season_coverage, "quarantined_rows": len(quarantined),
              "completed_seasons": sorted(set(games.season) - ({2008} if len(quarantined) else set())),
              "incomplete_seasons": {"2008": "Replay game 0020700367 has mixed December/March event dates"} if len(quarantined) else {}}
    for frame, name, schema in [(players, "players", PLAYER_SCHEMA), (games, "game_logs", GAME_SCHEMA), (adp, "adp", ADP_SCHEMA)]:
        write_parquet(frame, args.processed_dir / f"{name}.parquet", schema)
    if len(quarantined):
        quarantined.to_parquet(args.processed_dir / "quarantine.parquet", index=False)
    report["processed_sha256"] = {name: hashlib.sha256((args.processed_dir / f"{name}.parquet").read_bytes()).hexdigest()
                                  for name in ["players", "game_logs", "adp"]}
    (args.processed_dir / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
