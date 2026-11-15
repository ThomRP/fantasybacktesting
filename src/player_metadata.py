"""Optional static master-player metadata from explicit NBA/ESPN ID bridges.

These retrospective ESPN bio snapshots are suitable for identity/display only.
Their position labels and current-team fields must never become historical
strategy features. NBA roster biographies remain the preferred source.

Sources and join-key documentation:
https://github.com/sportsdataverse/hoopR-nba-data/blob/main/docs/datasets/player_crosswalk.md
https://github.com/sportsdataverse/hoopR-nba-data/blob/main/docs/datasets/player_core.md

The upstream docs do not calibrate confidence numerically. Consequently this
adapter accepts only exact_name/1.0 bridges, independently checks normalized
names, and rejects non-bijective IDs and conflicting birthdays. No fuzzy
matching or inferred birth dates are used.
"""
from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path
import re
import unicodedata

import pandas as pd

from .config import RAW_DIR, SEASONS
from .ingest import CachedDownloader, MIRROR

LOGGER = logging.getLogger(__name__)
COLUMNS = ["player_id", "player", "position", "birth_date", "season"]


def _name_key(value) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    words = re.sub(r"[^a-z0-9\s]", "", text).split()
    while words and words[-1] in {"jr", "sr", "ii", "iii", "iv", "v"}:
        words.pop()
    return " ".join(words)


def _ids(values: pd.Series) -> pd.Series:
    result = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return result.where(result.str.fullmatch(r"[1-9][0-9]*", na=False))


def fetch_player_metadata(season: int, *, raw_dir: Path = RAW_DIR,
                          downloader: CachedDownloader | None = None) -> tuple[Path, Path]:
    downloader = downloader or CachedDownloader()
    specifications = [
        ("nba_crosswalk", "nba_player_crosswalk", "crosswalk", {
            "season", "nba_player_id", "nba_player_name", "espn_athlete_id",
            "match_method", "match_confidence"}),
        ("espn_nba_player_core", "player_core", "core", {
            "season", "athlete_id", "full_name", "date_of_birth", "position_abbreviation"}),
    ]
    paths = []
    for tag, stem, local_name, required in specifications:
        def validate(payload: bytes, required=required) -> None:
            frame = pd.read_parquet(io.BytesIO(payload))
            if frame.empty or not required.issubset(frame.columns):
                raise ValueError("Supplemental player metadata schema is incomplete")
            if not pd.to_numeric(frame["season"], errors="raise").eq(season).all():
                raise ValueError("Supplemental metadata has unexpected season labels")
        paths.append(downloader.get(
            f"{MIRROR}/{tag}/{stem}_{season}.parquet",
            Path(raw_dir) / "player_metadata" / f"{local_name}_{season}.parquet",
            validator=validate,
        ))
    return tuple(paths)


def load_supplemental_bios(raw_dir: Path = RAW_DIR, seasons=SEASONS) -> pd.DataFrame:
    """Read existing caches, returning only unambiguous roster-shaped records.

    Missing caches are reported and skipped because this enrichment is optional.
    Disagreement across years on either ID mapping or birthday rejects that
    identity entirely. Preserve existing NBA roster values when using output.
    """
    crosswalks, cores = [], []
    for season in seasons:
        base = Path(raw_dir) / "player_metadata"
        xp, cp = base / f"crosswalk_{season}.parquet", base / f"core_{season}.parquet"
        if not xp.exists() or not cp.exists():
            LOGGER.warning("Supplemental metadata cache missing for %s", season)
            continue
        x, c = pd.read_parquet(xp), pd.read_parquet(cp)
        x = x.copy()
        x["nba_player_id"], x["espn_athlete_id"] = _ids(x.nba_player_id), _ids(x.espn_athlete_id)
        x = x.loc[x.nba_player_id.notna() & x.espn_athlete_id.notna()]
        c = c.copy()
        c["athlete_id"] = _ids(c.athlete_id)
        c = c.loc[c.athlete_id.notna()]
        crosswalks.append(x)
        cores.append(c)
    if not crosswalks:
        return pd.DataFrame(columns=COLUMNS)
    x, c = pd.concat(crosswalks, ignore_index=True), pd.concat(cores, ignore_index=True)
    bad_nba = set(x.groupby("nba_player_id").espn_athlete_id.nunique().loc[lambda s: s > 1].index)
    bad_espn = set(x.groupby("espn_athlete_id").nba_player_id.nunique().loc[lambda s: s > 1].index)
    accepted = x.match_method.eq("exact_name") & pd.to_numeric(x.match_confidence, errors="coerce").eq(1.0)
    x = x.loc[accepted & ~x.nba_player_id.isin(bad_nba) & ~x.espn_athlete_id.isin(bad_espn)].copy()
    # Preserve the published birthday calendar date; timestamp timezone conversion
    # must not shift a date of birth by a day.
    dates = c.date_of_birth.astype("string").str.slice(0, 10)
    c["birth_date"] = pd.to_datetime(dates, format="%Y-%m-%d", errors="coerce").dt.date
    c["_name_key"] = c.full_name.map(_name_key)
    ambiguous_dates = set(c.groupby("athlete_id").birth_date.nunique().loc[lambda s: s > 1].index)
    c = c.loc[~c.athlete_id.isin(ambiguous_dates) & c.birth_date.notna()].copy()
    # A core ID must have one normalized identity per year as well.
    ambiguous_names = c.groupby(["season", "athlete_id"])._name_key.nunique()
    ambiguous_names = set(ambiguous_names.loc[lambda s: s > 1].index)
    if ambiguous_names:
        c = c.loc[~pd.MultiIndex.from_frame(c[["season", "athlete_id"]]).isin(ambiguous_names)]
    c = c.drop_duplicates(["season", "athlete_id"])
    merged = x.merge(c, left_on=["season", "espn_athlete_id"], right_on=["season", "athlete_id"],
                     how="inner", validate="many_to_one", suffixes=("_bridge", "_core"))
    exact_names = merged.nba_player_name.map(_name_key).eq(merged._name_key) & merged._name_key.ne("")
    rejected_names = int((~exact_names).sum())
    merged = merged.loc[exact_names].copy()
    merged["player_id"], merged["player"] = merged.nba_player_id, merged.nba_player_name
    merged["position"] = merged.position_abbreviation.astype("string").str.upper()
    merged.loc[~merged.position.isin(["PG", "SG", "SF", "PF", "C", "G", "F", "G-F", "F-C"]), "position"] = pd.NA
    result = merged[COLUMNS].drop_duplicates(["season", "player_id"]).reset_index(drop=True)
    LOGGER.info("Supplemental bios: %d players; rejected %d NBA ID conflicts, %d ESPN ID conflicts, "
                "%d DOB conflicts and %d non-exact names", result.player_id.nunique(), len(bad_nba),
                len(bad_espn), len(ambiguous_dates), rejected_names)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--seasons", type=int, nargs="+", default=list(SEASONS))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    downloader = CachedDownloader()
    for season in args.seasons:
        fetch_player_metadata(season, raw_dir=args.raw_dir, downloader=downloader)
    load_supplemental_bios(args.raw_dir, args.seasons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
