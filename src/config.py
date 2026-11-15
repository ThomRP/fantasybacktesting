"""Reproducible research universe. Season numbers refer to the END year."""
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SEASONS = tuple(range(2007, 2027))  # 2006–07 through 2025–26
HOLDOUT_SEASONS = tuple(range(2023, 2027))

# Transcribed from the supplied scoring image. 3PM is NBA's FG3M; TO is TOV.
SCORING = {
    "fgm": 2.0, "fga": -1.0, "ftm": 1.0, "fta": -1.0,
    "fg3m": 1.0, "reb": 1.0, "ast": 2.0, "stl": 4.0,
    "blk": 4.0, "tov": -2.0, "pts": 1.0,
}
STAT_COLUMNS = tuple(SCORING)
# Preseason ADP must be known strictly before the season's first regular game.
SEASON_OPENERS = dict(zip(SEASONS, map(date.fromisoformat, [
    "2006-10-31", "2007-10-30", "2008-10-28", "2009-10-27", "2010-10-26",
    "2011-12-25", "2012-10-30", "2013-10-29", "2014-10-28", "2015-10-27",
    "2016-10-25", "2017-10-17", "2018-10-16", "2019-10-22", "2020-12-22",
    "2021-10-19", "2022-10-18", "2023-10-24", "2024-10-22", "2025-10-21",
])))


def season_label(end_year: int) -> str:
    return f"{end_year - 1}-{str(end_year)[-2:]}"

# Actual regular-season games; cancellation/lockout/pandemic seasons differ.
EXPECTED_GAMES = {season: 1230 for season in SEASONS}
EXPECTED_GAMES.update({2012: 990, 2013: 1229, 2020: 1059, 2021: 1080})
