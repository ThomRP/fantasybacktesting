"""NBA seasons use the ending year: 2015 means 2014–15."""
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw'
PROCESSED_DIR = ROOT / 'data' / 'processed'
SEASONS = tuple(range(2015, 2027))
SCORING = {
    'fgm': 2.0, 'fga': -1.0, 'ftm': 1.0, 'fta': -1.0, 'fg3m': 1.0,
    'reb': 1.0, 'ast': 2.0, 'stl': 4.0, 'blk': 4.0, 'tov': -2.0, 'pts': 1.0,
}
BOX_STATS = (*SCORING, 'fg3a', 'oreb', 'dreb', 'pf', 'plus_minus')
SHOOTING_RATES = ('fg_pct', 'fg3_pct', 'ft_pct')
ALL_STAR_DATES = dict(zip(SEASONS, map(date.fromisoformat, [
    '2015-02-15', '2016-02-14', '2017-02-19', '2018-02-18',
    '2019-02-17', '2020-02-16', '2021-03-07', '2022-02-20',
    '2023-02-19', '2024-02-18', '2025-02-16', '2026-02-15',
])))
SEASON_OPENERS = dict(zip(SEASONS, map(date.fromisoformat, [
    '2014-10-28', '2015-10-27', '2016-10-25', '2017-10-17',
    '2018-10-16', '2019-10-22', '2020-12-22', '2021-10-19',
    '2022-10-18', '2023-10-24', '2024-10-22', '2025-10-21',
])))
EXPECTED_GAMES = {s: 1230 for s in SEASONS}
EXPECTED_GAMES.update({2020: 1059, 2021: 1080})


def season_label(season):
    return f'{season - 1}-{str(season)[-2:]}'
