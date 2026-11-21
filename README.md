# Fantasy basketball data downloader

Download historical FantasyPros average draft position (ADP) and ESPN NBA game data for the same season. The files are inputs for backtesting how players performed relative to where they were drafted.

## Install and run

Use Python 3.10 or later and run the commands from the repository root. Set up the virtual environment once, then choose a download command for your OS.

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```powershell
# FantasyPros year=2019 if the page is for the 2019–20 draft
.\.venv\Scripts\python.exe download.py 2019 --adp-year-is start

# FantasyPros year=2019 if the page is for the 2018–19 draft
.\.venv\Scripts\python.exe download.py 2018 --adp-year-is end

# FantasyPros years 2015 through 2025 if the page uses the start year
.\.venv\Scripts\python.exe download.py 2015 2025 --adp-year-is start

# 2018–19 game data with play-by-play
.\.venv\Scripts\python.exe download.py 2018 --games-only --play-by-play
```

### macOS and Linux

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

```sh
# FantasyPros year=2019 if the page is for the 2019–20 draft
.venv/bin/python download.py 2019 --adp-year-is start

# FantasyPros year=2019 if the page is for the 2018–19 draft
.venv/bin/python download.py 2018 --adp-year-is end

# FantasyPros years 2015 through 2025 if the page uses the start year
.venv/bin/python download.py 2015 2025 --adp-year-is start

# 2018–19 game data with play-by-play
.venv/bin/python download.py 2018 --games-only --play-by-play
```

## Download options

The positional year is the **start of the NBA season**. Add a second starting year to download an inclusive range. Check the season label on the FantasyPros page and use `--adp-year-is start` or `--adp-year-is end` to match its `year` parameter.

Each season needs a populated ADP table before the downloader requests its game files. Use `--games-only` to skip the ADP lookup. Add `--play-by-play` for event-level data, which can add several hundred MB per season. The downloader reports failures by season and exits with a nonzero status if any season fails.

## Files

Each season goes into its own folder, such as `data/2018-19/`. Use `--output` to choose a different root.

| File | Contents |
| --- | --- |
| `adp.csv` | FantasyPros table columns, requested year, player name, and FantasyPros player URL when available |
| `games.csv` | ESPN schedule, game dates, teams, scores, and game status |
| `player_box_scores.csv` | Player stats by game, including minutes, shooting, rebounds, assists, steals, blocks, and turnovers |
| `team_box_scores.csv` | Team stats by game |
| `play_by_play.csv` | Optional game events |
| `game_json/<game_id>.json` | Original ESPN game records fetched when a requested CSV is missing a completed game |

Join the ESPN files on `game_id`. ESPN identifies players by `athlete_id`. The ADP file uses FantasyPros player names and URLs, so a future backtest will need a crosswalk between the two sources.

The downloader prints how many completed games each CSV covers. If a requested feed is absent from both the CSV and the original game record, it reports the gap and exits with a nonzero status. Regular season, postseason, and any other game types present in the source stay in the files. Filter ESPN `season_type` to `2` for regular-season scoring.

Sources are [FantasyPros NBA ADP](https://www.fantasypros.com/nba/adp/overall.php), the [SportsDataverse ESPN archive](https://github.com/sportsdataverse/sportsdataverse-data/releases), and the [original ESPN game records](https://github.com/sportsdataverse/hoopR-nba-raw).
