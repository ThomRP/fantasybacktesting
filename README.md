# Fantasy basketball data downloader

Download historical FantasyPros average draft position (ADP) and ESPN NBA game data for the same season. The files are inputs for backtesting how players performed relative to where they were drafted.

## Install

From the repository root, using Python 3.10 or later:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Download

The positional year is the **start of the NBA season**. Check the season label on the FantasyPros page and use `--adp-year-is` to say how its `year` parameter maps to that season. Both commands below request FantasyPros `year=2019`:

```sh
# If it is the ADP for 2019–20
.venv/bin/python download.py 2019 --adp-year-is start

# If it is the ADP for 2018–19
.venv/bin/python download.py 2018 --adp-year-is end
```

Add a second starting year for an inclusive range. For example, use `2015 2025 --adp-year-is start` to request FantasyPros years 2015 through 2025 under the start-year convention. Each season needs a populated ADP table before its game files are requested. Failures are printed by season, and a run with any failures exits with a nonzero status.

Use `--games-only` to fetch the game files independently of FantasyPros. Add `--play-by-play` for event-level data, which can add several hundred MB per season. For example:

```sh
.venv/bin/python download.py 2018 --games-only --play-by-play
```

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
