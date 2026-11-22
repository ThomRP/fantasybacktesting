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

## All-Star break backtests

Run the backtest from the repository root after downloading consecutive completed seasons. If you downloaded to a different folder, pass its path with `--data`.

### Post-break surge

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\python.exe backtest.py
```

**macOS and Linux**

```sh
.venv/bin/python backtest.py
```

Both backtests apply the league scoring in `backtest_common.py` to games that count in the regular-season standings. A player qualifies with at least one game on each side of the All-Star break and positive pre-break fantasy points per game. In `backtest.py`, a rise of 10% or more flags a surge.

In the following season, players beat ADP if their rank by total regular-season fantasy points is better than their average draft pick. A matched player who misses the entire season counts as a miss. The script prints how many ADP entries it matches to unique ESPN players and compares hit rates for surging players and other eligible players. Find the individual results in `data/backtest_results.csv`.

#### Ages 23–29

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\python.exe backtest.py --ages 23 29
```

**macOS and Linux**

```sh
.venv/bin/python backtest.py --ages 23 29
```

The age range applies to both groups. Age is measured on the All-Star date of the season before the draft being tested, and next-season ranks still use the full NBA player pool. The first run fetches ESPN birth dates and caches them under `data/player_core/`. It reports missing birth dates and saves results with `age_at_break` to `data/backtest_results_ages_23_29.csv`. [ESPN player data archive](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_nba_player_core)

### Post-break decline

This backtest compares next-season ADP hit rates for players whose fantasy points per game fell by at least 10% after the All-Star break and other eligible players.

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\python.exe backtest_decline.py
```

**macOS and Linux**

```sh
.venv/bin/python backtest_decline.py
```

Results go to `data/backtest_decline_results.csv`. Add `--ages 23 29` to either command for the age-restricted run, which saves to `data/backtest_decline_results_ages_23_29.csv`.

## Second-year players

The rookie season is the first season a player appeared in a regular-season NBA game. The test compares their ADP in the immediately following season with their finish by total regular-season fantasy points. It prints hit rates for second-year players and other drafted players who played the prior season. A matched player who misses the entire next season counts as a miss.

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\python.exe backtest_rookies.py
```

**macOS and Linux**

```sh
.venv/bin/python backtest_rookies.py
```

Results go to `data/backtest_rookie_results.csv`. The `second_year` column identifies players in the rookie cohort. On the first run, the script fetches ESPN regular-season appearance history back to 1993–94 and caches player IDs in `data/player_history/`. [ESPN season statistics archive](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_nba_player_season_stats)

The test also prints how many second-year players beat ADP at each position and exact listed height. `data/backtest_rookie_summary.csv` contains the overall rate and both breakdowns with player counts. The player-level CSV includes `rookie_position` and `height_inches`.

Position comes from the most frequent listing in the player's rookie regular-season box scores, with their ESPN profile as a fallback. Broad listings such as `G` and `F` are reported as given. Heights come from ESPN's listed profiles. Missing values appear in an `Unknown` group and count toward the overall rate. [ESPN player data archive](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_nba_player_core)

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

Join the ESPN files on `game_id`. ESPN identifies players by `athlete_id`. The backtest matches FantasyPros players to unique ESPN IDs using normalized player names.

The downloader prints how many completed games each CSV covers. If a requested feed is absent from both the CSV and the original game record, it reports the gap and exits with a nonzero status. Regular season, postseason, and any other game types present in the source stay in the files. Filter ESPN `season_type` to `2` for regular-season scoring.

Sources are [FantasyPros NBA ADP](https://www.fantasypros.com/nba/adp/overall.php), the [SportsDataverse ESPN archive](https://github.com/sportsdataverse/sportsdataverse-data/releases), and the [original ESPN game records](https://github.com/sportsdataverse/hoopR-nba-raw).
