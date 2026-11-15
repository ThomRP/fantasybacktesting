# NBA fantasy backtesting

Two datasets, covering **2014–15 through 2025–26**:

- `data/processed/game_logs.parquet`: every regular-season player appearance, exact date, NBA ID, name, teams, minutes, box-score stats and your fantasy points.
- `data/processed/adp.parquet`: **FantasyPros consensus AVG**, player identity, derived ADP rank, the site's original rank, and source timing.

`season` is the ending year: **2015 = 2014–15**. Current data contains **307,145 game rows and 2,787 ADP rows**. Raw downloads are cached, checksummed and reused. No biographies, draft simulator, database or UI.

## Run in Windows PowerShell

If your virtual environment is already active (`(.venv)` in the prompt), run these from the repository folder containing `requirements.lock`, one line at a time:

```powershell
git pull --ff-only
python -m pip install -r requirements.lock
python -m src.data
python -m src.backtest
```

Stop if a command fails. `Test-Path .\requirements.lock` must print `True`; otherwise enter the repository folder first. The new data command reuses compatible existing NBA and page caches. The previous ingest/normalize/archive commands have been replaced.

For a fresh checkout without an active virtual environment:

```powershell
git clone https://github.com/ThomRP/fantasybacktesting.git
cd fantasybacktesting
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m src.data
.\.venv\Scripts\python.exe -m src.backtest
```

Python 3.12 is tested. On Linux/macOS, use `.venv/bin/python` instead of `.\.venv\Scripts\python.exe`.

## Example: do post–All-Star surges disappoint at the next draft?

The default backtest:

1. In season **S**, compare fantasy PPG before and after that season's All-Star game date. Require **10 appearances on each side**.
2. Flag players whose post-break PPG increased by **at least 20%**. This signal is known before the following season's draft.
3. In season **S+1**, compare their actual fantasy finish rank against their ADP rank. Example: ADP rank 30, actual finish 55 means **25 ranks lost**, hence underperformance.
4. Report results for surge players and the eligible **non-surge reference group**, by season and overall.

Totals rank all NBA players plus identified drafted players with no games (zero totals). PPG ranks require 10 appearances in the outcome season; zero-game players have undefined PPG. Both metrics are reported, and ties count as neither underperformance nor rank loss. Draft rank is derived by sorting **AVG**, while `source_rank` preserves FantasyPros' displayed Rank, which can disagree where providers are missing.

Outputs in `reports/`:

| File | Contents |
|---|---|
| `summary.json` | Overall surge/reference comparison and exclusions |
| `season_summary.csv` | Results by outcome season and group |
| `surge_players.csv` | Graded surge players with pre/post PPG, ADP and actual ranks |
| `exclusions.csv` | Players not graded, with reasons |
| `data_coverage.csv` | Game/ADP counts and timing availability by season |
| `unmatched_adp.csv` | Prices without a reliable NBA identity match |

The verified default run grades **216 surge player-seasons and 694 reference player-seasons**. Total-points underperformance rates are **50.9% vs 56.3%**; PPG rates are **55.0% vs 58.3%**. This sample does **not** show surge players underperforming more often. These are descriptive comparisons, not a causal result or proof of a profitable strategy; the groups differ in ADP and other characteristics. No threshold optimization is performed. The 2019–20 season includes the shutdown/bubble; results involving it need that context.

Change the example's definition explicitly:

```powershell
python -m src.backtest --surge-threshold 0.30 --min-games 15
```

## Historical ADP timing

All 12 FantasyPros year pages are downloaded and stored. The URL uses the **starting** year: `https://www.fantasypros.com/nba/adp/overall.php?year=2019` is 2019–20. Current team/position/injury labels on these historical pages are ignored.

Source sort-key dates use UTC calendar days; displayed local dates can be one day earlier. Both are recorded in `data/processed/metadata.json`. These are source-reported dates, not independently captured preseason snapshots. A rank depends on **every quote in its market**, so its availability is the latest contributing date across the entire season's consensus.

The default backtest requires strictly preseason ADP. Opening-day, late and undated snapshots stay in the dataset but are excluded from grading. In particular, the current **2024–25 consensus page has a Yahoo update from July 2025**, making that season's consensus ranks unsafe for a draft-time test. Strict usable outcome years in the current source are **2018, 2020, 2023, 2024 and 2026**. The first dataset year cannot be evaluated because its prior-season signal is outside the horizon.

For a separately labelled opening-day sensitivity run:

```powershell
python -m src.backtest --include-opening-day --output-dir reports\opening_day
```

This admits uncertain intraday availability on opening day. It still excludes post-opening and unknown snapshots. To recover strict results for excluded years, obtain a dated **pre-opening-day consensus ADP snapshot**, including the full ranked market and its source evidence. Do not substitute expert rankings or backdate the current page. No manual files are required to run the provided example.

## Data and code

```text
src/config.py       # Horizon, scoring, season-opening and All-Star dates
src/cache.py        # Resumable verified downloads, throttling and retries
src/data.py         # Download, score, resolve IDs; write the two tables
src/backtest.py     # The example backtest and reports
```

NBA game logs come from the [SportsDataverse NBA Stats mirror](https://github.com/sportsdataverse/hoopR-nba-stats-data) at `https://github.com/sportsdataverse/sportsdataverse-data/releases/download/nba_stats_player_game_logs/player_game_logs_{ENDING_YEAR}.parquet`. Only regular-season rows are used. Validation checks scheduled game counts, two teams per game, complete stats and unique player/date keys. Rounded zero-minute appearances still count as games played.

Player IDs are stable across trades. Exact names and unambiguous normalized aliases link the ADP table; unresolved prices are retained with null IDs and reported, not silently dropped or assigned invented zero-game outcomes. Optional reviewed aliases go in `data/player_aliases.csv` with columns `adp_name,canonical_player_id`; IDs must already exist in the game data.

Your scoring is unchanged: FGM **+2**, FGA **−1**, FTM **+1**, FTA **−1**, 3PM **+1**, REB **+1**, AST **+2**, STL **+4**, BLK **+4**, TO **−2**, PTS **+1**. The input archives are not certified to match a particular custom-league ADP format. Historical stat-correction vintages and intraday game-result availability are not available.

Run the focused tests:

```powershell
python -m pytest -q
```
