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

The rookie season is the first season a player appeared in a regular-season NBA game. The test compares their ADP in the immediately following season with their league-wide rank by regular-season fantasy points per game (FPPG). Every player with at least one appearance is eligible for the ranking, and `next_games` shows the sample size. Players with zero appearances are reported as unranked and are excluded from the per-game hit rate and value averages.

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe backtest_rookies.py
```

**macOS and Linux**

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python backtest_rookies.py
```

Results go to `data/backtest_rookie_results.csv`. The `second_year` column identifies players in the rookie cohort. On the first run, the script fetches ESPN regular-season appearance history back to 1993–94 and caches player IDs in `data/player_history/`. [ESPN season statistics archive](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_nba_player_season_stats)

### Opener and majority starters

This group includes sophomores who started their team's first regular-season game and started **more than half** of their own regular-season appearances. Find their aggregate results in the `opener_majority` row of `data/backtest_rookie_summary.csv` and individual results in `data/backtest_rookie_opener_majority.csv`.

Open `data/backtest_rookie_opener_majority.pdf` for three plots covering all qualifying players, those who beat ADP, and those who tied or missed ADP. Each plot shows slots gained or lost by FPPG rank, with each player's ADP, starts, games played, actual FPPG, and FPPG versus ADP.

### Other breakdowns

The test also prints how many second-year players beat ADP by lineup role, rookie position, and exact listed height. `data/backtest_rookie_summary.csv` contains the overall rate and each breakdown with player counts.

A regular starter started at least half of their second-year regular-season appearances. The test compares them with players who started fewer than half and reports players with no appearances or incomplete starter flags separately. The player-level CSV includes `next_starts`, `next_start_pct`, and `next_lineup_role`.

The `opening_lineup` rows compare all sophomores who started their NBA team's first regular-season game with the other sophomores. This includes opening starters who later lost their starting role. `opening_lineup_among_regular_starters` splits the season-long regular starters by the same opener status. Each row includes average ADP. Filter the player-level CSV by `second_year`, `next_lineup_role`, and `next_opener_role` to inspect the individual picks. The CSV records the team, game ID, and date for verified opener starts. A missing team lineup can leave a role `Unknown`.

The opener is an observed lineup. For a strategy used before opening night, use starter projections dated before the fantasy draft to select players.

The player-level CSV also includes `rookie_position` and `height_inches`. Position comes from the most frequent listing in the player's rookie regular-season box scores, with their ESPN profile as a fallback. Broad listings such as `G` and `F` are reported as given. Heights come from ESPN's listed profiles. Missing values appear in an `Unknown` group and count toward the overall rate. [ESPN player data archive](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_nba_player_core)

## Player profiles with K-means and PCA

This backtest groups drafted players who appeared in the prior regular season. It uses ADP, age at the prior All-Star break, games played, fantasy points per game, the post-break change in fantasy points per game, and start percentage. All game features come from the prior regular season.

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe backtest_clusters.py
```

**macOS and Linux**

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python backtest_clusters.py
```

The script learns four profiles from earlier drafts and checks their ADP hit rates in the latest two completed drafts. It fits feature scaling, missing-value medians, K-means, and PCA using the earlier drafts. The holdout evaluation compares ADP with league-wide regular-season FPPG rank. Players with zero games are reported as unranked and excluded from the hit rates and value averages. You can change the split or profile count with `--holdout-seasons` and `--clusters`.

- `data/backtest_cluster_summary.csv` shows each profile's training center, holdout and ranked sample sizes, average ADP, hit rate, and FPPG margin alongside the overall holdout results.
- `data/backtest_cluster_results.csv` has the held-out player records, profile assignments, and FPPG finishes.
- `data/backtest_cluster_pca.csv` shows which features contribute to the first two components and how much input variation each component explains. The held-out hit rates measure associations with beating ADP.

## Predicting who beats ADP

`backtest_models.py` compares an ADP-only logistic model, logistic regression using all six pre-draft features, and shallow gradient-boosted trees. It uses expanding draft-season folds. Each fold tunes on earlier drafts by average precision, then tests on the next draft. The latest two drafts are held out from this process for final evaluation by default.

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\python.exe backtest_models.py
```

**macOS and Linux**

```sh
.venv/bin/python backtest_models.py
```

Each model selects its top 25% of available picks in each evaluation draft. Use `--top-pct` to choose how selective the rule is, `--holdout-seasons` to set the final evaluation window, and `--cv-folds` to set the maximum number of chronological folds for both evaluation and tuning (default five). Players with zero appearances can still be selected and appear in the unranked count. Hit rates and average FPPG margins use ranked players.

Logistic regression tests stronger regularization and quadratic terms with pairwise feature interactions. Boosted trees test depth, minimum leaf size, and L2 regularization. Compare whether the models improve on ADP-only across individual folds and in the final holdout.

- `data/backtest_model_summary.csv` compares selected picks with the overall holdout rate and the equally selective ADP-only rule. It also has FPPG margins, whole-holdout average precision and ROC AUC, and the selected validation parameters.
- `data/backtest_model_results.csv` lists each holdout pick's model scores, selection flags, and FPPG outcome.
- `data/backtest_model_cv.csv` shows out-of-sample results and chosen parameters for each earlier evaluation draft, including the dates used for its fit and tuning.
- `data/backtest_model_parameter_search.csv` shows the mean and standard deviation of average precision for every candidate parameter set considered before the final holdout.

## Draft value beyond hit rate

The backtests report hit rate alongside two measures of how large the gains and losses were.

- **Average ADP slots gained** is ADP minus finish rank, averaged over players with a rank. Positive means a better finish rank than ADP. The reports show the ranked sample size.
- **Average FPTS above ADP** is actual regular-season fantasy points minus the points scored at the player's ADP slot on that season's league-wide leaderboard. Fractional ADPs interpolate between adjacent slots. The benchmark has a zero floor, and a player who missed the season scores zero actual points in this calculation.

The surge and decline backtests use season totals and show `adp_slot_fpts` and `fpts_above_adp`. The rookie and cluster backtests use the same comparison on the per-game leaderboard, with `next_fppg`, `adp_slot_fppg`, and `fppg_above_adp` in the player-level files. Summary CSVs and console reports use the units of their respective leaderboard. Compare each group's average with the corresponding overall or comparison group value in the same run.

### Low-game and outlier sensitivity

The median shows the typical pick. The trimmed mean removes 10% of scored outcomes from each end, rounded down to a whole player count shown in `trimmed_players_per_tail`. The full mean includes every scored pick in the group.

The reports show the contribution of low-game players to the full mean. That contribution is their sum of margins divided by the number of scored picks. The season-total backtests count 0–10 games and the per-game backtests count 1–10. The `avg_fpts_above_adp_over_10_games` or `avg_fppg_above_adp_over_10_games` column shows the mean among the rest. These are retrospective sensitivity checks using actual games played.

The player-level files include `next_lineup_role` and `next_start_pct` for checking lineup changes. The rookie summary already reports value by second-year lineup role.

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

Join the ESPN files on `game_id`. ESPN identifies players by `athlete_id`. The backtests match FantasyPros players to unique ESPN IDs using normalized player names. The CSV reader strips FantasyPros team and position tags and trailing statuses from older downloads. The clustering and model backtests require a match to a player who appeared the prior regular season.

The downloader prints how many completed games each CSV covers. If a requested feed is absent from both the CSV and the original game record, it reports the gap and exits with a nonzero status. Regular season, postseason, and any other game types present in the source stay in the files. Filter ESPN `season_type` to `2` for regular-season scoring.

Sources are [FantasyPros NBA ADP](https://www.fantasypros.com/nba/adp/overall.php), the [SportsDataverse ESPN archive](https://github.com/sportsdataverse/sportsdataverse-data/releases), and the [original ESPN game records](https://github.com/sportsdataverse/hoopR-nba-raw).
