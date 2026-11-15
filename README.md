# NBA fantasy backtesting foundation

Python ingestion, point-in-time queries, and standardized cohort grading. No draft simulator or UI.

The research window is **2006–07 through 2025–26**: `season` always means the **ending year**, 2007–2026. The fixed, four-season holdout is **2023–2026**. Updating the research window is an explicit configuration change, not something a later clock silently changes.

## Run on Windows PowerShell

Python 3.12 is tested. From the repository folder, run each line separately and stop if a command fails. Do not copy shell prompts or add trailing backslashes. Virtual-environment activation is unnecessary.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m src.ingest --source sportsdataverse
.\.venv\Scripts\python.exe -m src.adp_archive
.\.venv\Scripts\python.exe -m src.player_metadata
.\.venv\Scripts\python.exe -m src.normalize --source sportsdataverse --enrich-bios
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m experiments.template --demo
.\.venv\Scripts\python.exe -m experiments.template --season 2025
```

Already completed the original setup? Update the code and fetch the new historical ADP without downloading NBA box scores again:

```powershell
git pull --ff-only
.\.venv\Scripts\python.exe -m src.adp_archive
.\.venv\Scripts\python.exe -m src.normalize --source sportsdataverse --enrich-bios
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m experiments.template --season 2025
Get-Content .\data\raw\adp\coverage_report.json
```

The dependency list is unchanged. On Linux/macOS, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`; create the environment with `python3 -m venv .venv`.

`--demo` uses clearly labeled synthetic observations to exercise a trained grader. The default example uses the actual Parquet data, selects a cohort from **prior-season PPG available before the target season**, and reports missing coverage instead of inventing performance. A supplied strategy cohort can be evaluated with `--cohort .\my_cohort.csv`, containing `season,canonical_player_id`.

The pipeline has no database or running service. Later tasks use this existing isolated checkout; no extra worktree is needed. Downloads are cached and resumable. Never delete the raw cache simply to rerun normalization.

## Data sources and current coverage

* **Official NBA:** `nba_api` endpoint definitions for `LeagueGameLog` (player rows, regular season), `CommonAllPlayers`, and `CommonPlayerInfo`. Run `.\.venv\Scripts\python.exe -m src.ingest --source nba` followed by `.\.venv\Scripts\python.exe -m src.normalize --source nba` when `stats.nba.com` is reachable. Official transport currently receives a network-policy CONNECT 403 in this environment. It is implemented and mocked in tests; a live official download has not been verified.
* **Working explicit fallback:** [SportsDataverse NBA Stats archive](https://github.com/sportsdataverse/hoopR-nba-stats-data), sourced from NBA Stats. Exact game-log URL: `https://github.com/sportsdataverse/sportsdataverse-data/releases/download/nba_stats_player_game_logs/player_game_logs_{season}.parquet`; roster URL: the same prefix with `nba_stats_rosters/rosters_{season}.parquet`. See the archive's [player-log schema](https://github.com/sportsdataverse/hoopR-nba-stats-data/blob/main/docs/datasets/player_game_logs.md). Optional metadata sources use the same release prefix with `nba_crosswalk/nba_player_crosswalk_{season}.parquet` and `espn_nba_player_core/player_core_{season}.parquet`; [crosswalk documentation](https://github.com/sportsdataverse/hoopR-nba-data/blob/main/docs/datasets/player_crosswalk.md). The adapter checks for actual player columns because older versions of this archive tag mistakenly contained team rows. Raw assets may include playoffs; normalization explicitly selects regular-season rows and recalculates fantasy scoring.
* **Historical ADP:** `src.adp_archive` now automatically checks year-specific [FantasyPros NBA ADP archives](https://www.fantasypros.com/nba/adp/overall.php?year=2017) and imports the **ESPN** column only. It verifies the requested season, ESPN's source-reported update date, and strictly preseason availability. It never substitutes consensus order, expert rankings, Yahoo prices, or current-year fallback pages. Raw HTML and checksums are retained. These dates are claims made by the source, not independently verified Wayback capture dates. The archive does not identify the precise scoring/league format, so these are historical ESPN market prices, not certified custom-league points ADP. Historical page teams/positions/injuries are current and are ignored.
* **Preserved 2024–25 snapshot:** the original [immutable ESPN CSV](https://raw.githubusercontent.com/toddehalexander/NBA_ADP_Player_Selector/8e501a78fa9989edaf532e24580f0a7365c077e1/nba-adp-All.csv) remains supported and is preserved; [2024-10-19 commit evidence](https://github.com/toddehalexander/NBA_ADP_Player_Selector/commit/8e501a78fa9989edaf532e24580f0a7365c077e1). Its 165 priced players include genuine preseason position groups. Use `src.adp_archive --pinned-only` for just that original importer. Existing manual season CSVs are preserved and prevent automatic imports of a second market for that season.

All 20 source seasons downloaded successfully, with expected unique-game counts (1230 except 2012:990, 2013:1229, 2020:1059, 2021:1080). Per-game validation requires two teams and at least five player appearances per team. Repeating ingestion verified all caches without further downloads. Source checks cannot independently certify every upstream statistic.

**Verified ADP coverage:** automatic imports plus the preserved snapshot yield **1,204 resolved players across seven seasons**: ending years **2015, 2018, 2020, 2023, 2024, 2025, 2026**. The first three train the chronological baseline; all four holdout seasons remain excluded from training. The real 2025 example now grades 133 of 143 signals and explicitly excludes 10 missing ADP matches. This demonstrates an operational grader, not proof of strategy performance or complete market coverage. All snapshots are partial, and missing observations may bias results.

**Manual gaps:** thirteen other seasons still need earlier or otherwise verifiable ESPN ADP snapshots. `data/raw/adp/coverage_report.json` lists each requested season, accepted row counts, source/date failures, and unresolved player names. See **[exact manual requirements and cutoff dates](docs/HISTORICAL_ADP.md)**. Missing dates, opening-day updates, and current-year fallback pages are rejected. Position ranks are unknown for the six FantasyPros imports: the original historical positions cannot be recovered from the current page labels. Total/PPG grading works without them; position strategies need separate dated position evidence.

**Replay quarantine:** NBA game `0020700367`, recorded as 2007-12-19, combines a suspended MIA–ATL game with its March replay, including Shawn Marion after his later trade. Those **23 raw rows** cannot meet exact event-date discipline and are retained separately in `data/processed/quarantine.parquet`. They are excluded from `game_logs.parquet`; all of season 2008 is marked incomplete for outcome grading so missing points cannot masquerade as end-of-season totals. Unexpected duplicate player/date keys cause normalization to fail rather than silently drop a row.

Current processed data has **503,822** accepted regular-season appearances, including **2,229** rounded zero-minute appearances. NBA player logs record actual appearances; zero-minute rows are not DNP placeholders. The master contains roster-only players as well as log players. Missing master biographies remain explicit null birth dates and `UNKNOWN` positions rather than invented values; `coverage.json` reports exact coverage. Optional `src.player_metadata` enriches 184 otherwise missing biographies through explicit, bijective NBA–ESPN IDs, exact normalized names, and conflict-free dates of birth. Eight players still have null birth dates and `UNKNOWN` positions. Master position/name/bio fields can reflect later roster information and are **identity/display metadata, not historical strategy features**. Use preseason source position for historical position signals.

## Layout and schemas

```text
data/raw/                  # Original cached payloads and checksum/provenance sidecars
  adp/                     # Manual CSV inputs and clearly labeled adapter staging CSVs
  archive_sources/         # Unmodified pinned ADP download
  sportsdataverse/         # Original Parquet logs/rosters
  nba/                     # Official JSON responses, when reachable
data/processed/
  players.parquet
  game_logs.parquet
  adp.parquet
  quarantine.parquet
  coverage.json            # Completion, gaps, scoring and processed-file checksums
src/config.py              # Universe, season dates and configurable scoring
src/ingest.py               # Verified caches, throttling, retries and fetch adapters
src/normalize.py            # Entity resolution, validation, dates, scoring, Parquet
src/adp_archive.py          # Historical + pinned preseason ADP import CLI
src/fantasypros_adp.py       # Dated ESPN archive parser, ID resolution, gap report
src/player_metadata.py      # Optional strict cross-source biography enrichment
src/benchmark.py            # Outcomes and monotone rank-to-value curves
src/grader.py               # Chronological cohort evaluation
experiments/template.py
tests/test_pipeline.py
```

`players.parquet`: `canonical_player_id` (`nba:<official ID>`), `full_name`, `normalized_name`, `primary_position`, nullable `birth_date` (Arrow date32). IDs are team independent. Unicode accents, punctuation, and suffixes are normalized for matching, but ambiguous names raise; explicit IDs or source aliases resolve collisions. Conflicting known birth dates for an ID also raise.

`game_logs.parquet`: `season`, exact `game_date` (date32), `canonical_player_id`, `team`, `opponent`, `minutes_or_snaps`, `fgm,fga,ftm,fta,fg3m,reb,ast,stl,blk,tov,pts`, `fantasy_points`. Dates are the source's NBA local game dates, not UTC-converted timestamps. Historical publication times and stat-correction vintages are not available; these are day-granularity game observations with an exclusive date cutoff, not intraday archival snapshots. `(season,game_date,canonical_player_id)` is unique. Scoring is the uploaded image, exactly:

```python
SCORING = {
    "fgm": 2, "fga": -1, "ftm": 1, "fta": -1, "fg3m": 1,
    "reb": 1, "ast": 2, "stl": 4, "blk": 4, "tov": -2, "pts": 1,
}
```

`adp.parquet`: `season`, `canonical_player_id`, `adp`, `adp_rank`, `position_rank`, plus required `available_at` (date32) and `source` provenance. Fractional ADP prices are preserved; overall ranks are positive integers. `position_rank` is a nullable Arrow integer: a missing historical position must have `position_rank_status=unavailable_historical_positions`; supplied ranks use `position_rank_status=provided`. Current positions are never substituted. All foreign keys must resolve. One source/market per season is enforced; no playerwise merging of providers or snapshot dates. When multiple snapshots exist, the latest **entire** preseason snapshot is selected.

## Add historical ADP

Drop CSV files into `data/raw/adp/`, then rerun normalization. Required columns:

```csv
season,canonical_player_id,adp,adp_rank,position_rank,available_at,source
2022,nba:203999,2.3,1,1,2021-10-15,https://example.org/archived-preseason-market.csv
```

This example shows the **format only**, not a sourced historical price. Instead of `canonical_player_id`, use `nba_player_id`, or `full_name` plus optional `birth_date` for unambiguous resolution. Programmatic `PlayerResolver`/`normalize_adp` also accept an explicit alias table (`source,source_player_id,canonical_player_id`). If omitted, `adp_rank` is derived by sorting ADP/name-independent canonical ID; supply the source's rank when available. `position_rank` must come from the historical market, never from the modern master position. If genuinely unavailable, leave it blank and add `position_rank_status=unavailable_historical_positions`; this permits total/PPG grading without inventing positional data. Keep documentary evidence for `available_at`: a filename, historical season label, or today's download date alone does not prove preseason availability. The normalizer enforces dates, but cannot authenticate unsupported manual assertions.

`available_at` must precede the configured season opener. Same-day snapshots are conservatively rejected because sources are modeled at day granularity. Missing prices remain missing, never estimated from actual performance. Coverage and missing-season warnings are emitted after every run. Do not mix different scoring formats or ADP providers in a research sample without an explicit research decision.

## Query and grade

```python
import pandas as pd
from src.normalize import filter_game_logs
from src.grader import evaluate_cohort

logs = pd.read_parquet("data/processed/game_logs.parquet")
# [start,end), also strictly earlier than as_of: no same-day/future games.
known = filter_game_logs(logs, start="2024-01-01", end="2024-01-15", as_of="2024-01-15")
cohort = pd.DataFrame({"season": [2025], "canonical_player_id": ["nba:203999"]})
report = evaluate_cohort(cohort)
print(report["summary"])
print(report["player_results"])
```

The standard entry point reads the coverage manifest and verifies completion, scoring, row counts, and saved file checksums. Incomplete seasons never train a baseline or produce a graded outcome. Direct DataFrame overrides are supported for tests/research, but callers must provide complete normalized regular seasons and validated preseason ADP, not rolling windows or DNP placeholders. Signal generation must use knowledge available before the draft; the grader cannot audit arbitrary strategy code.

`benchmark.fit_expected_curve(aggregate_outcomes(logs, adp, completed_seasons=coverage["completed_seasons"]))` fits an explicitly **descriptive** all-history curve, where `coverage` is loaded from `coverage.json`. Standalone `aggregate_outcomes` and `fit_chronological_curve` accept this explicit completion set; if omitted, callers assume responsibility for complete input seasons, just as with direct grader DataFrame overrides. The grader instead refits each development season using **strictly earlier completed seasons**. All four holdout seasons use the same frozen pre-2023 training pool; none of their outcomes enter training. Isotonic regression fits separate nonincreasing total-points and PPG expectations at ranks 1–200. Predictions outside observed training rank support are clipped to the nearest fitted endpoint; sparse coverage is not proof of reliable expectations at every slot. Players with no appearances contribute zero total and undefined PPG; a wholly missing/incomplete season is not converted into zeros.

`evaluate_cohort` returns `summary`, `era_breakdown`, `split_breakdown`, `player_results`, and `exclusions`. Metrics include mean/median total alpha, mean/median PPG alpha, and beat-ADP percent (`alpha_total > 0`, ties do not win). Total and PPG denominators are separate. Era buckets use ending years: 2006–2010, 2011–2015, 2016–2020, 2021–2025, 2026–2030. The last bucket contains only 2026 in this window. The holdout designation is independent of era grouping. Pre-register a strategy before inspecting holdout results; a split label cannot prevent human tuning.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Behavioral tests exercise exact scoring, immutable-cache/checksum/retry recovery, typed Parquet round trips, traded-player identity, ambiguous-name rejection, duplicate detection, exclusive knowledge cutoffs, valid ADP joins and provenance, monotonic curves, chronological/frozen training, absent-season handling, zero-game PPG, coverage-manifest integrity, and all reported metric denominators. Tests use deterministic fixtures and do not hit remote APIs. Actual artifact coverage and source rows are separately validated by the commands above.
