# Historical ADP: automatic imports and manual requirements

Run from the repository folder in **Windows PowerShell**:

```powershell
git pull --ff-only
.\.venv\Scripts\python.exe -m src.adp_archive
.\.venv\Scripts\python.exe -m src.normalize --source sportsdataverse --enrich-bios
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m experiments.template --season 2025
Get-Content .\data\raw\adp\coverage_report.json
```

Run each line separately; stop if it fails. There are no new packages. Keep your existing virtual environment and downloaded NBA data.

## What is fetched automatically

`src.adp_archive` now preserves the pinned 2024–25 CSV and checks all 20 target seasons at `https://www.fantasypros.com/nba/adp/overall.php?year=START_YEAR`. It uses the **ESPN price**, not the consensus rank or other providers. The `year` URL parameter is the starting year; the project's `season` column is the ending year.

Verified imports on the current archive:

| Season | `season` value | ESPN update date (UTC calendar day) | Resolved players | Usage |
|---|---:|---|---:|---|
| 2014–15 | 2015 | 2014-10-27 | 155 | Training |
| 2017–18 | 2018 | 2017-10-11 | 151 | Training |
| 2019–20 | 2020 | 2019-10-18 | 193 | Training |
| 2022–23 | 2023 | 2022-10-17 | 150 | Holdout |
| 2023–24 | 2024 | 2023-10-22 | 195 | Holdout |
| 2024–25 | 2025 | 2024-10-19 (pinned Git commit) | 165 | Holdout |
| 2025–26 | 2026 | 2025-10-14 | 195 | Holdout |

Total: **1,204** resolved ADP observations. Coverage is partial, not certified top-200 completeness. The archive does not specify the precise historical scoring/league format. ESPN market conventions can change over time; these prices are not guaranteed to match your custom scoring market. Source-reported update dates are retained as evidence; independent contemporaneous captures would provide stronger provenance.

Original HTML is cached unchanged with SHA256/provenance sidecars under `data/raw/archive_sources/`. Converted CSVs are clearly identified as adapter outputs under `data/raw/adp/`. Reruns verify caches and leave existing user inputs intact. A changed adapter output raises an error instead of overwriting it. A network denial is reported as `fetch_failed`, not mistaken for an absent historical archive. The command returns successfully with partial coverage; inspect its report. `--seasons 2015 2018 2020` restricts the ending-year targets; `--pinned-only` runs only the old single-snapshot importer.

## Exactly what to find manually

For each missing season, find an **ESPN NBA ADP table/export captured before the cutoff below**, ideally 200+ priced players, plus evidence of its date. Suitable evidence includes a dated original CSV/PDF, a contemporaneous repository commit, or a Wayback snapshot containing the actual table. An expert's projected ranking, season-end player ranking, today’s download time, or a season label alone does not establish preseason ADP.

| Season | `season` | Snapshot must be dated BEFORE | Why automatic import rejects the current page |
|---|---:|---|---|
| 2006–07 | 2007 | 2006-10-31 | Historical URL falls back to another season |
| 2007–08 | 2008 | 2007-10-30 | Historical URL falls back; game season also quarantined for grading |
| 2008–09 | 2009 | 2008-10-28 | Historical URL falls back |
| 2009–10 | 2010 | 2009-10-27 | Historical URL falls back |
| 2010–11 | 2011 | 2010-10-26 | Historical URL falls back |
| 2011–12 | 2012 | 2011-12-25 | Historical URL falls back |
| 2012–13 | 2013 | 2012-10-30 | Historical URL falls back |
| 2013–14 | 2014 | 2013-10-29 | No unique dated ESPN source entry |
| 2015–16 | 2016 | 2015-10-27 | Update dated opening day |
| 2016–17 | 2017 | 2016-10-25 | Update dated after opening day |
| 2018–19 | 2019 | 2018-10-16 | Update dated opening day |
| 2020–21 | 2021 | 2020-12-22 | Update dated opening day |
| 2021–22 | 2022 | 2021-10-19 | Update dated opening day |

Try the [Wayback Machine](https://web.archive.org/) for the ESPN NBA live-draft-results page or the relevant FantasyPros year-specific page. Wayback access was blocked in this cloud environment; it may work in your browser. There is no confirmed complete free 20-year archive. Older years may require a provider’s licensed export or your own archived research files. Do not pay for data without confirming the years, sport, provider, actual ADP (not expert rank), snapshot dates, and permitted export format. A single league's draft order is not market-wide ADP.

Opening-day prices remain rejected because our date-only pipeline cannot establish that the update preceded the first game. Do not backdate them or move the cutoff just to pass validation.

## Convert a sourced file

Save UTF-8 CSVs under `data/raw/adp/`, one coherent source/snapshot per season. The required header is:

```csv
season,canonical_player_id,adp,adp_rank,position_rank,position_rank_status,available_at,source
```

Fill rows using the **actual sourced values**:

- `season`: ending year, e.g. 2022 for 2021–22.
- `canonical_player_id`: `nba:<official NBA ID>` from `data/processed/players.parquet`. Alternatively supply `full_name` (and optional `birth_date`) instead of the ID; ambiguous names must be reviewed.
- `adp`: observed ESPN average draft pick. Preserve fractions.
- `adp_rank`: order by ESPN ADP, not FantasyPros consensus. If omitted for the entire input set it is derived; supplying it is safer when combining files containing explicit ranks. Preserve rank gaps when some players cannot be resolved.
- `position_rank`: historical preseason rank within the source's position group. Use `position_rank_status=provided` when known. If unavailable, leave `position_rank` blank and set `position_rank_status=unavailable_historical_positions`.
- `available_at`: evidenced preseason date, `YYYY-MM-DD`.
- `source`: source/capture URL or a meaningful archived-file reference. Retain the supporting original artifact separately under `data/raw/archive_sources/`.

Use one file per season to avoid mixing markets. To replace an automatically imported season with a better manual snapshot, first **move its generated CSV out of `data/raw/adp/` into a backup folder**, preserving its metadata. The importer skips seasons already covered by other CSV files; it does not overwrite or arbitrate user data. Normalization rejects mixed sources in a season.

Then rerun `src.adp_archive` to refresh its coverage report, `src.normalize`, tests, and the real-data example. No NBA downloads are needed. New manual data does not change the frozen holdout rules: only complete seasons earlier than 2023 can train the holdout baseline.

## Remaining player and position gaps

The report lists unresolvable names individually. Current imports omit Wang Zhelin, Rade Zagorac, Patric Young, and/or Marcus Zegarowski where they appear in the source but cannot be linked unambiguously to this NBA master. Omitted prices are reported and everyone else's original draft rank is retained. Do not invent NBA IDs or silently compact the remaining ranks. Supplying verified identity records would be a separate master-data enrichment.

Current position/team/injury labels on historical FantasyPros pages are not historical snapshots. Therefore all six automatic archive imports have nullable `position_rank`, explicitly marked `unavailable_historical_positions` (1,039 rows). Total/PPG grading does not require position rank. To test position-based strategies, obtain dated preseason eligibility/position groups and then derive ranks within those groups. The pinned 2024–25 CSV already provides its original preseason position groups.
