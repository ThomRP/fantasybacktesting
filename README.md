# The Draft Model (`fantasybacktesting`)

A walk-forward, zero-leakage pre-draft NBA fantasy projection engine and interactive dark-mode replay dashboard for **Points** (ESPN / Yahoo / Sleeper / custom scoring) and **Unified 9-Cat** (Roto + H2H + interactive Punting).

---

## Core Design Philosophy

1. **Pre-Draft Projection Edge Only:** Built strictly to help you leave draft day with a stronger roster — zero mid-season waivers, trades, streaming, or post-Game-1 data.
2. **Honest Evaluation (Zero Survivorship Bias):**
   * **Points (`AdjFPPG`):** Graded on a 65-game reliability scale filled with replacement waiver production ($\text{waiver} = 0.85 \times \text{FPPG}_{150}$), keeping 0-game injury busts at waiver value instead of dropping them.
   * **9-Cat (`z_tot`):** Season-total Z-scores across all 9 categories standardized over the top-180 qualifiers, using volume-weighted FG%/FT% impact and $0.75\times$ inverted Turnovers.
3. **League-Size-Independent Validation:** Primary dashboards grade every pre-draft disagreement against consensus ADP across **five universal pick bands** (`Picks 1–25`, `26–60`, `61–100`, `101–150`, `151–225`) — working identically for 8-team, 10-team, 12-team, 14-team, or 16-team leagues — using **Calls Right %** (`50%` = coin flip) and **Targets vs. Fades** edge (`% FPTS` / `Z`).
4. **Two-Stage Workflow:**
   * **Stage 1 (`--mode pure_stats`, June–August, 0% ADP):** Full-NBA projection using Year 0 / Year −1 box scores, age/usage curves, rookie draft capital + team context, and optional live ESPN injury return-date caps (`--use-live-injuries`).
   * **Stage 2 (`--mode residual`, September–October, Draft Season):** ADP-anchored residual ensemble (`37.5% Huber + 37.5% Structural Decomposition + 25% shallow L1 GBDT` for veterans + 11-feature Rookie Hype-Tax Huber + Empirical-Bayes returning-absence drag) with a scale-invariant market-collapse injury hinge, late-round ramp ($\lambda = 0.50 \to 0.85$ over ADP `100–150`), **strictly pre-Opening-Night offseason roster-turnover features** (Points), and **9 regularized `RidgeCV` per-category Punt heads** (9-Cat).

---

## Target Metric (`AdjFPPG` for Points)

For a regular season under the active scoring system, let $\text{FPPG}_{150}$ be the 150th-ranked player's raw FPPG ($\ge 20$ games played), and define replacement waiver value as $\text{waiver} = 0.85 \cdot \text{FPPG}_{150}$. For any player with $g$ regular-season games played and average $\text{raw\_FPPG}$:

$$\text{AdjFPPG} = \frac{\min(g, 65) \cdot \text{raw\_FPPG} + (65 - \min(g, 65)) \cdot \text{waiver}}{65}$$

Players who miss the entire season ($g = 0$) score $\text{waiver}$ rather than being excluded.

---

## Repository Structure

### Projection Engines & Dashboards
* **`draft_engine.py`** — Points Stage-1 (`pure_stats`) and Stage-2 (`residual` + `residual_wait`) engine with pre-Opening-Night roster-turnover features (`changed_team`, `net_vacated_pos_mpg`, `net_vacated_val`), preseason Vegas win totals for rookies, opt-in `--advanced-stats`, custom scoring presets (`default`, `espn`, `yahoo`, `sleeper`, JSON), live ESPN injury capping, and optional draft simulator (`--backtest`).
* **`cat9_engine.py`** — Unified 9-Cat residual engine (`0.75× TO`) powering Roto, H2H Each-Cat, and H2H Most-Cat, upgraded with **9 regularized `RidgeCV` per-category heads** for sharper Punt re-rankings while preserving exact balanced 9-Cat totals.
* **`board_templates.py`** — Shared dark-mode HTML/CSS/JS templates for **"The Draft Model"** featuring the 2-column **Edge by Draft Range** hero (`Multi-Year Track Record` vs. live-interactive `Selected Season`), game-by-game replay slider & trajectory chart (`15 Liked vs. Faded (vs ADP)`, `Top 150 Swaps`, `% Calls Right`), and **click-to-inspect side-by-side comparison drawers** with season `Finish` ranks.
* **`build_board_page.py`** — Generates **`board_2026_27.html`** (9-Yr Points: `2018–19`–`2026–27`) and **`board_13yr_2014_2027.html`** (13-Yr Points: `2014–15`–`2026–27`).
* **`build_cat9_board_page.py`** — Generates **`board_cat9_2026_27.html`** (9-Cat Unified with interactive client-side Punt chips).

### Data Ingestion, Utilities & Tests
* **`download.py`** — Downloads FantasyPros historical ADP (`adp.csv`) and ESPN/SportsDataverse schedules and box scores (`games.csv`, `player_box_scores.csv`, `team_box_scores.csv`).
* **`download_vegas_odds.py`** — Downloads verified Basketball-Reference preseason NBA Vegas Win Totals into `data/vegas_preseason_odds.csv`.
* **`download_preseason_rosters.py`** — Downloads dated Basketball-Reference NBA transaction logs (`2008–09`–`2026–27`) and compiles `data/preseason_rosters.csv` strictly cut off at `tx_date < opening_night` (`99.35%` validated accuracy vs. Week-1 jerseys with zero post-Game-1 data).
* **`backtest_common.py`**, **`adp_names.py`**, **`backtest_rookies.py`**, **`backtest_rookie_plots.py`**, **`backtest_full_pool.py`**, **`compare_early_season.py`** — Shared data loaders, name normalization, and evaluation helpers. *(Note: `experience_years` in `player_core` reflects current career totals and is strictly excluded.)*
* **`test_cat9_engine.py`**, **`test_injury_cap.py`**, **`test_reach_or_wait.py`** — Unit test suite (`15/15` passing).

---

## Setup & Data Download

Use Python 3.10+ from the repository root (`fantasybacktesting/`).

### Windows (PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Historical + live seasons
.\.venv\Scripts\python.exe download.py 2015 2025 --adp-year-is start
.\.venv\Scripts\python.exe download.py 2026 --adp-year-is start

# Preseason Vegas win totals (rookies) & pre-Opening-Night rosters (tx_date < Game 1)
.\.venv\Scripts\python.exe download_vegas_odds.py
.\.venv\Scripts\python.exe download_preseason_rosters.py
```

---

## Usage

### 1. Build Interactive Dashboards ("The Draft Model")
```powershell
# 9-Yr Points (2018-19 .. 2026-27 Live)
.\.venv\Scripts\python.exe build_board_page.py

# 13-Yr Points (2014-15 .. 2026-27 Live)
.\.venv\Scripts\python.exe build_board_page.py --min-train-season 2008-09 --cv-folds 10 --out board_13yr_2014_2027.html

# 9-Cat Unified (Roto + H2H + interactive Punt chips)
.\.venv\Scripts\python.exe build_cat9_board_page.py
```

### 2. Generate CLI Pre-Draft Boards
```powershell
# Stage 1 (June–August, 0% ADP + live ESPN injury caps)
.\.venv\Scripts\python.exe draft_engine.py --generate-board 2026-27 --mode pure_stats --use-live-injuries --scoring default

# Stage 2 (September–October, ADP-anchored champion + live ESPN injury caps)
.\.venv\Scripts\python.exe draft_engine.py --generate-board 2026-27 --mode residual --lam 0.50 --use-live-injuries --scoring default
```

### 3. Run Unit Tests
```powershell
.\.venv\Scripts\python.exe -m unittest discover -p "test_*.py" -v
```

---

## Verified Strict Out-of-Sample Pick-Band Performance

All historical seasons are evaluated strictly out-of-sample (chronological expanding-window training, zero future leakage; `2024–25` & `2025–26` held out from development; live `2026–27` dynamically equalized to `min_team_games = 16` across all 30 NBA teams through `2026-11-24`):

### Pick-Band Edge (All Completed Past Seasons)

| ADP Pick Band | Points 9-Yr (`8 Yrs: 2018–26`) | Points 13-Yr (`12 Yrs: 2014–26`) | 9-Cat Unified (`8 Yrs: 2018–26`) |
| :--- | :---: | :---: | :---: |
| **Picks 1–25** | `59.2%` right · `+3.6% FPTS` | `55.2%` right · `+3.0% FPTS` | `47.9%` right · `+1.91 Z` |
| **Picks 26–60** | `55.5%` right · `+12.7% FPTS` | `58.6%` right · `+9.3% FPTS` | `55.4%` right · `+1.36 Z` |
| **Picks 61–100** | `52.5%` right · `+8.0% FPTS` | `55.2%` right · `+7.3% FPTS` | `60.3%` right · `+1.54 Z` |
| **Picks 101–150** | **`61.0%` right · `+28.2% FPTS`** | **`61.7%` right · `+25.9% FPTS`** | **`58.6%` right · `+1.70 Z`** |
| **Picks 151–225** | **`64.8%` right · `+35.2% FPTS`** | **`61.0%` right · `+27.1% FPTS`** | **`64.7%` right · `+2.31 Z`** |
| **Overall (1–150)** | **`58.9%` (`8/8` Beat ADP Yrs)** | **`60.0%` (`12/12` Beat ADP Yrs)** | **`58.6%` (`8/8` Beat ADP Yrs)** |
| **Live `2026–27` (`16 GP`)** | **`59.2%` Calls Right** | **`58.2%` Calls Right** | `48.6%` Calls Right (`+42.3 Z` Top-15 gap) |

### 9-Cat Per-Category `RidgeCV` Punt Accuracy (Holdout `2024–26` Calls Right)
* **`Punt TO`:** `47.5%` → **`52.2%`** (`+4.7%` out-of-sample lift)
* **`Punt FT% + TO`:** `50.5%` → **`54.5%`** (`+4.0%` out-of-sample lift)
* **`Punt FT%`:** `57.4%` → **`58.0%`** | **`Punt FG%`:** `59.8%` → **`60.0%`** | **`Punt AST`:** `62.3%` → **`62.6%`**
