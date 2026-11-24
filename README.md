# Fantasy Basketball Draft Engine (`fantasybacktesting`)

A walk-forward, leak-free fantasy basketball draft projection engine, 12-team ESPN points-league draft simulator, and interactive multi-season replay dashboard. Built to harvest repeatable draft-day edges over consensus ADP (and generate standalone pre-ADP boards before fantasy drafts open) across any custom points-league scoring system.

---

## Core Design Philosophy

1. **Pre-Draft & Draft-Day Edge Only:** Focuses exclusively on walking out of the draft room with systematically underpriced, durable production rather than chasing noisy mid-season waiver/streaming variance.
2. **Honest Evaluation (Zero Survivorship Bias):** Every drafted player is graded on **Availability-Adjusted FPPG (`AdjFPPG`)** on a 65-game scale with 0-game injury busts included at replacement waiver value rather than dropped.
3. **Objective Roster Guardrails Only:** Enforces ESPN's **4-Center cap** (primary-listed `C`) and legal 10-starter (`PG, SG, SF, PF, C, G, F, 3×UTIL`) + 3-bench rosters. Intentionally omits subjective positional VORP.
4. **Two-Stage Workflow:**
   * **Stage 1 (`--mode pure_stats`, June–August, 0% ADP):** Projects ~350–400 NBA players completely independent of ADP using Year 0 / Year −1 box scores, shrunk 3P%, age/usage interactions, NBA draft pick, and team context (`37.5% Huber + 37.5% Structural MPG×FPPM×Avail + 25% shallow L1 GBDT`), paired with live ESPN injury return-date caps (`--use-live-injuries`).
   * **Stage 2 (`--mode residual`, September–October, Draft Season):** Switches to the ADP-anchored residual ensemble (`37.5% Huber + 37.5% Structural Decomposition + 25% shallow L1 GBDT` for veterans + 11-feature Rookie Hype-Tax Huber + Empirical-Bayes returning-absence drag), with a scale-invariant market-collapse injury hinge, a late-round sleeper ramp ($\lambda = 0.50 \to 0.85$ over ADP `100–150`), and an ADP-aware 2-pick **Reach-vs.-Wait** optimizer (`residual_wait`, $\lambda = 0.85$).

---

## Target Metric (`AdjFPPG`)

For a regular season under the active scoring system, let $\text{FPPG}_{150}$ be the 150th-ranked player's raw FPPG ($\ge 20$ games played), and define replacement waiver value as $\text{waiver} = 0.85 \cdot \text{FPPG}_{150}$. For any player with $g$ regular-season games played and average $\text{raw\_FPPG}$:

$$\text{AdjFPPG} = \frac{\min(g, 65) \cdot \text{raw\_FPPG} + (65 - \min(g, 65)) \cdot \text{waiver}}{65}$$

Players who miss the entire season ($g = 0$) score $\text{waiver}$ rather than being excluded.

---

## Repository Structure

### Production Engine & Interactive Dashboard
* **`draft_engine.py`** — Unified Stage-1 (`pure_stats`) and Stage-2 (`residual` + `residual_wait`) engine, verified preseason Vegas win totals for incoming rookies (`team_vegas_win_pct`, automatic fallback to prior win%), opt-in local advanced box metrics (`--advanced-stats`), custom scoring presets (`default`, `espn`, `yahoo`, `sleeper`, JSON file/dict), live ESPN injury API integration (`--use-live-injuries`, `--as-of-date`), and 12-team ESPN daily active-lineup backtest simulator (`--backtest`).
* **`build_board_page.py`** — Generates **`board_2026_27.html`**, a bright, streamlined offline interactive dashboard covering all **9 out-of-sample seasons** (6 CV folds `2018–19`–`2023–24`, 2 untouched Holdouts `2024–25`–`2025–26`, and Live `2026–27`) with a **4 games/sec real-time game-by-game replay slider** and interactive trajectory chart.
* **`board_2026_27.html`** — Ready-to-open offline HTML dashboard (double-click to view in any browser).

### Data, Utilities & Tests
* **`download.py`** — Downloads FantasyPros historical ADP (`adp.csv`) and ESPN/SportsDataverse schedules and box scores (`games.csv`, `player_box_scores.csv`, `team_box_scores.csv`).
* **`download_vegas_odds.py`** — Downloads verified Basketball-Reference preseason NBA Vegas Win Totals (`2016–17`–`2026–27`) into `data/vegas_preseason_odds.csv` for rookie team-competitiveness context.
* **`backtest_common.py`** — Shared box-score parser, default scoring constants, ESPN `player_core` DOB/profile loader, and CSV helpers. *(Note: `experience_years` in `player_core` reflects current career totals and is strictly excluded to prevent survivorship leakage.)*
* **`adp_names.py`** — FantasyPros player name normalization.
* **`backtest_rookies.py`** & **`backtest_rookie_plots.py`** — Career appearance archive loader (back to 1993–94) for leak-free rookie identification.
* **`backtest_full_pool.py`** & **`compare_early_season.py`** — Full-pool helpers and proportional in-progress season leaderboard evaluator.
* **`test_injury_cap.py`** & **`test_reach_or_wait.py`** — Unit test suites for pre-season/mid-season injury availability capping and 2-pick reach-vs.-wait survival EV.

---

## Setup & Data Download

Use Python 3.10+ from the repository root.

### Windows (PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Download completed historical seasons (2015-16 through 2025-26)
.\.venv\Scripts\python.exe download.py 2015 2025 --adp-year-is start

# Download current/upcoming season (2026-27)
.\.venv\Scripts\python.exe download.py 2026 --adp-year-is start
```

### macOS / Linux
```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download.py 2015 2025 --adp-year-is start
.venv/bin/python download.py 2026 --adp-year-is start
```

---

## Usage

### 1. Run Walk-Forward Backtest & 12-Team ESPN Draft Simulation
```powershell
# Default / ESPN scoring (1,152 simulated 12-team snake drafts per strategy)
.\.venv\Scripts\python.exe draft_engine.py --backtest --scoring default --seed 42

# Yahoo or Sleeper scoring (or custom JSON file / inline dict)
.\.venv\Scripts\python.exe draft_engine.py --backtest --scoring yahoo --seed 42
.\.venv\Scripts\python.exe draft_engine.py --backtest --scoring sleeper --seed 42
```

### 2. Generate Pre-Draft & Draft-Day Boards
```powershell
# Stage 1 (June–August, 0% ADP + live ESPN injury caps)
.\.venv\Scripts\python.exe draft_engine.py --generate-board 2026-27 --mode pure_stats --use-live-injuries --scoring default

# Stage 2 (September–October, ADP-anchored champion + live ESPN injury caps)
.\.venv\Scripts\python.exe draft_engine.py --generate-board 2026-27 --mode residual --lam 0.50 --use-live-injuries --scoring default
```
* Supports `--rookies-csv`, `--injuries-csv`, `--exclude-players`, `--opening-night YYYY-MM-DD`, and `--as-of-date YYYY-MM-DD`.

### 3. Build the Interactive Multi-Season Replay Webpage
```powershell
.\.venv\Scripts\python.exe build_board_page.py
start board_2026_27.html
```

### 4. Run Unit Tests
```powershell
.\.venv\Scripts\python.exe -m unittest -v test_injury_cap.py test_reach_or_wait.py
```

---

## Verified Walk-Forward Backtest Results

Same model architecture across all seasons, refit chronologically using only prior seasons (`2018–19` trained strictly on `2016–17`–`2017–18`; `2024–25` & `2025–26` held out completely untouched from development; `2026–27` evaluated zero-leakage). Evaluated via **1,152 simulated 12-team × 13-round ESPN snake drafts/strategy** (`draft_engine.py --backtest --seed 42`, R1 straight ADP, R2–13 $\text{Uniform}[-2, +2]$, exact daily 10-starter + 3-bench scoring, 4-C primary cap):

### Default / ESPN Points Scoring

| Strategy | All 8 Yrs Diff | Lift % | Untouched Holdout Diff | Avg Rank (All / HO) | Win % (All / HO) | Top-4 % (All / HO) | 0g Busts (H / F) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `market` (strict ADP) | +77.2 | +0.31% | -198.9 | 6.42 / 6.73 | 8.9% / 6.6% | 34.3% / 29.5% | 0.19 / 0.20 |
| `pure_stats` (0% ADP, blind) | -4,410.3 | -17.25% | -6,782.8 | 11.21 / 11.67 | 0.0% / 0.0% | 0.0% / 0.0% | 1.39 / 0.09 |
| `pure_stats_news_cut` (Stage 1 + collapse cut) | +758.5 | +3.01% | -1,647.0 *(+1,560 CV)* | 5.40 / 8.23 | 22.8% / 0.7% | 44.4% / 17.7% | 0.16 / 0.20 |
| `blend_adp_50` (50/50 Stage 1 + ADP) | +1,236.0 | +4.92% | +135.5 *(+1,603 CV)* | 4.72 / 6.34 | 19.8% / 3.1% | 53.3% / 33.7% | 0.18 / 0.20 |
| **`residual_unified`** (greedy, $\lambda = 0.50 \to 0.85$ ramp) | **+1,693.0** | **+6.75%** | **+1,801.2 (+7.14%)** | **4.32 / 4.36** | **28.0% / 29.2%** | **59.6% / 58.0%** | **0.03 / 0.21** *(0.00 HO)* |
| **`residual_wait`** (reach-vs.-wait, $\lambda = 0.85$) | **+1,966.5** | **+7.85%** | **+2,431.0 (+9.66%)** | **4.04 / 3.82** | **29.7% / 36.8%** | **62.9% / 65.6%** | **0.02 / 0.21** *(0.00 HO)* |

* **8-for-8 winning seasons** for `residual_wait` under Default scoring, with Holdout Top-150 capture **`241/300`** and Rank MAE **`43.93`** (vs. Market `237/300` and `50.54`).

### Yahoo Scoring Cross-Check (`--scoring yahoo --seed 42`)

| Strategy | All 8 Yrs Diff | Lift % | Untouched Holdout Diff | Avg Rank (All / HO) | Win % (All / HO) | Top-4 % (All / HO) | Holdout Top-150 (Rank MAE) |
|---|---:|---:|---:|---:|---:|---:|---|
| `residual_unified` | **+2,134.0** | **+8.70%** | **+2,211.5 (+9.04%)** | 3.69 / 3.74 | 31.2% / 23.6% | 68.8% / 68.4% | **250/300** (39.84 vs. 49.35 Mkt) |
| **`residual_wait`** | **+2,467.3** | **+10.07%** | **+2,451.1 (+10.03%)** | **3.39 / 3.48** | **36.6% / 29.5%** | **72.3% / 70.1%** | **250/300** (**38.40** vs. 49.35 Mkt) |

### Out-of-Sample `2026–27` Audit (241 Games Through `2026-11-21`, ~19.6% of Season)
* **Top-150 edge correlation `r(pred_edge, actual_vs_ADP)`:** **`+0.145`**, positive across all 4 round bands (`9/9` consecutive positive seasons `2018–19` through `2026–27`).
* **`ADP 151–225` sleeper discrimination:** Positive-edge sleepers hit **`10/14 (71.4%)`** actual Top-150 vs. **`14/63 (22.2%)`** for negative-edge players.
* **Stage 1 full-NBA Top-150 (0% ADP + injury guard):** Captured **`110/150`** actual Top-150 players (vs. Market `108/150`), hitting **`16/32 (50.0%)`** on `ADP > 150` / unranked breakouts.
