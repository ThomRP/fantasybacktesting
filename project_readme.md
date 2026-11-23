# Fantasy Basketball Draft Engine (`fantasybacktesting`)

A data-driven, leak-free fantasy basketball draft projection engine and 12-team ESPN points-league draft simulator. Built to harvest repeatable draft-day edges over consensus ADP (and generate standalone pre-ADP boards before fantasy drafts open) across any custom points-league scoring system.

---

## Core Philosophy

1. **Draft-Room Edge Over Mid-Season Noise:** Mid-season waivers, streaming, and freak injuries are high-variance and hard to repeat. This project focuses exclusively on walking out of the draft room with systematically underpriced, durable production.
2. **Honest Evaluation (Zero Survivorship Bias):** Every drafted player is graded on **Availability-Adjusted FPPG (`AdjFPPG`)** with 0-game busts included at waiver-wire value rather than dropped.
3. **Objective Roster Guardrails Only:** Enforces ESPN's **4-Center cap** (primary-listed `C`) and legal 10-starter (`PG, SG, SF, PF, C, G, F, 3xUTIL`) + 3-bench rosters. Intentionally omits positional VORP due to cross-platform eligibility subjectivity.
4. **Two-Stage Workflow:**
   * **Stage 1 (`--mode pure_stats`, June–August):** 100% ADP-free projections trained strictly on Year 0 / Year $-1$ box scores, age, NBA draft pick, and Year-0 team context, paired with live ESPN injury return-date availability caps.
   * **Stage 2 (`--mode residual`, September–October):** Once consensus ADP forms, switches to the ADP-anchored residual ensemble (`residual_unified`, $\lambda = 0.50$) so the market absorbs summer trades/depth-chart news while the model exploits age decay, `MPG`/`FPPM`/availability durability, offseason injury collapses, and the rookie hype tax.

---

## Target Metric (`AdjFPPG`)

For a regular season under the active scoring system, let $\text{FPPG}_{150}$ be the 150th-ranked player's raw FPPG (minimum 20 games played), and define replacement waiver value as $\text{waiver} = 0.85 \cdot \text{FPPG}_{150}$. For any player with $g$ regular-season games played and raw average $\text{raw\_FPPG}$:

$$\text{AdjFPPG} = \frac{\min(g, 65) \cdot \text{raw\_FPPG} + (65 - \min(g, 65)) \cdot \text{waiver}}{65}$$

A 0-game injured bust scores $\text{waiver}$ instead of being omitted as unranked.

---

## Repository Structure

### Production Entry Point
* **`draft_engine.py`** — Unified final engine supporting:
  * Custom points scoring (`default`, `espn`, `yahoo`, `sleeper`, JSON file, or inline JSON dict).
  * Stage-1 (`pure_stats`, 0% ADP) and Stage-2 (`residual`, $\lambda = 0.50$) board generation.
  * Live ESPN NBA injury API fetcher (`--use-live-injuries`) and offline CSV overrides (`--injuries-csv`, `--rookies-csv`).
  * Full chronological backtest + 12-team ESPN daily active-lineup draft simulator (`--backtest`).

### Data & Shared Utilities
* **`download.py`** — Downloads FantasyPros historical ADP (`adp.csv`) and ESPN/SportsDataverse schedules and box scores (`games.csv`, `player_box_scores.csv`, `team_box_scores.csv`).
* **`backtest_common.py`** — Shared box-score parser, default scoring constants, `player_core` profile/DOB loader, and CSV helpers.
* **`adp_names.py`** — FantasyPros player name normalization.
* **`backtest_rookies.py`** — Career NBA appearance history loader (back to 1993–94) used for leak-free rookie identification.

### Research & Validation Scripts
* **`backtest_final_board.py`** — Returning-veteran-only 2-model ensemble (`huber_clean + decomposed_clean`) CV/Holdout validator.
* **`backtest_rookie_debut.py`** — Standalone incoming-rookie study (draft pick, redshirt status, rookie age, established team top-8 rotation age, 30+ veteran minute share, prior win %).
* **`backtest_full_pool.py`** — Unified veteran + rookie board backtest and ESPN simulator that validated `unified_equal` ($\lambda = 0.50$).
* **`simulate_draft.py`** — Standalone 12-team ESPN snake draft simulator with 4-C cap and daily active-lineup scoring.
* **`compare_early_season.py`** — Live in-progress season evaluator (scales `ref_games` proportionally to games elapsed so far).

---

## Setup & Data Download (Windows PowerShell)

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Download historical seasons (example: 2015-16 through 2025-26)
.\.venv\Scripts\python.exe download.py 2015 2025 --adp-year-is start

# Download current/upcoming season (2026-27)
.\.venv\Scripts\python.exe download.py 2026 --adp-year-is start
```

---

## Usage (`draft_engine.py`)

### 1. Run Full Historical Backtest & 12-Team ESPN Draft Simulation
```powershell
# Default / ESPN scoring
.\.venv\Scripts\python.exe draft_engine.py --backtest

# Yahoo or Sleeper scoring (or custom JSON file / inline dict)
.\.venv\Scripts\python.exe draft_engine.py --backtest --scoring yahoo
.\.venv\Scripts\python.exe draft_engine.py --backtest --scoring sleeper
```

### 2. Stage 1 — Pre-ADP Board (June–August, 0% ADP + Live Injury Caps)
```powershell
.\.venv\Scripts\python.exe draft_engine.py --generate-board 2026-27 --mode pure_stats --use-live-injuries --scoring default
```
* Works even if `data/2026-27/adp.csv` does not exist yet.
* Optional: pass `--rookies-csv rookies.csv` (`player,draft_pick,team,age,pos,is_redshirt`) right after the NBA Draft before ESPN updates `player_core`.
* Optional: pass `--opening-night YYYY-MM-DD` (defaults to Oct 21 of the draft year; set to today's date if running mid-season).

### 3. Stage 2 — Draft-Day Board (September–October, ADP-Aware Champion)
```powershell
.\.venv\Scripts\python.exe draft_engine.py --generate-board 2026-27 --mode residual --lam 0.50 --use-live-injuries --scoring default
```
* Uses `--mode auto` by default (automatically runs `residual` if `adp.csv` is present, else `pure_stats`).
* Optional: manually remove confirmed season-enders via `--exclude-players "Player A,Player B"` or supply offline return dates via `--injuries-csv injuries.csv` (`player,status,return_date,notes`).

### 4. Too-Early In-Season Audit (2026–27)
```powershell
.\.venv\Scripts\python.exe compare_early_season.py --board-lambda 0.50
```

---

## Verified Backtest Results

Evaluated chronologically across **8 out-of-sample seasons** (6 CV: `2018-19` to `2023-24`; 2 unseen Holdout: `2024-25`, `2025-26`) using **1,152 simulated 12-team $\times$ 13-round ESPN snake drafts per strategy** (exact daily 10-starter + 3-bench scoring, max 4 primary centers, Round 1 straight ADP, Rounds 2–13 $\text{Uniform}[-2, +2]$ for the 11 villains):

### Default / ESPN Points Scoring (`draft_engine.py --backtest`)

| Strategy | All-8-Yr ESPN Diff | Holdout Diff (288 Drafts) | Avg Rank (All / HO) | Win % (All / HO) | Top-4 % (All / HO) | Beat Field % (All / HO) | 0g Busts (H / F) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `market` (strict ADP) | $+66.7$ ($+0.26\%$) | $-107.5$ ($-0.42\%$) | $6.41$ / $6.56$ | $7.7\%$ / $6.4\%$ | $33.7\%$ / $35.1\%$ | $51.7\%$ / $51.4\%$ | $0.17$ / $0.20$ |
| `pure_stats` (0% ADP, no injury news) | $-4,908.0$ ($-19.18\%$) | $-6,658.9$ ($-25.68\%$) | $11.43$ / $11.58$ | $0.0\%$ / $0.0\%$ | $0.0\%$ / $0.7\%$ | $0.3\%$ / $0.7\%$ | $1.42$ / $0.09$ |
| `pure_stats_news_cut` (0% ADP + collapse cut) | $-49.7$ ($-0.20\%$) | $-1,327.4$ ($-5.21\%$) | $6.47$ / $7.78$ | $11.4\%$ / $1.0\%$ | $34.5\%$ / $21.5\%$ | $47.6\%$ / $38.5\%$ | $0.25$ / $0.19$ |
| `blend_adp_50` (50/50 stats + ADP) | $+751.6$ ($+2.99\%$) | $-107.3$ ($-0.42\%$) | $5.47$ / $6.69$ | $16.2\%$ / $3.1\%$ | $47.2\%$ / $30.2\%$ | $61.8\%$ / $52.4\%$ | $0.20$ / $0.20$ |
| **`residual_unified` ($\lambda = 0.50$)** | **$+1,533.7$ ($+6.12\%$)** | **$+2,422.2$ ($+9.63\%$)** | **$4.40$ / $3.41$** | **$20.5\%$ / $25.7\%$** | **$58.5\%$ / $72.6\%$** | **$77.0\%$ / $89.6\%$** | **$0.03$ / $0.21$** |

* **8-for-8 winning seasons** for `residual_unified` under default scoring ($+452.5$ to $+2,549.0$ ESPN FPTS) and **0.00 zero-game busts** across all 288 holdout drafts.
* **Yahoo Scoring Cross-Check (`--scoring yahoo`):** `residual_unified` ($\lambda = 0.50$) delivered **$+1,604.7$ FPTS ($+6.53\%$)** across all 8 seasons ($+2,358.7$ in holdout, **$24.7\%$** win rate overall / **$27.4\%$** in holdout, $0.00$ holdout 0-game busts).
* **Too-Early Live 2026–27 Check (241 games through Nov 21, 2026 — 19.6% of season):** `unified_equal` ($\lambda = 0.50$) leads the 11-team field by **$+272.7$ ESPN FPTS ($+5.19\%$, $+1,389$ full-season pace)** with a **$27.1\%$ win rate** and $0.12$ vs. $0.53$ zero-game busts.