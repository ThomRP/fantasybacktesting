# Project Handoff Context — The Draft Model (`fantasybacktesting`)

Paste this file into **Google AI Studio System Instructions** (or your opening prompt) alongside `draft_engine.py` / `cat9_engine.py` to continue development seamlessly.

---

## 1. User Goals & Design Philosophy

* **Product Name & Positioning:** **"The Draft Model"** — a sleek, dark-mode, self-contained HTML pre-draft NBA fantasy projection dashboard (Points & Unified 9-Cat) with strict out-of-sample game-by-game historical replay and live tracking.
* **Strict Scope:** Focus **exclusively on pre-draft projections**. Do **not** build mid-season waiver/trade/streaming optimizers and **never** use any data after Game 1 of a target season.
* **League-Size-Independent Evaluation:** The dashboard UI dropped 12-team/14-team draft simulations in favor of universal **Pick-Band Edge** metrics (`Picks 1–25`, `26–60`, `61–100`, `101–150`, `151–225`) — **Calls Right %** (`50%` = coin flip) and **Targets vs. Fades** edge (`% FPTS` / `Z`) — that apply equally to any league size and avoid deterministic draft-bot cliff noise.
* **Plain-English, Verifiable UI:** Avoid cryptic stat jargon (`r`, `neg-edge`, `cutoff swaps`). Keep `Season Breakdown` open by default with **click-to-inspect side-by-side comparison drawers** showing exact player receipts (`ADP`, `Edge`, `GP`, `FPPG`, `FPTS`/`Act Z`, `vs ADP`, and season `Finish` rank + `Avg ADP` / `Avg Finish` footers). Keep Holdout seasons styled uniformly with other past seasons (`Past Seasons` vs. `Live`).
* **Equalized Live Tracking:** Live `2026–27` game logs are dynamically truncated to `min_team_games` shared by all 30 NBA teams (currently **16 equalized games** across 270 completed games through `2026-11-24`) so asymmetric NBA schedules don't distort early-season totals.
* **Two-Stage Architecture:**
  1. **Stage 1 (`--mode pure_stats`, June–August, 0% ADP):** Standalone pre-ADP projections with optional live ESPN injury return-date caps.
  2. **Stage 2 (`--mode residual`, September–October, Draft Season):** ADP-anchored residual champion (`residual_unified`, $\lambda = 0.50 \to 0.85$ late-ADP ramp) absorbing summer news while harvesting systematic draft-room biases.

---

## 2. Environment & Data Inventory

* **Local Workspace:** `C:\Users\thomr\Desktop\fasntasybascketest\fantasybacktesting` (Windows PowerShell, Python venv `.\.venv\Scripts\python.exe`).
* **Completed Seasons Downloaded:** `2006-07` through `2025-26` (historical draft pools `2008-09` through `2025-26`), plus in-progress `2026-27` (`270` completed games through `2026-11-24`, `min_team_games = 16`).
* **Data Layout (gitignored under `/data/`):**
  * `data/<YYYY-YY>/{adp.csv, games.csv, player_box_scores.csv, team_box_scores.csv}`
  * `data/player_core/` (ESPN bio/DOB/height/draft pick/round/year) and `data/player_history/` (career appearances back to 1993–94)
  * `data/vegas_preseason_odds.csv` (verified Basketball-Reference preseason Vegas Win Totals via `download_vegas_odds.py`)
  * `data/preseason_rosters.csv` (`9,683` player-season rows compiled from dated Basketball-Reference transaction logs `2008–09`–`2026–27` via `download_preseason_rosters.py`, strictly enforced `tx_date < opening_night`; validated `99.35%` accurate vs. actual Week-1 jerseys, `100.0%` on both Holdout seasons `2024–25` and `2025–26`).
* **CRITICAL LEAKAGE WARNINGS:**
  * Never use `experience_years` or `current_team_id` from `data/player_core/player_core_*.csv` (they reflect current 2026 values — survivorship/future trade leakage).
  * Never use `(TEAM)` abbreviations from historical FantasyPros `adp.csv` (FantasyPros overwrites team tags with later trades when scraped historically). Always use `data/preseason_rosters.csv` (`tx_date < opening_night`) for historical roster turnover.
  * Never pull mid-season live ESPN injuries retroactively onto a pre-draft board (only apply injury snapshots stamped before Opening Night).

---

## 3. Validated Modeling Architecture

### Target Variables
* **Points (`AdjFPPG`):** Availability-Adjusted FPPG, ref $= 65$ games, $\text{waiver} = 0.85 \times \text{FPPG}_{150}$ ($\ge 20$g qualifiers), 0-game busts included at $\text{waiver}$:
  $$\text{AdjFPPG} = \frac{\min(g, 65) \cdot \text{raw\_FPPG} + (65 - \min(g, 65)) \cdot \text{waiver}}{65}$$
* **9-Cat (`z_tot`):** Season-total 9-category Z-scores over top-180 qualifiers with volume-weighted FG%/FT% impact and $0.75\times$ inverted TO (signed interpolation preserving negative Z-scores past pick #90).

### Points Stage 2 Champion (`draft_engine.py`, `--mode residual`)
* **Returning Veterans (`37.5% Huber + 37.5% Structural Decomposition + 25% Shallow L1 GBDT`):**
  1. `huber_clean` (`HuberRegressor`, $\alpha = 25.0$, $\epsilon = 1.35$): 26 leak-free features — original 23 box-score/age/platform/collapse-hinge features PLUS **3 validated strictly pre-Opening-Night roster-turnover features** (`changed_team`, `net_vacated_pos_mpg`, `net_vacated_val`) loaded from `data/preseason_rosters.csv`.
  2. `decomposed_clean`: 3-part structural spline-Ridge decomposition ($E[\text{avail}] \cdot E[\text{MPG}] \cdot E[\text{FPPM}] + (1 - E[\text{avail}]) \cdot \text{waiver}$) calibrated onto `fppg_above_adp` via Huber (with `changed_team`, `net_vacated_pos_mpg`, `net_vacated_val` in MPG and `changed_team`, `net_vacated_val` in FPPM).
  3. `gbdt_shallow`: Heavily regularized `HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.03, max_iter=100, max_depth=2, min_samples_leaf=35, l2_regularization=20.0)`.
  * **Market-Collapse Hinge:** Splits `prior_fppg - mkt_expected_adj_fppg` at `0.38 * waiver` ($\approx 8$ FPPG) gated by `adp - prior_raw_rank >= 20` slots (`+0.60` buy moderate drops, `-0.50` cut catastrophic injury/role collapses).
* **Returning-Absence Veterans:** Empirical-Bayes shrunk median residual (`n / (n + 10)` toward `vet_peer_center`, capturing $\approx -2.95$ AdjFPPG rust/rest drag).
* **Incoming Rookies (`rookie_full_huber`, 11 features, $\alpha = 30.0$):** Captures the $-2.84$ AdjFPPG Rookie Hype Tax using `log_draft_pick`, `adp_minus_pick_expected`, `is_redshirt`, `rookie_age`, `team_vegas_win_pct`, and roster-age context.
* **Late-Round Shrinkage Ramp:** $\lambda = 0.50$ for ADP $\le 100$, ramping smoothly to $\lambda_{\text{late}} = 0.85$ over ADP `100–150`.

### Unified 9-Cat Champion (`cat9_engine.py`, `--to-weight 0.75`)
* **Balanced 9-Cat Residual Model:** Same 3-leg veteran ensemble + 11-feature rookie Huber + returning-absence drag, **without** roster-turnover features (explicitly rejected on Balanced 9-Cat after walk-forward ablation showed extra usage boosts counting stats while hurting `FG%`, `FT%`, and `TO`, leaving Balanced 9-Cat flat).
* **Per-Category Punt Heads (`proj_cat_z`):** Upgraded from uniform scalar shift to a **50/50 blend of 9 regularized `RidgeCV` per-category heads + Empirical-Bayes shape**, normalized so `proj_cat_z @ w9 == residual_proj` identically. Swept **5-for-5 Punt strategies** on both CV and untouched Holdout.

---

## 4. Current Verified Benchmarks (Do Not Regress Below These)

### League-Size-Independent Out-of-Sample Pick-Band Benchmarks
* **9-Yr Points (`board_2026_27.html`, 8 past yrs `2018–26` + Live `2026–27` at `16 GP`):**
  * Overall `1–150` Calls Right: **`58.9%`** (**`8/8` Beat ADP Yrs**) | Holdout residual $r = \mathbf{+0.231}$ (up from `+0.212`).
  * Pick Bands: `1–25` `59.2%` (`+3.6% FPTS`) · `26–60` `55.5%` (`+12.7%`) · `61–100` `52.5%` (`+8.0%`) · `101–150` **`61.0%` (`+28.2%`)** · `151–225` **`64.8%` (`+35.2% FPTS`)**.
  * **Live `2026–27` (`G 16/16`):** **`59.2%`** Calls Right (up from `56.6%`).
* **13-Yr Points (`board_13yr_2014_2027.html`, 12 past yrs `2014–26` + Live `2026–27` at `16 GP`):**
  * Overall `1–150` Calls Right: **`60.0%`** (**`12/12` Beat ADP Yrs — 100%!**).
  * Pick Bands: `1–25` `55.2%` (`+3.0%`) · `26–60` `58.6%` (`+9.3%`) · `61–100` `55.2%` (`+7.3%`) · `101–150` **`61.7%` (`+25.9%`)** · `151–225` **`61.0%` (`+27.1% FPTS`)**.
  * **Live `2026–27` (`G 16/16`):** **`58.2%`** Calls Right (up from `55.3%`).
* **Points Yahoo Scoring Cross-Check (`2016–17+`, 8 past yrs):**
  * Overall `1–150` Calls Right: **`59.92%`** (Holdout **`63.14%`**, `+1.52%` lift) | Holdout $r = \mathbf{+0.288}$.
* **Unified 9-Cat (`board_cat9_2026_27.html`, 8 past yrs `2018–26` + Live `2026–27` at `16 GP`):**
  * Overall `1–150` Calls Right: **`58.6%`** (**`8/8` Beat ADP Yrs**) | `151–225`: **`64.7%` (`+2.31 Z`)**.
  * **Punt Holdout (`2024–26`) Calls Right:** `Punt TO` **`52.2%`** (`+4.7%`), `Punt FT%+TO` **`54.5%`** (`+4.0%`), `Punt FT%` **`58.0%`**, `Punt FG%` **`60.0%`**, `Punt AST` **`62.6%`**.
* **Unit Tests:** `test_cat9_engine.py` (5), `test_injury_cap.py` (6), `test_reach_or_wait.py` (4) — all `15/15` passing (`0.002s`).

### Rejected / Completed Research Signals (Do Not Repeat)
* **Local Advanced Impact Stats (`BPM`, `VORP`, `On-Off`, `USG%`) in Stage 2:** Flat on Holdout (kept strictly opt-in via `--advanced-stats`).
* **Vegas Win Totals for Veterans:** `Avail` (`+0.07`) and `FPPM` (`-0.07`) effects cancel out; kept exclusively for **Incoming Rookies**.
* **Roster-Turnover Features on Balanced 9-Cat:** Extra volume hurts `TO` and `%` categories, leaving Balanced 9-Cat flat (`58.74%` vs. `58.49%`); kept exclusively in **Points**.
* **Pre-Draft NBA Schedule / Off-Nights / B2B & Historical Salaries/Contract Years:** Hurt or added variance out-of-sample.

---

## 5. Highest-ROI Remaining Research Directions

1. **Rookie Pre-NBA Box-Score Translation:** Our rookie model currently excels at fading hyped rookies as a class (`r_edge = +0.05–0.08` inside Top-150 ADP) because it has zero college/EuroLeague/G-League stats. Adding pre-NBA `STL%`, `BLK%`, `REB%`, `AST/TO`, and `FTA rate` is the #1 lever for rookie discrimination.
2. **Preseason Vegas Player Props (October Sharp Money):** Preseason season-average PTS/REB/AST lines & MIP/ROY odds capture training-camp rotation battles and quiet injuries faster than casual ADP.
3. **Opening-Night Injury Snapshot Automation:** Before next October's drafts (`2027–28`), capture and archive an Opening-Night-Eve ESPN injury snapshot (`data/2027-28/preseason_injuries.json`) to catch late-October injuries before stale ADP fully adjusts.

---

## 6. Working Rules for the AI Assistant

* Never introduce future data leakage (strict chronological expanding-window CV, untouched holdout seasons, `tx_date < opening_night` hard guard on roster transactions, never use `experience_years` or `current_team_id` from `player_core` or team tags from historical `adp.csv`).
* Keep 0-game injury busts in both training and evaluation — never silently drop injured players.
* Always validate any proposed model change strictly out-of-sample on **both CV and untouched Holdout** (Points Default, Yahoo, Extended Archive, and 9-Cat + Punt) and verify all 15 unit tests (`.\.venv\Scripts\python.exe -m unittest discover -p "test_*.py" -v`) before shipping.
* Keep dashboards built through `board_templates.py` (`build_board_page.py` / `build_cat9_board_page.py`) and compatible with Windows PowerShell commands.
