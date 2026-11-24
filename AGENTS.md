# Project Handoff Context — Fantasy Basketball Draft Engine (`fantasybacktesting`)

Paste this file into **Google AI Studio System Instructions** (or your opening prompt) alongside `draft_engine.py` to continue development seamlessly.

---

## 1. User Goals & Design Philosophy

* **Primary Objective:** Build a repeatable, leak-free, data-driven fantasy basketball draft engine that consistently beats consensus ADP in points leagues (ESPN-style 12-team, 13-round snake, 4-Center primary cap, 10 starters + 3 bench).
* **Strict Scope:** Focus **exclusively on pre-draft and draft-day decision-making**. Do **not** build mid-season waiver/trade/streaming optimizers—the user wants a controlled, repeatable edge in the draft room rather than chasing noisy mid-season variance.
* **No Subjective VORP:** Do not add positional VORP adjustments (apps disagree on dual-position tags). Enforce only objective roster guardrails: ESPN's **4-Center cap** (primary-listed `C`) and legal starting lineups (`PG, SG, SF, PF, C, G, F, 3xUTIL + 3 BE`).
* **Two-Stage Product Vision:**
  1. **Stage 1 (June–August, Pre-ADP):** Generate independent, 100% ADP-free projections available before the first fantasy draft opens, using live injury return-date estimates to handle offseason injuries.
  2. **Stage 2 (September–October, Draft Season):** Once consensus ADP stabilizes, switch to the ADP-anchored residual model (`residual_unified`, $\lambda = 0.50$) to absorb summer trades/role news while capturing systematic draft-room biases.
* **Commercial / Real-World Interest:** Potentially testing in Best Ball / cash points leagues or packaging as a lightweight points-league draft kit/companion (points leagues are underserved relative to 9-cat tools like Basketball Monster and Hashtag Basketball).

---

## 2. Environment & Data Inventory

* **Local Workspace:** `C:\Users\thomr\Desktop\fasntasybascketest\fantasybacktesting` (Windows PowerShell, Python venv `.\.venv\Scripts\python.exe`).
* **Completed Seasons Downloaded:** `2015-16` through `2025-26` (10 historical draft pairs `2016-17` through `2025-26`), plus in-progress `2026-27` (241 games through `2026-11-21`).
* **Data Layout:** `data/<YYYY-YY>/{adp.csv, games.csv, player_box_scores.csv, team_box_scores.csv}`, plus caches in `data/player_core/` (ESPN bio/DOB/height/draft pick/round/year) and `data/player_history/` (career appearances back to 1993–94).
* **Known Minor Data Notes:** Two historical regular-season games have missing box scores (`2016-17` game `400900132`, `2017-18` game `400975285`). Historical `adp.csv` team abbreviations on FantasyPros sometimes reflect later rosters—always prefer actual season-opener `team_id` from `player_box_scores.csv` for historical rookie team mapping. **CRITICAL LEAKAGE WARNING:** `experience_years` in `data/player_core/player_core_*.csv` reflects *current* (2026) career totals even in older files (survivorship leakage)—**never** use `experience_years` as a model feature.
* **Live Injury Endpoint:** Unauthenticated ESPN JSON at `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries` (cached to `data/<season>/espn_injuries_cache.json`; filter out healthy `status == "Active"` notes that have no `injury_type` or `returnDate`). Mid-season injury caps anchor to `ref_start = max(opening_night, as_of_date or date.today())` and scale remaining games proportionally (`--as-of-date` supported).

---

## 3. Validated Modeling Architecture (`draft_engine.py`)

### Target Variable
Availability-Adjusted FPPG (`AdjFPPG`), ref $= 65$ games, $\text{waiver} = 0.85 \times \text{FPPG}_{150}$ ($\ge 20$g qualifiers), 0-game busts included at $\text{waiver}$:

$$\text{AdjFPPG} = \frac{\min(g, 65) \cdot \text{raw\_FPPG} + (65 - \min(g, 65)) \cdot \text{waiver}}{65}$$

### Stage 2 Champion (`--mode residual`, Tier 1 + Tier 2 + Reach-vs.-Wait)
* **Returning Veterans (37.5% Huber + 37.5% Structural Decomposition + 25% Shallow L1 GBDT):**
  1. `huber_clean` (`HuberRegressor`, $\alpha = 25.0$, $\epsilon = 1.35$): 23 leak-free features (`log_adp`, `mkt_expected_adj_fppg`, `prior_minus_mkt_capped`, `market_collapse_excess`, `age_at_prior_break`, `age_under_24`, `age_over_30`, `prior_games`, `prior_mpg`, `prior_fppm`, `prior_start_pct`, `prior_post_break_mpg_chg`, `prior_tp_pct` [Empirical-Bayes shrunk 3P%], `age30_x_low_g`, `young_x_post_mpg`, `log_adp_x_age30`, `yoy_fppg_delta`, `yoy_mpg_delta`, `two_yr_weighted_fppg`, `two_yr_games_avg`, `no_prev2_season`, `adp_platform_std`, `espn_vs_avg`).
  2. `decomposed_clean`: 3-part structural spline-Ridge decomposition ($E[\text{avail}] \cdot E[\text{MPG}] \cdot E[\text{FPPM}] + (1 - E[\text{avail}]) \cdot \text{waiver}$, with `prior_post_break_mpg_chg` in MPG and `prior_tp_pct` in FPPM) calibrated onto `fppg_above_adp` via Huber.
  3. `gbdt_shallow`: Heavily regularized `HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.03, max_iter=100, max_depth=2, min_samples_leaf=35, l2_regularization=20.0)` capturing pairwise interactions robustly.
  * **Why `logistic_base` was dropped:** It suffered from survivorship bias once 0-game busts were kept, dragging injured stars onto the board and losing in Rounds 1–2.
  * **Market-Collapse Hinge (Critical):** Uses raw `prior_fppg - mkt_expected_adj_fppg` split at a scale-invariant threshold (`0.38 * waiver`, $\approx 8.0$ FPPG on ESPN) gated by `adp - prior_raw_rank >= 20` slots. Learned opposite signs (`prior_minus_mkt_capped = +0.60`, `market_collapse_excess = -0.50`): buys moderate market discounts while automatically cutting torn-Achilles/ACL stars without false-flagging healthy top-5 stars.
* **Returning-Absence Veterans (`returning_absence`):** Players returning from a 0-game missed season receive a zero-leakage Empirical-Bayes shrunk median residual (`n / (n + 10)` shrunk toward `vet_peer_center`) rather than a neutral `0` residual, capturing the historical ~`-2.95` AdjFPPG rust/rest drag.
* **Incoming Rookies (`rookie_full_huber`, 11 features, `HuberRegressor`, $\alpha = 30.0$):**
  * Features: `log_draft_pick`, `log_adp`, `mkt_expected_adj_fppg`, `adp_minus_pick_expected` (zero-leakage Rookie Hype Tax fit inside training folds), `is_redshirt`, `rookie_age`, `team_top8_avg_age`, `team_vegas_win_pct` (verified Basketball-Reference preseason Vegas Win Totals from `data/vegas_preseason_odds.csv`, auto-fallback to `team_prior_win_pct`), `team_vet30_min_share`, `team_prior_top150_count`, `espn_vs_avg`.
  * **Key Empirical Findings (10 seasons, 137 drafted rookies vs. 2,057 vets):**
    * Drafted rookies lose to ADP by $-2.84$ AdjFPPG ($27.7\%$ beat ADP) vs. $-1.18$ for veterans—a $-1.66$ rookie hype tax.
    * 2nd-round/undrafted rookies went $0/16$ ($-5.28$); contenders ($\ge .550$ win %) went $13.3\%$ ($-4.92$); age $> 21.5$ went $15.2\%$ ($-5.13$).
    * Positive drivers: `is_redshirt` ($+0.33$ weight, $62.5\%$ beat ADP), top-3 picks ($51.6\%$), losing teams (`team_vegas_win_pct` improved Rookie MAE 4-for-4 across Default & Yahoo CV + Holdout), and **older established veteran rosters** (`team_vet30_min_share = +0.30`, `team_top8_avg_age = +0.23`).
* **Separable Opt-In Extension (`--advanced-stats`, default `False`):**
  * Computes local `BPM`, `VORP`, shrunk `On-Off`, and `USG%` from regular-season box scores (`compute_advanced_box_metrics`). Helps Stage 1 (`pure_stats`, 0% ADP), but kept opt-in `False` by default because Stage-2 Holdout showed no draft-sim lift over the core model.
* **Draft Execution (`residual_unified` vs. `residual_wait`):**
  * Greedy drafting (`residual_unified`) uses $\lambda = 0.50$ for ADP $\le 100$, ramping smoothly to $\lambda_{\text{late}} = 0.85$ over ADP $100\text{–}150$ so high-edge `ADP 151–225` sleepers can climb into the Top-150 board.
  * **ADP-Aware Reach-vs.-Wait (`residual_wait`, `--wait-lam 0.85`):** Uses exact 2-pick survival EV (`choose_reach_or_wait_pick`: maximize $V(i) + \mathbb{E}[\max_{j \neq i,\, j \text{ survives } K} V(j)]$, with `surv_sigma = max(2.5, 0.7 * adp_var)`, `surv_margin = max(1.0, 0.4 * adp_var)`, and `min_wait_gain = 0.05` FPPG) to avoid double-damping edge while capturing free surplus when top targets survive to the next snake turn.

### Stage 1 Pre-ADP Engine (`--mode pure_stats`, 0% ADP)
* Trained strictly on Year 0 / Year $-1$ stats (top 260 prior AdjFPPG players with $\ge 15$g; incoming draftees pick $\le 45$) with zero ADP features, using the upgraded 3-leg veteran ensemble (`37.5% Huber + 37.5% Structural Decomposition + 25% shallow L1 GBDT`) including 100% ADP-free `prior_tp_pct`, `age30_x_low_g`, and `young_x_post_mpg`.
* **Key Lesson from Backtesting (Adverse Selection):** Although `pure_stats` beats Market ADP in Holdout Full-NBA Top-150 capture (`239/300` vs. `237/300`) and Rank MAE (`46.54` vs. `50.54`), drafting purely from `pure_stats` against 11 ADP villains loses $-4,410$ FPTS because villains skip offseason-injured/demoted players and Hero vacuums them up ($1.39$ zero-game busts/draft). Even knowing 100% of 0-game season-enders with 0% ADP still loses $-1,722$ FPTS (because ADP also prices summer trades, role demotions, and partial-season absences). Adding the injury collapse cut recovers to `+758.5` FPTS (Default) / `+826.1` FPTS (Yahoo), blending 50% ADP reaches `+1,156.1` / `+1,329.7`, while Stage 2 reaches **`+2,000.9` FPTS** (default) / **`+2,467.1` FPTS** (Yahoo).

---

## 4. Benchmark Summary (Do Not Regress Below These)

All numbers from `draft_engine.py --backtest --seed 42` (6 CV + 2 Holdout = 8 seasons, 1,152 12-team ESPN drafts/strategy, R1 straight ADP, R2–13 $\text{Uniform}[-2, +2]$, exact daily active lineups, 4-C primary cap):

* **Default Scoring — `residual_unified` (greedy, $\lambda = 0.50 \to 0.85$ late ramp):** $+1,693.0$ FPTS ($+6.75\%$) all 8 yrs | $+1,801.2$ ($+7.14\%$) holdout | Avg Rank $4.32$ ($4.36$ HO) | Win % $28.0\%$ ($29.2\%$ HO) | Top-4 % $59.6\%$ | 0g Busts $0.03$ vs. $0.21$ ($0.00$ HO).
* **Default Scoring — `residual_wait` (wait, $\lambda = 0.85$):** **$+1,966.5$ FPTS ($+7.85\%$)** all 8 yrs (8/8 winning seasons) | **$+2,431.0$ ($+9.66\%$)** holdout | Avg Rank $4.04$ ($3.82$ HO) | Win % $29.7\%$ ($36.8\%$ HO) | Top-4 % $62.9\%$ ($65.6\%$ HO) | 0g Busts $0.02$ vs. $0.21$ ($0.00$ vs. $0.18$ HO).
* **Yahoo Scoring — `residual_unified` (greedy, $\lambda = 0.50 \to 0.85$ late ramp):** $+2,134.0$ FPTS ($+8.70\%$) all 8 yrs | $+2,211.5$ ($+9.04\%$) holdout | Avg Rank $3.69$ | Win % $31.2\%$ | Top-4 % $68.8\%$ | 0g Busts $0.04$ vs. $0.21$ ($0.00$ HO) | Holdout Top-150 `250/300` (Rank MAE `39.84`).
* **Yahoo Scoring — `residual_wait` (wait, $\lambda = 0.85$):** **$+2,467.3$ FPTS ($+10.07\%$)** all 8 yrs | **$+2,451.1$ ($+10.03\%$)** holdout | Avg Rank $3.39$ ($3.48$ HO) | Win % $36.6\%$ ($29.5\%$ HO) | Top-4 % $72.3\%$ ($70.1\%$ HO) | 0g Busts $0.04$ vs. $0.21$ ($0.00$ vs. $0.18$ HO).
* **Live 2026–27 Zero-Leakage Check (through `2026-11-21`, ~19.6% of season):** Top-150 correlation $r(\text{pred\_edge}, \text{actual\_vs\_ADP}) = +0.145$ (**9/9 out-of-sample seasons** positive `2018-19` through `2026-27`); Stage-1 full-NBA Top-150 captured `110/150` (with injury guard, `16/32 = 50%` `ADP > 150` hit rate); Stage-2 `ADP 151–225` positive-edge sleepers hit `10/14 (71.4%)` Top-150 vs. `14/63 (22.2%)` for negative-edge.
* **Interactive Replay Dashboard:** `build_board_page.py` generates bright, streamlined offline `board_2026_27.html` covering all 9 out-of-sample seasons (6 CV + 2 Holdout + Live `2026-27`) with a 4 games/sec real-time game-by-game replay slider and live SVG trajectory chart.
* **Unit Tests:** `test_injury_cap.py` (6 tests) and `test_reach_or_wait.py` (4 tests) — all 10 passing via `.\.venv\Scripts\python.exe -m unittest -v test_injury_cap.py test_reach_or_wait.py`.

### Rejected / Completed Research Signals (Do Not Repeat)
* **Local Advanced Impact Stats (`BPM`, `VORP`, `On-Off`, `USG%`) in Stage 2:** ADP already prices impact; flat on Holdout (kept strictly opt-in via `--advanced-stats`). `rel_ts` and `stocks36` carry strong Center bias and hurt drafts.
* **Vegas Win Totals for Veterans:** `Avail` (`+0.07`) and `FPPM` (`-0.07`) effects cancel out; kept exclusively for **Incoming Rookies**.
* **Pre-Draft NBA Schedule / Off-Nights / B2B:** Bench overflow in 10-starter + 3-bench daily ESPN leagues is only `0.40%–0.53%` of season FPTS; adjusting for schedule hurts talent (`-152` to `-1,447` FPTS).
* **ESPN Historical Salaries & CBA Rookie-Scale Contract Years:** ESPN historical `/nba/salaries` overwrites mid-season buyouts (buyout leakage). Even after buyout scrubbing, position-neutral salary improved Default CV (`+113` Wait) but every salary variant lost `-105` to `-515` FPTS on Yahoo CV Wait (penalizing cheap high-block/rebound Centers). 1st-Rd Yr 4 RFA contract years beat ADP (`49.3%`, `+0.50`–`+0.56` error) but sample size (`N = 75`) added CV variance.

---

## 5. High-Value Next Steps

1. **Tier 3 External Pre-Draft Signals (Highest Remaining Alpha Ceiling):**
   * Box-score + ADP signals are now near their natural ceiling without overfitting. Fresh orthogonal alpha requires external pre-season data: **Vegas Season Win Totals** (tanking/competitiveness risk), **Pre-season Player Prop Lines** (sharper minutes/usage pricing after summer trades), and clean **September depth-chart snapshots** (for true `changed_team` / `vacated_usage` features).
2. **Interactive Draft-Room Companion / UI:**
   * Build a lightweight CLI or single-file HTML/React draft assistant powered by `draft_engine.py`'s exported CSV + `choose_reach_or_wait_pick`: mark drafted players off in real time, track the 4-C primary cap and 10 starting slots, display `Edge vs. Peers`, `CHECK_INJURY` / `INJ_CAP` badges, and highlight optimal reach-vs.-wait picks on the clock.
3. **Best Ball Stress-Test:**
   * Test the upgraded engine under Underdog / Drafters Best Ball scoring presets and wider draft-room variance ($\text{Uniform}[-5, +5]$ or $\mathcal{N}(0, 4)$).

---

## 6. Working Rules for the AI Assistant

* Never introduce future data leakage (strict chronological expanding-window CV and untouched holdout seasons; never use `experience_years` from `player_core`).
* Keep 0-game injury busts in both training and evaluation—never silently drop injured players.
* Always validate any proposed change against `draft_engine.py --backtest` (both default and Yahoo scoring) and unit tests (`test_injury_cap.py`, `test_reach_or_wait.py`), and compare against the `+1,725.8` / `+2,000.9` (default) and `+2,161.1` / `+2,467.1` (Yahoo) baselines.
* Keep `draft_engine.py` self-contained and compatible with Windows PowerShell commands (`.\.venv\Scripts\python.exe ...`).