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
* **Known Minor Data Notes:** Two historical regular-season games have missing box scores (`2016-17` game `400900132`, `2017-18` game `400975285`). Historical `adp.csv` team abbreviations on FantasyPros sometimes reflect later rosters—always prefer actual season-opener `team_id` from `player_box_scores.csv` for historical rookie team mapping.
* **Live Injury Endpoint:** Unauthenticated ESPN JSON at `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries` (cached to `data/<season>/espn_injuries_cache.json`; filter out healthy `status == "Active"` notes that have no `injury_type` or `returnDate`).

---

## 3. Validated Modeling Architecture (`draft_engine.py`)

### Target Variable
Availability-Adjusted FPPG (`AdjFPPG`), ref $= 65$ games, $\text{waiver} = 0.85 \times \text{FPPG}_{150}$ ($\ge 20$g qualifiers), 0-game busts included at $\text{waiver}$:

$$\text{AdjFPPG} = \frac{\min(g, 65) \cdot \text{raw\_FPPG} + (65 - \min(g, 65)) \cdot \text{waiver}}{65}$$

### Stage 2 Champion (`--mode residual`, $\lambda = 0.50$, Best Draft-Room Performance)
* **Returning Veterans (50/50 Fixed Equal Ensemble — beat OOF stacking in live 2026–27):**
  1. `huber_clean` (`HuberRegressor`, $\alpha = 25.0$, $\epsilon = 1.35$): 18 features (`log_adp`, `mkt_expected_adj_fppg`, `prior_minus_mkt_capped`, `market_collapse_excess`, `age_at_prior_break`, `age_under_24`, `age_over_30`, `prior_games`, `prior_mpg`, `prior_fppm`, `prior_start_pct`, `yoy_fppg_delta`, `yoy_mpg_delta`, `two_yr_weighted_fppg`, `two_yr_games_avg`, `no_prev2_season`, `adp_platform_std`, `espn_vs_avg`).
  2. `decomposed_clean`: 3-part structural spline-Ridge decomposition ($E[\text{avail}] \cdot E[\text{MPG}] \cdot E[\text{FPPM}] + (1 - E[\text{avail}]) \cdot \text{waiver}$) calibrated onto `fppg_above_adp` via Huber.
  * **Why `logistic_base` was dropped:** It suffered from survivorship bias once 0-game busts were kept, dragging injured stars onto the board and losing in Rounds 1–2.
  * **Market-Collapse Hinge (Critical):** Uses raw `prior_fppg - mkt_expected_adj_fppg` split at a scale-invariant threshold (`0.38 * waiver`, $\approx 8.0$ FPPG on ESPN) gated by `adp - prior_raw_rank >= 20` slots. Learned opposite signs (`prior_minus_mkt_capped = +0.58`, `market_collapse_excess = -0.51`): buys moderate market discounts while automatically cutting torn-Achilles/ACL stars (`Tatum`, `Haliburton`, `Lillard`, `Kyrie`, `Dejounte Murray` in 2025–26; `Butler`, `KPJ`, `Westbrook` in 2026–27) without false-flagging healthy top-5 stars (`Giannis`).
* **Incoming Rookies (`rookie_full_huber`, 11 features, `HuberRegressor`, $\alpha = 30.0$):**
  * Features: `log_draft_pick`, `log_adp`, `mkt_expected_adj_fppg`, `adp_minus_pick_expected` (zero-leakage Rookie Hype Tax fit inside training folds), `is_redshirt`, `rookie_age`, `team_top8_avg_age`, `team_prior_win_pct`, `team_vet30_min_share`, `team_prior_top150_count`, `espn_vs_avg`.
  * **Key Empirical Findings (10 seasons, 137 drafted rookies vs. 2,057 vets):**
    * Drafted rookies lose to ADP by $-2.84$ AdjFPPG ($27.7\%$ beat ADP) vs. $-1.18$ for veterans—a $-1.66$ rookie hype tax.
    * 2nd-round/undrafted rookies went $0/16$ ($-5.28$); contenders ($\ge .550$ win %) went $13.3\%$ ($-4.92$); age $> 21.5$ went $15.2\%$ ($-5.13$).
    * Positive drivers: `is_redshirt` ($+0.33$ weight, $62.5\%$ beat ADP), top-3 picks ($51.6\%$), losing teams (`team_prior_win_pct = -0.30`), and **older established veteran rosters** (`team_vet30_min_share = +0.30`, `team_top8_avg_age = +0.23`—aging veterans rest/get hurt/traded on non-contenders, opening rookie minutes).
* **Global Scaling ($\lambda = 0.50$):** Fixed $\lambda = 0.50$ outperformed $\lambda = 0.80$ in live 2026–27 testing by preventing aggressive late-round over-reaching while preserving full swap quality.

### Stage 1 Pre-ADP Engine (`--mode pure_stats`, 0% ADP)
* Trained strictly on Year 0 / Year $-1$ stats (top 260 prior AdjFPPG players with $\ge 15$g; incoming draftees pick $\le 45$) with zero ADP features.
* **Key Lesson from Backtesting (Adverse Selection):** Although `pure_stats` beat Market ADP in Holdout Global Top-150 Rank MAE ($46.33$ vs. $50.54$), drafting purely from `pure_stats` against 11 ADP villains lost $-4,908$ FPTS because villains skip offseason-injured/demoted players and Hero vacuums them up ($1.42$ zero-game busts/draft). Adding a news/injury collapse cut recovers $+4,858$ FPTS (`-49.7` vs. field), and blending 50% ADP reaches $+751.6$ FPTS, while Stage 2 `residual_unified` reaches **$+1,533.7$ FPTS** (default) / **$+1,604.7$ FPTS** (Yahoo).

---

## 4. Benchmark Summary (Do Not Regress Below These)

All numbers from `draft_engine.py --backtest` (6 CV + 2 Holdout = 8 seasons, 1,152 12-team ESPN drafts/strategy, R1 straight ADP, R2–13 $\text{Uniform}[-2, +2]$, exact daily active lineups, 4-C primary cap):

* **Default Scoring (`residual_unified`, $\lambda = 0.50$):** $+1,533.7$ FPTS ($+6.12\%$) all 8 yrs | $+2,422.2$ ($+9.63\%$) holdout | Avg Rank $4.40$ ($3.41$ HO) | Win % $20.5\%$ ($25.7\%$ HO) | Top-4 % $58.5\%$ ($72.6\%$ HO) | 0g Busts $0.03$ vs. $0.21$ ($0.00$ vs. $0.18$ HO).
* **Yahoo Scoring (`residual_unified`, $\lambda = 0.50$):** $+1,604.7$ FPTS ($+6.53\%$) all 8 yrs | $+2,358.7$ ($+9.66\%$) holdout | Avg Rank $4.30$ ($3.44$ HO) | Win % $24.7\%$ ($27.4\%$ HO) | Top-4 % $59.5\%$ ($70.5\%$ HO) | 0g Busts $0.05$ vs. $0.21$ ($0.00$ vs. $0.18$ HO).
* **Live 2026–27 Check (`compare_early_season.py --board-lambda 0.50`, through 2026-11-21):** $+272.7$ FPTS ($+5.19\%$, $+1,389$ full-season pace) | Win % $27.1\%$ | 0g Busts $0.12$ vs. $0.53$.

---

## 5. Known Gotchas & High-Value Next Steps

1. **Fix Mid-Season Injury Reference Date (Quick Bugfix):**
   * In `draft_engine.py` -> `evaluate_injury_availability_cap`, missed games are currently computed as `(ret_date - opening_night).days`. That is exact pre-season (August–October), but if `--use-live-injuries` is run *after* Opening Night (e.g., Nov 23), players returning tomorrow get penalized for October games they already played.
   * **Fix:** Anchor to `ref_start = max(op_date, date.today())` when running live mid-season (or add an explicit `--as-of-date` argument) and scale remaining season games accordingly.
2. **ADP-Aware "Reach vs. Wait" Draft Optimizer (High-Value Next Feature):**
   * Currently Hero greedily drafts the highest-projected available player on the board. In a real snake draft, if Hero loves a player at ADP 115 (ranked #88 by our model), reaching at pick 88 wastes surplus value if that player has an $85\%$ probability of surviving to Hero's next pick at #105.
   * Build a survival-probability pick optimizer (`EV_now` vs. `EV_if_wait` over the $K$ picks until Hero's next turn given opponent ADP noise) and benchmark it directly in `draft_engine.py`'s 12-team simulator.
3. **Interactive Draft-Room Companion / UI:**
   * Build a lightweight CLI or single-file HTML/React draft assistant powered by `draft_engine.py`'s exported CSV: mark drafted players off in real time, track the 4-C primary cap and 10 starting slots, display `Edge vs. Peers`, `CHECK_INJURY` / `INJ_CAP` badges, and highlight optimal reach-vs.-wait picks on the clock.
4. **Best Ball Stress-Test:**
   * Since our draft-only, no-waivers simulator matches **NBA Best Ball** rules closely (Underdog / Drafters), test the engine under Best Ball scoring presets and wider draft-room variance ($\text{Uniform}[-5, +5]$ or $\mathcal{N}(0, 4)$).

---

## 6. Working Rules for the AI Assistant

* Never introduce future data leakage (strict chronological expanding-window CV and untouched holdout seasons).
* Keep 0-game injury busts in both training and evaluation—never silently drop injured players.
* Always validate any proposed change against the 12-team ESPN daily-lineup simulator (`draft_engine.py --backtest`) and compare against the `+1,533.7` FPTS (default) / `+1,604.7` FPTS (Yahoo) baseline.
* Keep `draft_engine.py` self-contained and compatible with Windows PowerShell commands (`.\.venv\Scripts\python.exe ...`).