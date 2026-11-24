# Fantasy Basketball Draft Engine (`fantasybacktesting`)

A concise technical reference for the leak-free fantasy basketball draft engine. See [`README.md`](README.md) for full setup and usage instructions, and [`AGENTS.md`](AGENTS.md) for modeling rules and benchmark guardrails.

---

## Architecture Summary

* **Target (`AdjFPPG`):** Availability-adjusted FPPG on a 65-game reference scale ($\text{waiver} = 0.85 \cdot \text{FPPG}_{150}$, $\ge 20$g qualifiers; 0-game injury busts kept at waiver).
* **Stage 1 (`--mode pure_stats`, 0% ADP):** Full-NBA projection (`37.5% Huber + 37.5% Structural MPG×FPPM×Avail + 25% shallow L1 GBDT`) using Year 0 / Year −1 box scores, shrunk 3P% (`prior_tp_pct`), age/usage interactions (`age30_x_low_g`, `young_x_post_mpg`), and rookie draft/team context, paired with live ESPN injury return-date caps (`--use-live-injuries`).
* **Stage 2 (`--mode residual`, Draft Season):** ADP-anchored residual champion combining the 3-leg veteran ensemble (with scale-invariant market-collapse injury hinge), 11-feature Rookie Hype-Tax Huber model, Empirical-Bayes returning-absence bias, late-ADP sleeper ramp ($\lambda = 0.50 \to 0.85$ over ADP `100–150`), and ADP-aware 2-pick Reach-vs.-Wait survival optimizer (`residual_wait`, $\lambda = 0.85$).

---

## Active Files

* `draft_engine.py` — Unified Stage-1/Stage-2 projection engine (with Rookie preseason Vegas win totals & opt-in `--advanced-stats`), live injury fetcher, and 12-team ESPN draft simulator.
* `build_board_page.py` / `board_2026_27.html` — Bright, streamlined multi-season (6 CV + 2 Holdout + Live `2026–27`) interactive Top-150 & real-time 4 games/sec replay dashboard.
* `download.py`, `download_vegas_odds.py`, `backtest_common.py`, `adp_names.py`, `backtest_rookies.py`, `backtest_rookie_plots.py`, `backtest_full_pool.py`, `compare_early_season.py` — Data ingestion (including verified preseason Vegas win totals) and shared evaluation utilities.
* `test_injury_cap.py`, `test_reach_or_wait.py` — Unit tests (10/10 passing).

---

## Verified Benchmarks (`draft_engine.py --backtest --seed 42`)

1,152 simulated 12-team × 13-round ESPN snake drafts/strategy across 6 CV (`2018–19`–`2023–24`) + 2 untouched Holdout (`2024–25`, `2025–26`) seasons:

* **Default Scoring — `residual_unified`:** `+1,693.0` FPTS (`+6.75%`) all 8 yrs | `+1,801.2` (`+7.14%`) Holdout | Avg Rank `4.32` | Win % `28.0%` | 0g Busts `0.03` (`0.00` HO).
* **Default Scoring — `residual_wait`:** **`+1,966.5` FPTS (`+7.85%`)** all 8 yrs (`8/8` winning seasons) | **`+2,431.0` (`+9.66%`)** Holdout | Avg Rank `4.04` (`3.82` HO) | Win % `29.7%` (`36.8%` HO) | 0g Busts `0.02` (`0.00` HO).
* **Yahoo Scoring — `residual_unified`:** `+2,134.0` FPTS (`+8.70%`) all 8 yrs | `+2,211.5` (`+9.04%`) Holdout | Holdout Top-150 `250/300` (Rank MAE `39.84`).
* **Yahoo Scoring — `residual_wait`:** **`+2,467.3` FPTS (`+10.07%`)** all 8 yrs | **`+2,451.1` (`+10.03%`)** Holdout | Avg Rank `3.39` (`3.48` HO) | Win % `36.6%` | Top-4 `72.3%` | Holdout Top-150 `250/300` (Rank MAE `38.40`).
* **Live `2026–27` Zero-Leakage Check (241 games through `2026-11-21`):** Top-150 `r = +0.145` (`9/9` seasons positive `2018–27`), `ADP 151–225` positive-edge sleeper hit rate `10/14 (71.4%)` vs. `14/63 (22.2%)`.
