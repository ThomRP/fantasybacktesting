# Fantasy Basketball Draft Engine (`fantasybacktesting`)

A concise technical reference for the leak-free fantasy basketball draft engine. See [`README.md`](README.md) for full setup and usage instructions, and [`AGENTS.md`](AGENTS.md) for modeling rules and benchmark guardrails.

---

## Architecture Summary

* **Target (`AdjFPPG`):** Availability-adjusted FPPG on a 65-game reference scale ($\text{waiver} = 0.85 \cdot \text{FPPG}_{150}$, $\ge 20$g qualifiers; 0-game injury busts kept at waiver).
* **Stage 1 (`--mode pure_stats`, 0% ADP):** Full-NBA projection (`37.5% Huber + 37.5% Structural MPG×FPPM×Avail + 25% shallow L1 GBDT`) using Year 0 / Year −1 box scores, shrunk 3P% (`prior_tp_pct`), age/usage interactions (`age30_x_low_g`, `young_x_post_mpg`), and rookie draft/team context, paired with live ESPN injury return-date caps (`--use-live-injuries`).
* **Stage 2 (`--mode residual`, Draft Season):** ADP-anchored residual champion combining the 3-leg veteran ensemble (with scale-invariant market-collapse injury hinge), 11-feature Rookie Hype-Tax Huber model, Empirical-Bayes returning-absence bias, late-ADP sleeper ramp ($\lambda = 0.50 \to 0.85$ over ADP `100–150`), and ADP-aware 2-pick Reach-vs.-Wait survival optimizer (`residual_wait`, $\lambda = 0.85$).

---

## Active Files

* `draft_engine.py` — Unified Stage-1/Stage-2 projection engine, live injury fetcher, and 12-team ESPN draft simulator.
* `build_board_page.py` / `board_2026_27.html` — Dark-mode multi-season (6 CV + 2 Holdout + Live `2026–27`) interactive Top-150 & real-time 4 games/sec replay dashboard.
* `download.py`, `backtest_common.py`, `adp_names.py`, `backtest_rookies.py`, `backtest_rookie_plots.py`, `backtest_full_pool.py`, `compare_early_season.py` — Data ingestion and shared evaluation utilities.
* `test_injury_cap.py`, `test_reach_or_wait.py` — Unit tests (10/10 passing).

---

## Verified Benchmarks (`draft_engine.py --backtest --seed 42`)

1,152 simulated 12-team × 13-round ESPN snake drafts/strategy across 6 CV (`2018–19`–`2023–24`) + 2 untouched Holdout (`2024–25`, `2025–26`) seasons:

* **Default Scoring — `residual_unified`:** `+1,725.8` FPTS (`+6.88%`) all 8 yrs | `+1,885.1` (`+7.47%`) Holdout | Avg Rank `4.29` | Win % `28.0%` | 0g Busts `0.03` (`0.00` HO).
* **Default Scoring — `residual_wait`:** **`+2,000.9` FPTS (`+7.99%`)** all 8 yrs (`8/8` winning seasons) | **`+2,532.7` (`+10.06%`)** Holdout | Avg Rank `4.01` (`3.70` HO) | Win % `29.7%` (`37.2%` HO) | 0g Busts `0.02` (`0.00` HO).
* **Yahoo Scoring — `residual_unified`:** `+2,161.1` FPTS (`+8.81%`) all 8 yrs | `+2,216.1` (`+9.06%`) Holdout | Holdout Top-150 `250/300` (Rank MAE `39.86`).
* **Yahoo Scoring — `residual_wait`:** **`+2,467.1` FPTS (`+10.07%`)** all 8 yrs | **`+2,426.7` (`+9.93%`)** Holdout | Avg Rank `3.39` | Win % `36.6%` | Top-4 `72.3%` | Holdout Top-150 `250/300` (Rank MAE `38.41`).
* **Live `2026–27` Zero-Leakage Check (241 games through `2026-11-21`):** Top-150 `r = +0.142` (positive in all 4 ADP bands; `9/9` seasons positive `2018–27`), H2H flips `190/343 (55.4%)`, `ADP 151–225` positive-edge sleeper hit rate `10/14 (71.4%)` vs. `14/63 (22.2%)`.
