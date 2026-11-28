# The Draft Model (`fantasybacktesting`)

A concise technical reference for the zero-leakage NBA pre-draft projection engine and interactive dashboards. See [`README.md`](README.md) for setup and usage, and [`AGENTS.md`](AGENTS.md) for modeling rules, rejected experiments, and benchmark guardrails.

---

## Architecture Summary

* **Targets:**
  * **Points (`AdjFPPG`):** Availability-adjusted FPPG on a 65-game reliability scale ($\text{waiver} = 0.85 \cdot \text{FPPG}_{150}$, $\ge 20$g qualifiers; 0-game injury busts kept at waiver).
  * **9-Cat (`z_tot`):** Season-total 9-category Z-scores over top-180 qualifiers with volume-weighted FG%/FT% impact and $0.75\times$ inverted TO.
* **Stage 1 (`--mode pure_stats`, 0% ADP):** Full-NBA projection (`37.5% Huber + 37.5% Structural MPG×FPPM×Avail + 25% shallow L1 GBDT`) paired with optional live ESPN injury return-date caps (`--use-live-injuries`).
* **Stage 2 (`--mode residual`, Draft Season):** ADP-anchored residual champion combining the 3-leg veteran ensemble (with market-collapse injury hinge and late-ADP ramp $\lambda = 0.50 \to 0.85$), 11-feature Rookie Hype-Tax Huber model (with preseason Vegas win totals), Empirical-Bayes returning-absence drag, **pre-Opening-Night offseason roster-turnover features** (`changed_team`, `net_vacated_pos_mpg`, `net_vacated_val`, strictly `tx_date < opening_night` for Points), and **9 regularized `RidgeCV` per-category Punt heads** (for 9-Cat).

---

## Active Files

* `draft_engine.py` — Points Stage-1/Stage-2 engine (with pre-Opening-Night roster-turnover features, Rookie Vegas win totals, opt-in `--advanced-stats`, live injury caps, and draft simulator).
* `cat9_engine.py` — Unified 9-Cat engine (`0.75× TO`, Roto + H2H) with 9 per-category `RidgeCV` Punt heads.
* `board_templates.py` — Shared dark-mode UI templates (`The Draft Model`: 2-column Pick-Band Edge Hero, game-by-game replay chart, and click-to-inspect side-by-side comparison drawers with season `Finish` ranks).
* `build_board_page.py` / `board_2026_27.html` / `board_13yr_2014_2027.html` — 9-Yr (`2018–27`) and 13-Yr (`2014–27`) Points interactive dashboards.
* `build_cat9_board_page.py` / `board_cat9_2026_27.html` — 9-Cat interactive dashboard with client-side Punt chips.
* `download.py`, `download_vegas_odds.py`, `download_preseason_rosters.py`, `backtest_common.py`, `adp_names.py`, `backtest_rookies.py`, `backtest_rookie_plots.py`, `backtest_full_pool.py`, `compare_early_season.py` — Data ingestion and shared utilities.
* `test_cat9_engine.py`, `test_injury_cap.py`, `test_reach_or_wait.py` — Unit tests (`15/15` passing).

---

## Verified Out-of-Sample Benchmarks

* **9-Yr Points (`board_2026_27.html`, 8 past yrs `2018–26`):** **`58.9%`** Calls Right (`1–150`, **`8/8` Beat ADP Yrs**) | `Picks 101–150` **`61.0%` (`+28.2% FPTS`)** | `Picks 151–225` **`64.8%` (`+35.2% FPTS`)** | **Live `2026–27` (`16 equalized GP`):** **`59.2%`** Calls Right.
* **13-Yr Points (`board_13yr_2014_2027.html`, 12 past yrs `2014–26`):** **`60.0%`** Calls Right (`1–150`, **`12/12` Beat ADP Yrs**) | **Live `2026–27` (`16 equalized GP`):** **`58.2%`** Calls Right.
* **9-Cat Unified (`board_cat9_2026_27.html`, 8 past yrs `2018–26`):** **`58.6%`** Calls Right (`1–150`, **`8/8` Beat ADP Yrs**) | `Picks 151–225` **`64.7%` (`+2.31 Z`)** | Per-category `RidgeCV` Punt Holdout: `Punt TO` **`52.2%`** (`+4.7%`), `Punt FT%+TO` **`54.5%`** (`+4.0%`), `Punt FT%` **`58.0%`**.
