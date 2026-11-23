"""Test market, uniform, and band-specific strategies for a top-150 draft board."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from backtest_common import (
    adp_slot_points,
    draft_outcomes,
    name_key,
    read_adp,
    read_csv,
    season_pairs,
    write_csv,
)


BOARD_SIZE = 150
BANDS = ((1, 25), (26, 50), (51, 75), (76, 100), (101, 125), (126, 175))
TARGETS_PER_BAND = 5
BOX_SCORE_RULES = ("market", "projected_fppg", "projected_value")
SAVED_RULES = ("logistic_base", "value_base")


def saved_scores(data):
    scores = {}
    for split, filename in (("holdout", "backtest_value_results.csv"), ("cv", "backtest_value_cv_results.csv")):
        path = data / filename
        if path.is_file():
            for row in read_csv(path):
                key = split, row["draft_season"], row["athlete_id"], float(row["adp"])
                scores[key] = {rule: float(row[f"{rule}_score"]) for rule in SAVED_RULES}
    return scores


def candidate_pools(data):
    forecasts = defaultdict(dict)
    for row in read_csv(data / "backtest_fppg_predictions.csv"):
        if not row["training_through"] < row["draft_season"]:
            raise ValueError(f"{row['draft_season']}: forecast was not trained on earlier seasons")
        forecasts[row["draft_season"]][row["athlete_id"]] = row
    scores = saved_scores(data)

    for _, _, following, (_, prior), (_, actual) in season_pairs(data, split_at_all_star=False):
        predictions = forecasts.get(following.name)
        if not predictions:
            continue
        matches, _ = draft_outcomes(following, prior, {}, match_prior_only=True, rank_per_game=True)
        prior_ids = {name_key(name): athlete_id for name, athlete_id, _ in matches}
        outcomes, _ = draft_outcomes(following, prior, actual, rank_per_game=True)
        outcomes_by_name = {name_key(name): outcome for name, _, outcome in outcomes}
        prior_leaderboard = sorted((player.total_points / player.total_games for player in prior.values()), reverse=True)

        market = {}
        for name, adp in sorted(read_adp(following / "adp.csv"), key=lambda item: (item[1], item[0].casefold())):
            market.setdefault(name_key(name), (name, adp))

        pool = []
        for market_rank, (key, (name, adp)) in enumerate(list(market.items())[:BANDS[-1][1]], start=1):
            prediction = predictions.get(prior_ids.get(key))
            method_scores = {}
            if prediction:
                method_scores = {
                    "projected_fppg": float(prediction["ridge_fppg_prediction"]),
                    "projected_value": float(prediction["ridge_box_score_prediction"]) - adp_slot_points(prior_leaderboard, adp),
                }
                score_key = prediction["split"], following.name, prediction["athlete_id"], adp
                method_scores.update(scores.get(score_key, {}))
            outcome = outcomes_by_name.get(key, {})
            pool.append({
                "draft_season": following.name,
                "player": name,
                "adp": adp,
                "market_rank": market_rank,
                "scores": method_scores,
                "actual_fppg": outcome.get("next_fppg"),
                "actual_rank": outcome.get("next_rank"),
                "beat_adp": outcome.get("beat_adp"),
                "fppg_above_adp": outcome.get("fppg_above_adp"),
            })
        if len(pool) >= BOARD_SIZE:
            yield following.name, pool
        else:
            print(f"{following.name}: only {len(pool)} market candidates, skipping the top-{BOARD_SIZE} board")


def band_order(pool, first, last, rule):
    band = pool[first - 1:last]
    if rule == "market":
        return band
    ranked = sorted((row for row in band if rule in row["scores"]),
                    key=lambda row: (-row["scores"][rule], row["market_rank"]))
    ranked = iter(ranked)
    return [next(ranked) if rule in row["scores"] else row for row in band]


def target_margin(bands):
    margins = [row["fppg_above_adp"] for band in bands for row in band[:TARGETS_PER_BAND]
               if row["fppg_above_adp"] is not None]
    return mean(margins) if margins else None


def best_rule(development, rules, band_indices):
    gains = {}
    for rule in rules:
        bands = (draft[rule][index] for draft in development for index in band_indices)
        margin = target_margin(bands)
        gains[rule] = margin if margin is not None else float("-inf")
    return max(gains, key=gains.get)


def board_metrics(draft, portfolio, group, placements, rule):
    selected = [row for _, row, _ in placements]
    ranked = [(rank, row) for rank, row, _ in placements if row["actual_rank"] is not None]
    targets = [row for _, row, target in placements if target]
    margins = [row["fppg_above_adp"] for row in targets if row["fppg_above_adp"] is not None]
    errors = [abs(rank - row["actual_rank"]) for rank, row in ranked]
    entrants = [row for row in selected if row["market_rank"] > BOARD_SIZE]
    return {
        "draft_season": draft,
        "board": portfolio,
        "band": group,
        "band_rule": rule,
        "board_players": len(selected),
        "players_with_outcomes": len(ranked),
        "actual_top_150_captured": sum(row["actual_rank"] <= BOARD_SIZE for _, row in ranked),
        "board_mean_abs_rank_error": round(mean(errors), 2) if errors else None,
        "entrants_from_outside_market_top_150": len(entrants),
        "entrants_finishing_top_150": sum(row["actual_rank"] is not None and row["actual_rank"] <= BOARD_SIZE
                                        for row in entrants),
        "band_target_picks": len(targets),
        "targets_with_outcomes": len(margins),
        "targets_beating_adp": sum(row["beat_adp"] is True for row in targets),
        "target_avg_fppg_above_adp": round(mean(margins), 3) if margins else None,
    }


def record_board(draft, portfolio, bands, rules, summaries, rankings):
    board, records = [], []
    for (first, last), band, rule in zip(BANDS, bands, rules):
        label = f"{first}–{last}"
        placements = [(rank, row, rank - first < TARGETS_PER_BAND)
                      for rank, row in enumerate(band, start=first) if rank <= BOARD_SIZE]
        board.extend(placements)
        records.append((label, rule, placements))
        for rank, row, target in placements:
            rankings.append({
                "draft_season": draft,
                "board": portfolio,
                "board_rank": rank,
                "band": label,
                "band_rule": rule,
                "market_fallback": rule != "market" and rule not in row["scores"],
                "band_target": target,
                **{key: value for key, value in row.items() if key not in ("draft_season", "scores")},
            })
    summaries.append(board_metrics(draft, portfolio, "Overall", board, portfolio_rule(rules)))
    summaries.extend(board_metrics(draft, portfolio, label, placements, rule) for label, rule, placements in records)
    return board


def portfolio_rule(rules):
    return rules[0] if len(set(rules)) == 1 else "varies_by_band"


def run_backtest(data, holdout_seasons):
    drafts = dict(candidate_pools(data))
    seasons = sorted(drafts)
    if len(seasons) <= holdout_seasons:
        raise ValueError("need a development draft and the requested holdouts with forecasts and at least 150 ADP players")
    development, holdout = seasons[:-holdout_seasons], seasons[-holdout_seasons:]
    scored = [row for pool in drafts.values() for row in pool if "projected_fppg" in row["scores"]]
    rules = BOX_SCORE_RULES + tuple(rule for rule in SAVED_RULES if scored and all(rule in row["scores"] for row in scored))
    if not scored:
        raise ValueError("no market players matched to prior-season forecasts")

    orders = {year: {rule: [band_order(pool, first, last, rule) for first, last in BANDS] for rule in rules}
              for year, pool in drafts.items()}
    training = [orders[year] for year in development]
    uniform = best_rule(training, rules, range(len(BANDS)))
    per_band = [best_rule(training, rules, (i,)) for i in range(len(BANDS))]
    fixed_rules = {rule: [rule] * len(BANDS) for rule in rules}
    fixed_rules["uniform_learned"] = [uniform] * len(BANDS)
    fixed_rules["band_specific"] = per_band

    summaries, rankings = [], []
    overall = defaultdict(list)
    for season in holdout:
        season_orders = orders[season]
        for portfolio, chosen in fixed_rules.items():
            bands = [season_orders[rule][i] for i, rule in enumerate(chosen)]
            board = record_board(season, portfolio, bands, chosen, summaries, rankings)
            overall[portfolio].extend(board)

    pooled = [board_metrics("All holdouts", portfolio, "Overall", board, portfolio_rule(fixed_rules[portfolio]))
              for portfolio, board in overall.items()]
    write_csv(data / "backtest_draft_board_summary.csv", pooled + summaries)
    write_csv(data / "backtest_draft_board_rankings.csv", rankings)
    print(f"Development drafts: {', '.join(development)}")
    print(f"Holdout drafts: {', '.join(holdout)}")
    print(f"Uniform rule: {uniform}")
    print("Band rules: " + ", ".join(f"{first}–{last} {rule}" for (first, last), rule in zip(BANDS, per_band)))
    for rule in SAVED_RULES:
        if rule not in rules:
            print(f"Skipping {rule}: saved scores do not cover every forecasted market candidate in these drafts")
    for result in pooled:
        print(f"{result['board']}: {result['actual_top_150_captured']}/{BOARD_SIZE * len(holdout)} actual top-150 "
              f"players captured, {result['targets_beating_adp']}/{result['band_target_picks']} band targets beat ADP, "
              f"{result['target_avg_fppg_above_adp']} FPPG vs ADP for ranked targets")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"), help="folder with downloads and saved forecasts")
    parser.add_argument("--holdout-seasons", type=int, default=2, help="number of latest drafts evaluated (default: 2)")
    args = parser.parse_args()
    if args.holdout_seasons < 1:
        parser.error("use at least one holdout season")
    try:
        run_backtest(**vars(args))
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(1, f"Draft board backtest failed: {error}\n")


if __name__ == "__main__":
    main()
