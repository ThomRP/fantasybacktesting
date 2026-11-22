"""Pre-draft workload features from the previous NBA regular season."""

import json
from collections import defaultdict
from dataclasses import dataclass

from backtest_common import fantasy_points, parse_minutes, read_csv, regular_season_game

OPPORTUNITY_FEATURES = (
    "prior_mpg",
    "prior_post_break_mpg_change",
    "prior_fpts_per_36",
    "prior_fga_per_36",
    "prior_post_break_start_pct_change",
)


@dataclass(frozen=True)
class Appearance:
    minutes: float | None
    fpts: int | None
    attempts: int | None
    started: float | None


def appearance(stats, starter):
    try:
        points = fantasy_points(stats)
    except (KeyError, TypeError, ValueError):
        points = None
    try:
        attempts = int(stats["field_goals_attempted"])
    except (KeyError, TypeError, ValueError):
        attempts = None
    if attempts is not None and attempts < 0:
        attempts = None
    start_status = {"true": 1.0, "false": 0.0}.get(str(starter).strip().lower())
    return Appearance(parse_minutes(stats["minutes"]), points, attempts, start_status)


def archived_appearances(path):
    """Read the alternate ESPN feed when a whole game's CSV lines are absent."""
    with path.open(encoding="utf-8") as source:
        game = json.load(source)
    for team in game["boxscore"].get("players", []):
        for group in team["statistics"]:
            for player in group["athletes"]:
                stats = dict(zip(group["keys"], player["stats"]))
                if player["didNotPlay"] or stats.get("minutes", "") in ("", "--"):
                    continue
                stats["field_goals_made"], _, stats["field_goals_attempted"] = stats.get(
                    "fieldGoalsMade-fieldGoalsAttempted", "",
                ).partition("-")
                stats["free_throws_made"], _, stats["free_throws_attempted"] = stats.get(
                    "freeThrowsMade-freeThrowsAttempted", "",
                ).partition("-")
                stats["three_point_field_goals_made"] = stats.get(
                    "threePointFieldGoalsMade-threePointFieldGoalsAttempted", "",
                ).partition("-")[0]
                yield str(player["athlete"]["id"]), appearance(stats, player.get("starter"))


def player_appearances(folder, game_dates):
    covered_games = set()
    for row in read_csv(folder / "player_box_scores.csv"):
        game_id = row["game_id"]
        if game_id not in game_dates or row["did_not_play"].lower() == "true" or row["minutes"] in ("", "--"):
            continue
        covered_games.add(game_id)
        yield game_id, row["athlete_id"], appearance(row, row.get("starter"))

    for game_id in sorted(game_dates.keys() - covered_games):
        path = folder / "game_json" / f"{game_id}.json"
        if path.is_file():
            for athlete_id, line in archived_appearances(path):
                yield game_id, athlete_id, line


def observed_sum(values):
    return sum(values) if values and all(value is not None for value in values) else None


def rate_change(before, after, attribute):
    earlier = observed_sum([getattr(line, attribute) for line in before])
    later = observed_sum([getattr(line, attribute) for line in after])
    if earlier is None or later is None:
        return None
    return later / len(after) - earlier / len(before)


def per_36(minutes, values):
    total = observed_sum(values)
    if not minutes or total is None:
        return None
    return 36 * total / minutes


def load_opportunity_features(prior_folder, *, all_star_date):
    """Return workload features by ESPN ID using available prior-season appearances.

    Games absent from both local feeds contribute no observations. For observed
    appearances, a rate is missing if any required value is unavailable. Changes
    require appearances before and after the break and are post minus pre.
    """
    game_dates = {
        game["game_id"]: game["game_date"] for game in read_csv(prior_folder / "games.csv")
        if regular_season_game(game)
    }
    halves = defaultdict(lambda: ([], []))
    cutoff = str(all_star_date) if all_star_date is not None else None
    for game_id, athlete_id, line in player_appearances(prior_folder, game_dates):
        before, after = halves[athlete_id]
        (after if cutoff is not None and game_dates[game_id] > cutoff else before).append(line)

    features = {}
    for athlete_id, (before, after) in halves.items():
        lines = before + after
        minutes = observed_sum([line.minutes for line in lines])
        start_change = rate_change(before, after, "started")
        features[athlete_id] = {
            "prior_mpg": minutes / len(lines) if minutes is not None else None,
            "prior_post_break_mpg_change": rate_change(before, after, "minutes"),
            "prior_fpts_per_36": per_36(minutes, [line.fpts for line in lines]),
            "prior_fga_per_36": per_36(minutes, [line.attempts for line in lines]),
            "prior_post_break_start_pct_change": 100 * start_change if start_change is not None else None,
        }
    return features
