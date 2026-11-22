"""Test whether players who surge after the All-Star break beat next season's ADP."""

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SCORING = {
    "field_goals_made": 2,
    "field_goals_attempted": -1,
    "free_throws_made": 1,
    "free_throws_attempted": -1,
    "three_point_field_goals_made": 1,
    "rebounds": 1,
    "assists": 2,
    "steals": 4,
    "blocks": 4,
    "turnovers": -2,
    "points": 1,
}
SURGE_THRESHOLD = 0.10
ADP_COLUMNS = ("avg", "average", "adp", "averagedraftposition", "consensusadp", "consensus")
NBA_TEAM_IDS = {str(team) for team in range(1, 31)}


@dataclass
class PlayerSeason:
    names: set[str] = field(default_factory=set)
    pre_games: int = 0
    post_games: int = 0
    pre_points: int = 0
    post_points: int = 0

    @property
    def total_points(self):
        return self.pre_points + self.post_points


def fantasy_points(stats):
    return sum(int(stats[column]) * weight for column, weight in SCORING.items())


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        yield from csv.DictReader(source)


def regular_season_game(game):
    headline = game["notes_headline"].lower()
    cup_final = "championship" in headline and (
        "nba cup" in headline or "in-season tournament" in headline
    )
    return (
        game["season_type"] == "2"
        and game["status_type_completed"].lower() == "true"
        and game["home_id"] in NBA_TEAM_IDS
        and game["away_id"] in NBA_TEAM_IDS
        and not cup_final
    )


def archived_lines(path):
    """Read player lines from original ESPN JSON saved for a missing CSV game."""
    with path.open(encoding="utf-8") as source:
        game = json.load(source)
    for team in game["boxscore"].get("players", []):
        for group in team["statistics"]:
            for player in group["athletes"]:
                stats = dict(zip(group["keys"], player["stats"]))
                if player["didNotPlay"] or stats.get("minutes", "") in ("", "--"):
                    continue
                stats["field_goals_made"], stats["field_goals_attempted"] = stats[
                    "fieldGoalsMade-fieldGoalsAttempted"
                ].split("-")
                stats["free_throws_made"], stats["free_throws_attempted"] = stats[
                    "freeThrowsMade-freeThrowsAttempted"
                ].split("-")
                threes = stats["threePointFieldGoalsMade-threePointFieldGoalsAttempted"]
                stats["three_point_field_goals_made"] = threes.split("-")[0]
                athlete = player["athlete"]
                yield str(athlete["id"]), athlete["displayName"], fantasy_points(stats)


def player_lines(folder, game_dates):
    covered_games = set()
    for row in read_csv(folder / "player_box_scores.csv"):
        game_id = row["game_id"]
        if game_id not in game_dates or row["did_not_play"].lower() == "true" or row["minutes"] in ("", "--"):
            continue
        covered_games.add(game_id)
        yield game_id, row["athlete_id"], row["athlete_display_name"], fantasy_points(row)

    for game_id in sorted(game_dates.keys() - covered_games):
        path = folder / "game_json" / f"{game_id}.json"
        if path.is_file():
            for athlete_id, name, points in archived_lines(path):
                covered_games.add(game_id)
                yield game_id, athlete_id, name, points

    missing = game_dates.keys() - covered_games
    if missing:
        print(f"{folder.name}: missing player box scores for regular-season games {', '.join(sorted(missing))}")


def read_season(folder):
    schedule = list(read_csv(folder / "games.csv"))
    if not any(g["season_type"] == "3" and g["status_type_completed"].lower() == "true" for g in schedule):
        print(f"Skipping {folder.name}: the schedule has no completed playoff game to confirm the regular season ended")
        return None
    all_star_date = max(
        (g["game_date"] for g in schedule if g["notes_headline"].lower().startswith("nba all-star")),
        default=None,
    )
    if all_star_date is None:
        raise ValueError(f"{folder.name}: no All-Star date found in games.csv")

    game_dates = {g["game_id"]: g["game_date"] for g in schedule if regular_season_game(g)}
    players = defaultdict(PlayerSeason)
    for game_id, athlete_id, name, points in player_lines(folder, game_dates):
        player = players[athlete_id]
        player.names.add(name)
        if game_dates[game_id] > all_star_date:
            player.post_games += 1
            player.post_points += points
        else:
            player.pre_games += 1
            player.pre_points += points
    if not players:
        raise ValueError(f"{folder.name}: no usable regular-season player box scores")
    return players


def name_key(name):
    name = re.sub(r"\s*\([^()]*\)\s*$", "", name.casefold())
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r"\s+(?:jr\.?|sr\.?|ii|iii|iv)$", "", name.strip())
    return re.sub(r"[^a-z0-9]", "", name)


def read_adp(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = csv.DictReader(source)
        columns = {re.sub(r"[^a-z0-9]", "", label.lower()): label for label in rows.fieldnames or []}
        average = next((columns[key] for key in ADP_COLUMNS if key in columns), None)
        if average is None or "playername" not in columns:
            raise ValueError(f"{path}: expected player_name and an average ADP column")
        for row in rows:
            name = row[columns["playername"]]
            try:
                adp = float(row[average])
            except (TypeError, ValueError):
                continue
            if name and math.isfinite(adp) and adp > 0:
                yield name, adp


def compare_seasons(prior, following, prior_players, following_players):
    # Match the draft to ESPN by name, then track the athlete by ESPN ID.
    athlete_ids = defaultdict(set)
    for players in (prior_players, following_players):
        for athlete_id, player in players.items():
            for name in player.names:
                athlete_ids[name_key(name)].add(athlete_id)

    # Rank the entire NBA player pool. Tied totals receive the same rank.
    totals = sorted((player.total_points for player in following_players.values()), reverse=True)
    rank_by_total = {}
    for place, total in enumerate(totals, start=1):
        rank_by_total.setdefault(total, place)

    picks = list(read_adp(following / "adp.csv"))
    if not picks:
        raise ValueError(f"{following.name}: no numeric average ADP values")
    matched = set()
    results = []
    for name, adp in picks:
        ids = athlete_ids[name_key(name)]
        if len(ids) != 1:
            continue
        (athlete_id,) = ids
        if athlete_id in matched:
            continue
        matched.add(athlete_id)

        player = prior_players.get(athlete_id)
        if player is None or not player.pre_games or not player.post_games or player.pre_points <= 0:
            continue
        pre_fppg = player.pre_points / player.pre_games
        post_fppg = player.post_points / player.post_games
        # Compute the change from integer totals so exactly +10% qualifies.
        baseline = player.pre_points * player.post_games
        growth = (player.post_points * player.pre_games - baseline) / baseline

        after = following_players.get(athlete_id)
        rank = rank_by_total[after.total_points] if after else None
        results.append({
            "signal_season": prior.name,
            "draft_season": following.name,
            "player": name,
            "athlete_id": athlete_id,
            "pre_games": player.pre_games,
            "post_games": player.post_games,
            "pre_fppg": round(pre_fppg, 3),
            "post_fppg": round(post_fppg, 3),
            "fppg_change_pct": round(100 * growth, 3),
            "surged": growth >= SURGE_THRESHOLD,
            "adp": adp,
            "next_games": after.pre_games + after.post_games if after else 0,
            "next_total_fpts": after.total_points if after else 0,
            "next_rank": rank,
            "adp_slots_gained": round(adp - rank, 3) if rank is not None else None,
            "beat_adp": rank is not None and rank < adp,
        })

    print(f"{prior.name} to {following.name}: matched {len(matched)}/{len(picks)} ADP entries to unique players, "
          f"{len(results)} with eligible splits")
    return results


def run_backtest(data):
    seasons = {int(p.name[:4]): p for p in data.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}", p.name)}
    stats = {}
    results = []
    for year, prior in sorted(seasons.items()):
        following = seasons.get(year + 1)
        if following is None:
            continue
        if not (following / "adp.csv").is_file():
            print(f"Skipping {prior.name} to {following.name}: missing {following.name}/adp.csv")
            continue
        for folder in (prior, following):
            if folder not in stats:
                stats[folder] = read_season(folder)
        if stats[prior] is None or stats[following] is None:
            continue
        results.extend(compare_seasons(prior, following, stats[prior], stats[following]))

    if not results:
        raise ValueError("no eligible player-seasons found. Use consecutive downloaded seasons with ADP in the later season")

    output = data / "backtest_results.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    for label, surge in (("Surged (10% or more)", True), ("Other eligible players", False)):
        group = [row for row in results if row["surged"] == surge]
        hits = sum(row["beat_adp"] for row in group)
        if group:
            print(f"{label}: {hits}/{len(group)} beat ADP ({hits / len(group):.1%})")
        else:
            print(f"{label}: no eligible player-seasons")
    print(f"Saved {len(results)} player-seasons to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"), help="download folder (default: data)")
    args = parser.parse_args()
    try:
        run_backtest(args.data)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(1, f"Backtest failed: {error}\n")


if __name__ == "__main__":
    main()
