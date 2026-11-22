"""Shared scoring, season comparisons, and reporting for the ADP backtests."""

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from statistics import mean, median
from tempfile import TemporaryDirectory

from adp_names import clean_adp_player_name

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
ADP_COLUMNS = ("avg", "average", "adp", "averagedraftposition", "consensusadp", "consensus")
NBA_TEAM_IDS = {str(team) for team in range(1, 31)}


@dataclass(frozen=True)
class Backtest:
    column: str
    label: str
    output_stem: str
    matches: Callable[[float], bool]


@dataclass
class PlayerSeason:
    names: set[str] = field(default_factory=set)
    pre_games: int = 0
    post_games: int = 0
    pre_points: int = 0
    post_points: int = 0
    positions: Counter[str] = field(default_factory=Counter)
    starts: int = 0
    games_without_start_status: int = 0

    @property
    def total_points(self):
        return self.pre_points + self.post_points

    @property
    def total_games(self):
        return self.pre_games + self.post_games


def fantasy_points(stats):
    return sum(int(stats[column]) * weight for column, weight in SCORING.items())


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        yield from csv.DictReader(source)


def regular_season_game(game):
    """Select completed games that count in the regular-season standings."""
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
        team_id = str((team.get("team") or {}).get("id", ""))
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
                position = (athlete.get("position") or {}).get("abbreviation")
                yield str(athlete["id"]), athlete["displayName"], fantasy_points(stats), position, player.get("starter"), team_id


def player_lines(folder, game_dates):
    covered_games = set()
    for row in read_csv(folder / "player_box_scores.csv"):
        game_id = row["game_id"]
        if game_id not in game_dates or row["did_not_play"].lower() == "true" or row["minutes"] in ("", "--"):
            continue
        covered_games.add(game_id)
        yield (
            game_id, row["athlete_id"], row["athlete_display_name"], fantasy_points(row),
            row.get("athlete_position_abbreviation"), row.get("starter"), row.get("team_id"),
        )

    for game_id in sorted(game_dates.keys() - covered_games):
        path = folder / "game_json" / f"{game_id}.json"
        if path.is_file():
            for athlete_id, name, points, position, starter, team_id in archived_lines(path):
                covered_games.add(game_id)
                yield game_id, athlete_id, name, points, position, starter, team_id

    missing = game_dates.keys() - covered_games
    if missing:
        print(f"{folder.name}: missing player box scores for regular-season games {', '.join(sorted(missing))}")


def read_season(folder, *, split_at_all_star=True):
    schedule = list(read_csv(folder / "games.csv"))
    if not any(g["season_type"] == "3" and g["status_type_completed"].lower() == "true" for g in schedule):
        print(f"Skipping {folder.name}: the schedule has no completed playoff game to confirm the regular season ended")
        return None
    all_star_date = None
    if split_at_all_star:
        all_star_date = max(
            (g["game_date"] for g in schedule if g["notes_headline"].lower().startswith("nba all-star")),
            default=None,
        )
        if all_star_date is None:
            raise ValueError(f"{folder.name}: no All-Star date found in games.csv")

    game_dates = {g["game_id"]: g["game_date"] for g in schedule if regular_season_game(g)}
    players = defaultdict(PlayerSeason)
    for game_id, athlete_id, name, points, position, starter, _ in player_lines(folder, game_dates):
        player = players[athlete_id]
        player.names.add(name)
        if position:
            player.positions[position] += 1
        start_status = str(starter).strip().lower()
        if start_status == "true":
            player.starts += 1
        elif start_status != "false":
            player.games_without_start_status += 1
        if all_star_date is not None and game_dates[game_id] > all_star_date:
            player.post_games += 1
            player.post_points += points
        else:
            player.pre_games += 1
            player.pre_points += points
    if not players:
        raise ValueError(f"{folder.name}: no usable regular-season player box scores")
    return (date.fromisoformat(all_star_date) if all_star_date else None), players


def read_player_profiles(path):
    profiles = {row["athlete_id"]: row for row in read_csv(path)}
    if not profiles:
        raise ValueError(f"{path}: no player profiles found")
    return profiles


def load_player_profiles(data, ending_year):
    cache = data / "player_core"
    path = cache / f"player_core_{ending_year}.csv"
    if path.is_file():
        return read_player_profiles(path)

    from download import ESPN_ARCHIVE, download_file

    url = f"{ESPN_ARCHIVE}/espn_nba_player_core/{path.name}"
    cache.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=cache) as temp:
        staged = Path(temp) / path.name
        download_file(url, staged)
        profiles = read_player_profiles(staged)
        staged.replace(path)
    return profiles


def load_birth_dates(data, ending_year):
    dates = {
        athlete_id: date.fromisoformat(profile["date_of_birth"][:10])
        for athlete_id, profile in load_player_profiles(data, ending_year).items()
        if profile["date_of_birth"]
    }
    if not dates:
        raise ValueError(f"No player birth dates found for the season ending {ending_year}")
    return dates


def filter_by_age(rows, birth_dates, all_star_date, ages):
    youngest, oldest = ages
    eligible = []
    missing_birth_dates = 0
    for row in rows:
        birth = birth_dates.get(row["athlete_id"])
        if birth is None:
            missing_birth_dates += 1
            continue
        age = all_star_date.year - birth.year
        if (all_star_date.month, all_star_date.day) < (birth.month, birth.day):
            age -= 1
        if youngest <= age <= oldest:
            eligible.append({**row, "age_at_break": age})

    print(f"  Ages {youngest}-{oldest}: kept {len(eligible)}/{len(rows)} eligible rows, "
          f"{missing_birth_dates} missing birth dates")
    return eligible


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
            name = clean_adp_player_name(row[columns["playername"]] or "")
            try:
                adp = float(row[average])
            except (TypeError, ValueError):
                continue
            if name and math.isfinite(adp) and adp > 0:
                yield name, adp


def adp_slot_points(totals, adp):
    """Interpolate points between adjacent leaderboard spots for a fractional ADP."""
    index = max(adp - 1, 0)
    if index >= len(totals):
        return 0.0
    place = int(index)
    earlier = max(totals[place], 0)
    later = max(totals[place + 1], 0) if place + 1 < len(totals) else 0
    return earlier + (later - earlier) * (index - place)


def lineup_details(player):
    if player is None or not player.total_games:
        return {"next_starts": 0, "next_start_pct": None, "next_lineup_role": "No games"}
    if player.games_without_start_status:
        return {"next_starts": None, "next_start_pct": None, "next_lineup_role": "Unknown"}

    role = "Regular starter" if player.starts * 2 >= player.total_games else "Started fewer than half"
    return {
        "next_starts": player.starts,
        "next_start_pct": round(100 * player.starts / player.total_games, 1),
        "next_lineup_role": role,
    }


def draft_outcomes(following, prior_players, following_players, *, match_prior_only=False, rank_per_game=False):
    # Match the draft to ESPN by name, then track the athlete by ESPN ID.
    athlete_ids = defaultdict(set)
    player_sets = (prior_players,) if match_prior_only else (prior_players, following_players)
    for players in player_sets:
        for athlete_id, player in players.items():
            for name in player.names:
                athlete_ids[name_key(name)].add(athlete_id)

    # Rank the entire NBA player pool before rounding any scores. Ties share a rank.
    if rank_per_game:
        scores = {athlete_id: player.total_points / player.total_games
                  for athlete_id, player in following_players.items() if player.total_games}
    else:
        scores = {athlete_id: player.total_points for athlete_id, player in following_players.items()}
    leaderboard = sorted(scores.values(), reverse=True)
    rank_by_score = {}
    for place, score in enumerate(leaderboard, start=1):
        rank_by_score.setdefault(score, place)

    picks = list(read_adp(following / "adp.csv"))
    if not picks:
        raise ValueError(f"{following.name}: no numeric average ADP values")
    matched = set()
    outcomes = []
    for name, adp in picks:
        ids = athlete_ids[name_key(name)]
        if len(ids) != 1:
            continue
        (athlete_id,) = ids
        if athlete_id in matched:
            continue
        matched.add(athlete_id)

        after = following_players.get(athlete_id)
        actual_points = after.total_points if after else 0
        score = scores.get(athlete_id)
        pick_score = adp_slot_points(leaderboard, adp)
        rank = rank_by_score.get(score)
        outcome = {
            "adp": adp,
            "next_games": after.total_games if after else 0,
            "next_total_fpts": actual_points,
            "next_rank": rank,
            "adp_slots_gained": round(adp - rank, 3) if rank is not None else None,
            "beat_adp": rank < adp if rank is not None else (None if rank_per_game else False),
        }
        if rank_per_game:
            outcome.update(
                next_fppg=round(score, 3) if score is not None else None,
                adp_slot_fppg=round(pick_score, 3),
                fppg_above_adp=round(score - pick_score, 3) if score is not None else None,
            )
        else:
            outcome.update(adp_slot_fpts=round(pick_score, 3), fpts_above_adp=round(actual_points - pick_score, 3))
        outcome.update(lineup_details(after))
        outcomes.append((name, athlete_id, outcome))
    return outcomes, len(picks)


def compare_seasons(prior, following, prior_players, following_players, backtest):
    outcomes, total_picks = draft_outcomes(following, prior_players, following_players)
    results = []
    for name, athlete_id, outcome in outcomes:
        player = prior_players.get(athlete_id)
        if player is None or not player.pre_games or not player.post_games or player.pre_points <= 0:
            continue
        pre_fppg = player.pre_points / player.pre_games
        post_fppg = player.post_points / player.post_games
        # Calculate the change before rounding the values for the CSV.
        baseline = player.pre_points * player.post_games
        growth = (player.post_points * player.pre_games - baseline) / baseline

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
            backtest.column: backtest.matches(growth),
            **outcome,
        })

    print(f"{prior.name} to {following.name}: matched {len(outcomes)}/{total_picks} ADP entries to unique players, "
          f"{len(results)} with eligible splits")
    return results


def season_pairs(data, *, split_at_all_star=True):
    seasons = {int(p.name[:4]): p for p in data.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}", p.name)}
    stats = {}
    for year, prior in sorted(seasons.items()):
        following = seasons.get(year + 1)
        if following is None:
            continue
        if not (following / "adp.csv").is_file():
            print(f"Skipping {prior.name} to {following.name}: missing {following.name}/adp.csv")
            continue
        for folder in (prior, following):
            if folder not in stats:
                stats[folder] = read_season(folder, split_at_all_star=split_at_all_star)
        if stats[prior] is None or stats[following] is None:
            continue
        yield year, prior, following, stats[prior], stats[following]


def run_backtest(data, backtest, ages=None):
    results = []
    for year, prior, following, prior_stats, following_stats in season_pairs(data):
        all_star_date, prior_players = prior_stats
        _, following_players = following_stats
        season_results = compare_seasons(prior, following, prior_players, following_players, backtest)
        if ages is not None and season_results:
            birth_dates = load_birth_dates(data, year + 1)
            season_results = filter_by_age(season_results, birth_dates, all_star_date, ages)
        results.extend(season_results)

    suffix = f"_ages_{ages[0]}_{ages[1]}" if ages is not None else ""
    output = data / f"{backtest.output_stem}{suffix}.csv"
    report_results(output, results, backtest.column, backtest.label)


def write_csv(output, rows, *, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(rows[0])
    with output.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def adp_hit_counts(rows):
    scored = [row["beat_adp"] for row in rows if row["beat_adp"] is not None]
    return sum(scored), len(scored)


def value_metrics(rows, *, per_game=False):
    unit = "fppg" if per_game else "fpts"
    margin_key = f"{unit}_above_adp"
    slots = [row["adp_slots_gained"] for row in rows if row["adp_slots_gained"] is not None]
    scored = [row for row in rows if row[margin_key] is not None]
    margins = sorted(row[margin_key] for row in scored)
    trim_per_tail = len(margins) // 10
    trimmed = margins[trim_per_tail:len(margins) - trim_per_tail]
    low_games = [row[margin_key] for row in scored if row["next_games"] <= 10]
    other_games = [row[margin_key] for row in scored if row["next_games"] > 10]
    return {
        "ranked_players": len(slots),
        "avg_adp_slots_gained_ranked": round(mean(slots), 1) if slots else None,
        f"avg_{margin_key}": round(mean(margins), 1) if margins else None,
        "median_adp_slots_gained_ranked": round(median(slots), 1) if slots else None,
        f"median_{margin_key}": round(median(margins), 1) if margins else None,
        f"trimmed_avg_{margin_key}": round(mean(trimmed), 1) if trimmed else None,
        "trimmed_players_per_tail": trim_per_tail,
        "players_10_games_or_fewer": len(low_games),
        f"{unit}_contribution_10_games_or_fewer": round(sum(low_games) / len(scored), 1) if scored else None,
        f"avg_{margin_key}_over_10_games": round(mean(other_games), 1) if other_games else None,
    }


def format_value_metrics(metrics, *, per_game=False):
    unit = "fppg" if per_game else "fpts"
    margin_key = f"{unit}_above_adp"
    result = f"{metrics[f'avg_{margin_key}']:+.1f} avg {unit.upper()} vs ADP, {metrics[f'median_{margin_key}']:+.1f} median"
    if metrics["trimmed_players_per_tail"]:
        result += f", {metrics[f'trimmed_avg_{margin_key}']:+.1f} trimmed avg"
    if metrics["players_10_games_or_fewer"]:
        count = metrics["players_10_games_or_fewer"]
        contribution = metrics[f"{unit}_contribution_10_games_or_fewer"]
        minimum = 1 if per_game else 0
        result += f", {minimum}-10 games: {count} ({contribution:+.1f} {unit.upper()} per pick)"
    if metrics["ranked_players"]:
        result += f", {metrics['avg_adp_slots_gained_ranked']:+.1f} avg slots ({metrics['ranked_players']} ranked)"
    return result


def report_results(output, results, cohort_column, label, other_label="Other eligible players", *, per_game=False):
    if not results:
        raise ValueError("no eligible player-seasons found. Use consecutive downloaded seasons with ADP in the later season")

    write_csv(output, results)
    for group_label, matched in ((label, True), (other_label, False)):
        group = [row for row in results if row[cohort_column] == matched]
        hits, count = adp_hit_counts(group)
        if count:
            unranked = f", {len(group) - count} unranked" if count < len(group) else ""
            print(f"{group_label}: {hits}/{count} beat ADP ({hits / count:.1%}){unranked}, "
                  f"{format_value_metrics(value_metrics(group, per_game=per_game), per_game=per_game)}")
        elif group:
            print(f"{group_label}: {len(group)} unranked player-seasons")
        else:
            print(f"{group_label}: no eligible player-seasons")
    print(f"Saved {len(results)} player-seasons to {output}")


def run_cli(description, run, *, allow_ages=True):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", type=Path, default=Path("data"), help="download folder (default: data)")
    if allow_ages:
        parser.add_argument("--ages", nargs=2, type=int, metavar=("MIN", "MAX"),
                            help="inclusive age range at the signal season's All-Star break")
    args = parser.parse_args()
    ages = getattr(args, "ages", None)
    if ages is not None and not 0 <= ages[0] <= ages[1]:
        parser.error("--ages requires a nonnegative minimum and a maximum at least as large")
    try:
        if allow_ages:
            run(args.data, ages)
        else:
            run(args.data)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(1, f"Backtest failed: {error}\n")
