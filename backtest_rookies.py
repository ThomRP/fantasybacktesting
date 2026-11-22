"""Check whether players beat ADP the season after their NBA debut."""

import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

from backtest_common import (
    draft_outcomes, format_value_metrics, load_player_profiles, name_key, player_lines, read_csv,
    regular_season_game, report_results, run_cli, season_pairs, value_metrics, write_csv,
)
from backtest_rookie_plots import write_starter_plots

FIRST_ARCHIVE_YEAR = 1994  # ESPN season files are named by ending year.
OPENER_ROLES = ("Started team opener", "Did not start team opener", "Unknown")


def players_with_appearances(data, ending_year):
    cache = data / "player_history"
    path = cache / f"players_{ending_year}.txt"
    if path.is_file():
        players = set(path.read_text(encoding="utf-8").split())
        if players:
            return players

    from download import ESPN_ARCHIVE, download_file

    filename = f"player_season_stats_{ending_year}.csv"
    url = f"{ESPN_ARCHIVE}/espn_nba_player_season_stats/{filename}"
    cache.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=cache) as temp:
        stats = Path(temp) / filename
        download_file(url, stats)
        players = {
            row["athlete_id"] for row in read_csv(stats)
            if row["stat_name"] == "gamesPlayed" and row["season"] == str(ending_year)
            and float(row["value"] or 0) > 0
        }
        if not players:
            raise ValueError(f"No NBA regular-season appearances in {filename}")
        staged = stats.with_suffix(".txt")
        staged.write_text("\n".join(sorted(players)) + "\n", encoding="utf-8")
        staged.replace(path)
    return players


def listed_height(profile):
    try:
        inches = float(profile.get("height") or 0)
    except ValueError:
        return None
    return inches if math.isfinite(inches) and inches > 0 else None


def add_rookie_profiles(data, ending_year, rows, prior_players):
    try:
        profiles = load_player_profiles(data, ending_year)
    except (OSError, ValueError, KeyError) as error:
        print(f"Could not load ESPN player profiles for the season ending {ending_year}: {error}")
        profiles = {}

    for row in rows:
        if not row["second_year"]:
            continue
        athlete_id = row["athlete_id"]
        profile = profiles.get(athlete_id, {})
        positions = prior_players[athlete_id].positions.most_common(1)
        row["rookie_position"] = positions[0][0] if positions else profile.get("position_abbreviation")
        row["height_inches"] = listed_height(profile)


def read_opening_lineups(folder):
    games = sorted(
        (row for row in read_csv(folder / "games.csv") if regular_season_game(row)),
        key=lambda row: (row["game_date"], row["game_id"]),
    )
    openers = {}
    for game in games:
        for team_id in (game["home_id"], game["away_id"]):
            openers.setdefault(team_id, game)

    game_dates = {game["game_id"]: game["game_date"] for game in openers.values()}
    lineups = defaultdict(lambda: defaultdict(set))
    for game_id, athlete_id, name, _, _, starter, team_id in player_lines(folder, game_dates):
        opener = openers.get(team_id)
        if opener and game_id == opener["game_id"] and str(starter).strip().lower() == "true":
            lineups[team_id][name_key(name)].add(athlete_id)

    verified_teams = 0
    starters = {}
    for team_id, game in openers.items():
        if len(lineups[team_id]) != 5:
            continue
        verified_teams += 1
        for ids in lineups[team_id].values():
            for athlete_id in ids:
                starters.setdefault(athlete_id, (team_id, game["game_id"], game["game_date"]))
    complete = bool(openers) and verified_teams == len(openers)
    if not complete:
        print(f"{folder.name}: verified {verified_teams}/{len(openers)} team opening lineups. Unconfirmed roles are unknown.")
    return starters, complete


def opener_details(athlete_id, starters, complete):
    if athlete_id in starters:
        team_id, game_id, game_date = starters[athlete_id]
        role = "Started team opener"
    else:
        team_id = game_id = game_date = None
        role = "Did not start team opener" if complete else "Unknown"
    return {
        "next_opener_role": role,
        "next_opener_team_id": team_id,
        "next_opener_game_id": game_id,
        "next_opener_date": game_date,
    }


def is_opener_majority_starter(row):
    starts = row["next_starts"]
    return (
        row["second_year"]
        and row["next_opener_role"] == "Started team opener"
        and starts is not None
        and 2 * starts > row["next_games"]
    )


def hit_rate_row(breakdown, group, rows):
    hits = sum(row["beat_adp"] for row in rows)
    return {
        "breakdown": breakdown,
        "group": group,
        "players": len(rows),
        "beat_adp": hits,
        "beat_adp_pct": round(100 * hits / len(rows), 1) if rows else None,
        "avg_adp": round(mean(row["adp"] for row in rows), 1) if rows else None,
        **value_metrics(rows),
    }


def report_rookie_breakdown(output, results):
    rookies = [row for row in results if row["second_year"]]
    by_role = defaultdict(list)
    by_opener = defaultdict(list)
    regular_starters_by_opener = defaultdict(list)
    by_position = defaultdict(list)
    by_height = defaultdict(list)
    for row in rookies:
        by_role[row["next_lineup_role"]].append(row)
        opener_role = row["next_opener_role"]
        by_opener[opener_role].append(row)
        if row["next_lineup_role"] == "Regular starter":
            regular_starters_by_opener[opener_role].append(row)
        by_position[row["rookie_position"] or "Unknown"].append(row)
        by_height[row["height_inches"]].append(row)

    majority_starters = [row for row in rookies if row["opener_majority_starter"]]
    summary = [
        hit_rate_row("overall", "Second-year players", rookies),
        hit_rate_row("opener_majority", "Started opener and majority of games", majority_starters),
    ]
    for breakdown, groups in (("opening_lineup", by_opener),
                              ("opening_lineup_among_regular_starters", regular_starters_by_opener)):
        summary.extend(hit_rate_row(breakdown, role, groups[role]) for role in OPENER_ROLES)
    for role in ("Regular starter", "Started fewer than half", "No games", "Unknown"):
        if by_role[role] or role == "Regular starter":
            summary.append(hit_rate_row("lineup_role", role, by_role[role]))
    for position in sorted(by_position):
        summary.append(hit_rate_row("position", position, by_position[position]))
    for inches in sorted(by_height, key=lambda height: (height is None, height)):
        height = "Unknown"
        if inches is not None:
            feet, remainder = divmod(inches, 12)
            height = f"{feet:g} ft {remainder:g} in"
        summary.append(hit_rate_row("height", height, by_height[inches]))

    write_csv(output, summary)
    for row in summary[1:]:
        label = f"{row['breakdown'].replace('_', ' ').title()} {row['group']}"
        if row["players"]:
            print(f"{label}: {row['beat_adp']}/{row['players']} beat ADP ({row['beat_adp_pct']:.1f}%), "
                  f"{format_value_metrics(row)}")
        else:
            print(f"{label}: no eligible player-seasons")
    print(f"Saved second-year breakdown to {output}")


def run_backtest(data):
    played_before = set()
    archive_through = FIRST_ARCHIVE_YEAR - 1
    results = []
    for year, prior, following, prior_stats, following_stats in season_pairs(data, split_at_all_star=False):
        if year < FIRST_ARCHIVE_YEAR:
            print(f"Skipping {prior.name}: ESPN appearance history starts in 1993-94")
            continue
        if archive_through < FIRST_ARCHIVE_YEAR:
            print("Checking ESPN appearance history back to 1993-94", flush=True)
        # The year the prior season starts is the ending year of the season before it.
        for ending_year in range(archive_through + 1, year + 1):
            played_before.update(players_with_appearances(data, ending_year))
        archive_through = year

        _, prior_players = prior_stats
        _, following_players = following_stats
        opener_starters, opener_lineups_complete = read_opening_lineups(following)
        rookies = prior_players.keys() - played_before
        played_before.update(prior_players)
        outcomes, total_picks = draft_outcomes(following, prior_players, following_players)
        rows = [
            {
                "prior_season": prior.name,
                "draft_season": following.name,
                "player": name,
                "athlete_id": athlete_id,
                "prior_games": prior_players[athlete_id].total_games,
                "second_year": athlete_id in rookies,
                **outcome,
                **opener_details(athlete_id, opener_starters, opener_lineups_complete),
                "rookie_position": None,
                "height_inches": None,
            }
            for name, athlete_id, outcome in outcomes
            if athlete_id in prior_players
        ]
        for row in rows:
            row["opener_majority_starter"] = is_opener_majority_starter(row)
        second_year_count = sum(row["second_year"] for row in rows)
        if second_year_count:
            add_rookie_profiles(data, year + 1, rows, prior_players)
        print(f"{prior.name} to {following.name}: matched {len(outcomes)}/{total_picks} ADP entries to unique players, "
              f"{second_year_count} second-year players and {len(rows) - second_year_count} other returning players")
        results.extend(rows)

    report_results(data / "backtest_rookie_results.csv", results, "second_year",
                   "Second-year players", "Other returning players")
    report_rookie_breakdown(data / "backtest_rookie_summary.csv", results)
    cohort = [row for row in results if row["opener_majority_starter"]]
    cohort.sort(key=lambda row: (-row["adp_slots_gained"], row["player"].casefold(), row["draft_season"]))
    cohort_csv = data / "backtest_rookie_opener_majority.csv"
    plots = data / "backtest_rookie_opener_majority.pdf"
    write_csv(cohort_csv, cohort, fieldnames=list(results[0]))
    write_starter_plots(plots, cohort)
    print(f"Saved {len(cohort)} qualifying players to {cohort_csv} and {plots}")


if __name__ == "__main__":
    run_cli(__doc__, run_backtest, allow_ages=False)
