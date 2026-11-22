"""Check whether players beat ADP the season after their NBA debut."""

import math
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from backtest_common import (
    draft_outcomes, format_value_metrics, load_player_profiles, read_csv, report_results,
    run_cli, season_pairs, value_metrics, write_csv,
)

FIRST_ARCHIVE_YEAR = 1994  # ESPN season files are named by ending year.


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


def hit_rate_row(breakdown, group, rows):
    hits = sum(row["beat_adp"] for row in rows)
    return {
        "breakdown": breakdown,
        "group": group,
        "players": len(rows),
        "beat_adp": hits,
        "beat_adp_pct": round(100 * hits / len(rows), 1) if rows else None,
        **value_metrics(rows),
    }


def report_rookie_breakdown(output, results):
    rookies = [row for row in results if row["second_year"]]
    by_role = defaultdict(list)
    by_position = defaultdict(list)
    by_height = defaultdict(list)
    for row in rookies:
        by_role[row["next_lineup_role"]].append(row)
        by_position[row["rookie_position"] or "Unknown"].append(row)
        by_height[row["height_inches"]].append(row)

    summary = [hit_rate_row("overall", "Second-year players", rookies)]
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
                "rookie_position": None,
                "height_inches": None,
            }
            for name, athlete_id, outcome in outcomes
            if athlete_id in prior_players
        ]
        second_year_count = sum(row["second_year"] for row in rows)
        if second_year_count:
            add_rookie_profiles(data, year + 1, rows, prior_players)
        print(f"{prior.name} to {following.name}: matched {len(outcomes)}/{total_picks} ADP entries to unique players, "
              f"{second_year_count} second-year players and {len(rows) - second_year_count} other returning players")
        results.extend(rows)

    report_results(data / "backtest_rookie_results.csv", results, "second_year",
                   "Second-year players", "Other returning players")
    report_rookie_breakdown(data / "backtest_rookie_summary.csv", results)


if __name__ == "__main__":
    run_cli(__doc__, run_backtest, allow_ages=False)
