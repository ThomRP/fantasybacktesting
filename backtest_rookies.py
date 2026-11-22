"""Check whether players beat ADP the season after their NBA debut."""

from pathlib import Path
from tempfile import TemporaryDirectory

from backtest_common import draft_outcomes, read_csv, report_results, run_cli, season_pairs

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
            }
            for name, athlete_id, outcome in outcomes
            if athlete_id in prior_players
        ]
        second_year_count = sum(row["second_year"] for row in rows)
        print(f"{prior.name} to {following.name}: matched {len(outcomes)}/{total_picks} ADP entries to unique players, "
              f"{second_year_count} second-year players and {len(rows) - second_year_count} other returning players")
        results.extend(rows)

    report_results(data / "backtest_rookie_results.csv", results, "second_year",
                   "Second-year players", "Other returning players")


if __name__ == "__main__":
    run_cli(__doc__, run_backtest, allow_ages=False)
