"""Download FantasyPros NBA ADP and ESPN game data by season."""

import argparse
import csv
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlsplit
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from adp_names import clean_adp_player_name

FANTASYPROS = "https://www.fantasypros.com/nba/adp/overall.php"
ESPN_ARCHIVE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
ESPN_RAW = "https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-raw/main/nba/json/final"
GAME_FILES = (
    ("games", "espn_nba_schedules/nba_schedule"),
    ("player_box_scores", "espn_nba_player_boxscores/player_box"),
    ("team_box_scores", "espn_nba_team_boxscores/team_box"),
    ("play_by_play", "espn_nba_pbp/play_by_play"),
)


def open_url(url):
    return urlopen(Request(url, headers={"User-Agent": "fantasybacktesting/0.1"}), timeout=60)


def parse_adp(html, year):
    soup = BeautifulSoup(html, "html.parser")
    selected = soup.select_one('select[name="year"] option[selected], select#year option[selected]')
    if selected and selected.get("value", "").isdigit() and selected["value"] != str(year):
        raise ValueError(f"FantasyPros returned ADP for {selected['value']} instead of {year}")

    ranks = {"rank", "rk", "overallrank"}
    averages = {"avg", "average", "adp", "consensus", "consensusadp", "averagedraftposition"}
    for table in soup.find_all("table"):
        rows = iter(table.find_all("tr"))
        for row in rows:
            cells = row.find_all("th", recursive=False)
            headers = []
            for index, cell in enumerate(cells, start=1):
                logo = cell.find("img")
                label = cell.get_text(" ", strip=True) or (logo.get("alt", "") if logo else "")
                headers.append(label or f"column_{index}")
            keys = [re.sub(r"[^a-z0-9]", "", header.lower()) for header in headers]
            if "player" in keys and ranks.intersection(keys) and averages.intersection(keys):
                break
        else:
            continue

        player = keys.index("player")
        rank = next(i for i, key in enumerate(keys) if key in ranks)
        average = next(i for i, key in enumerate(keys) if key in averages)
        records = []
        for row in rows:
            cells = row.find_all(["th", "td"], recursive=False)
            values = [cell.get_text(" ", strip=True) for cell in cells]
            if len(values) != len(headers) or not values[rank].isdigit() or not values[player]:
                continue
            link = cells[player].select_one('a[href*="/nba/players/"]')
            name = clean_adp_player_name(link.get_text(" ", strip=True) if link else values[player])
            player_url = urljoin(FANTASYPROS, link["href"]) if link else ""
            records.append([year, name, player_url, *values])
        if any(re.fullmatch(r"\d+(?:\.\d+)?", record[average + 3]) for record in records):
            return ["adp_year", "player_name", "fantasypros_player_url", *headers], records

    raise ValueError(f"No populated ADP table found for FantasyPros year {year}")


def fetch_adp(year):
    with open_url(f"{FANTASYPROS}?year={year}") as response:
        if parse_qs(urlsplit(response.url).query).get("year") != [str(year)]:
            raise ValueError(f"FantasyPros redirected away from the requested ADP year {year}")
        return parse_adp(response.read(), year)


def download_file(url, path):
    with open_url(url) as response, path.open("wb") as output:
        shutil.copyfileobj(response, output)
        length = response.headers.get("Content-Length")
        if length and output.tell() != int(length):
            raise ValueError(f"Incomplete download from {url}")


def game_ids(path, end_year, completed_only=False):
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = csv.reader(source)
        headers = next(rows, [])
        required = {"game_id", "season"}
        if completed_only:
            required.add("status_type_completed")
        if not required.issubset(headers):
            raise ValueError(f"Missing expected columns in {path.name}")
        game = headers.index("game_id")
        season = headers.index("season")
        status = headers.index("status_type_completed") if completed_only else None
        ids = set()
        for row in rows:
            if not row:
                continue
            if len(row) != len(headers) or row[season] != str(end_year) or not row[game].isdigit():
                raise ValueError(f"Invalid row {rows.line_num} in {path.name} for season ending {end_year}")
            if status is None or row[status].lower() == "true":
                ids.add(row[game])
        if not ids:
            raise ValueError(f"No {'completed ' if completed_only else ''}games in {path.name}")
        return ids


def download_games(directory, end_year, play_by_play):
    coverage = {}
    for name, release in GAME_FILES if play_by_play else GAME_FILES[:-1]:
        path = directory / f"{name}.csv"
        download_file(f"{ESPN_ARCHIVE}/{release}_{end_year}.csv", path)
        coverage[name] = game_ids(path, end_year, completed_only=name == "games")

    completed = coverage.pop("games")
    gaps = {name: completed - ids for name, ids in coverage.items()}
    for name, ids in coverage.items():
        print(f"  {name}: {len(completed & ids):,}/{len(completed):,} completed games in CSV")

    missing = set().union(*gaps.values())
    uncovered = []
    if missing:
        raw = directory / "game_json"
        raw.mkdir()
        for game in sorted(missing):
            path = raw / f"{game}.json"
            try:
                download_file(f"{ESPN_RAW}/{game}.json", path)
                data = json.loads(path.read_bytes())
                if not isinstance(data, dict) or str(data.get("gameId")) != game:
                    raise ValueError("Archived JSON has an unexpected game ID")
            except (OSError, ValueError) as error:
                path.unlink(missing_ok=True)
                print(f"  Game {game}: {error}", file=sys.stderr)
                uncovered.append(game)
                continue
            box = data.get("boxscore") or {}
            raw_data = {
                "player_box_scores": box.get("players"),
                "team_box_scores": box.get("teams"),
                "play_by_play": data.get("plays"),
            }
            absent = [name for name, ids in gaps.items() if game in ids and not raw_data[name]]
            if absent:
                print(f"  Game {game}: missing {', '.join(absent)} in CSV and JSON", file=sys.stderr)
                uncovered.append(game)
        print(f"  Additional game JSON: {len(list(raw.iterdir())):,} files for CSV gaps")
    return uncovered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start_year", type=int, help="first NBA season's starting year, e.g. 2019 for 2019–20")
    parser.add_argument("last_year", type=int, nargs="?", help="last starting year to fetch, inclusive")
    parser.add_argument("--adp-year-is", choices=("start", "end"), help="which season year FantasyPros uses")
    parser.add_argument("--games-only", action="store_true", help="download the ESPN game data without ADP")
    parser.add_argument("--play-by-play", action="store_true", help="also download ESPN play-by-play CSVs")
    parser.add_argument("--output", type=Path, default=Path("data"), help="output directory (default: data)")
    args = parser.parse_args()

    if not args.games_only and args.adp_year_is is None:
        parser.error("specify --adp-year-is start or end to pair ADP with the right NBA season")
    last_year = args.start_year if args.last_year is None else args.last_year
    if not 1946 <= args.start_year <= last_year <= 9998:
        parser.error("provide NBA season starting years in ascending order")

    args.output.mkdir(parents=True, exist_ok=True)
    seen_adp = {}
    failed = False
    for year in range(args.start_year, last_year + 1):
        end_year = year + 1
        season = f"{year}-{end_year % 100:02d}"
        print(season, flush=True)
        try:
            with tempfile.TemporaryDirectory(dir=args.output) as temp:
                stage = Path(temp)
                if not args.games_only:
                    adp_year = year if args.adp_year_is == "start" else end_year
                    print(f"  Fetching FantasyPros ADP year {adp_year}", flush=True)
                    headers, records = fetch_adp(adp_year)
                    signature = tuple(sorted(tuple(record[1:]) for record in records))
                    if signature in seen_adp:
                        raise ValueError(f"ADP table matches FantasyPros year {seen_adp[signature]} exactly")
                    seen_adp[signature] = adp_year
                    with (stage / "adp.csv").open("w", encoding="utf-8", newline="") as output:
                        writer = csv.writer(output)
                        writer.writerow(headers)
                        writer.writerows(records)
                    print(f"  ADP: {len(records):,} players")

                uncovered = download_games(stage, end_year, args.play_by_play)
                destination = args.output / season
                destination.mkdir(exist_ok=True)
                for path in stage.iterdir():
                    if path.is_dir():
                        (destination / path.name).mkdir(exist_ok=True)
                        for game in path.iterdir():
                            game.replace(destination / path.name / game.name)
                    else:
                        path.replace(destination / path.name)
                print(f"  Saved to {destination}")
                if uncovered:
                    failed = True
        except (OSError, ValueError, csv.Error) as error:
            print(f"  {season}: {error}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
