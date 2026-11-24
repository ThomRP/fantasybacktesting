"""Download and verify authentic historical NBA preseason Vegas Win Totals (O/U) & Championship Odds
from Basketball-Reference (`https://www.basketball-reference.com/leagues/NBA_{end_year}_preseason_odds.html`)
for seasons 2016-17 through 2026-27, saving to `data/vegas_preseason_odds.csv`.

Usage:
  .\\.venv\\Scripts\\python.exe download_vegas_odds.py
"""

import csv
import re
import time
from html import unescape
from pathlib import Path
from urllib.request import Request, urlopen


# Map Basketball-Reference 3-letter team codes -> ESPN team abbreviations used in data/*/games.csv
BREF_TO_ESPN = {
    "ATL": "ATL", "BOS": "BOS", "BRK": "BKN", "NJN": "BKN",
    "CHO": "CHA", "CHA": "CHA", "CHH": "CHA", "CHI": "CHI",
    "CLE": "CLE", "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GSW": "GS",
    "HOU": "HOU", "IND": "IND", "LAC": "LAC", "LAL": "LAL", "MEM": "MEM",
    "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NOP": "NO",  "NOH": "NO", "NOK": "NO",
    "NYK": "NY",  "OKC": "OKC", "SEA": "OKC", "ORL": "ORL", "PHI": "PHI",
    "PHO": "PHX", "POR": "POR", "SAC": "SAC", "SAS": "SA",  "TOR": "TOR",
    "UTA": "UTAH", "WAS": "WSH",
}


def american_to_implied_prob(odds_str: str):
    s = (odds_str or "").strip().replace(",", "")
    if not s:
        return None
    try:
        val = float(s)
    except ValueError:
        return None
    if val > 0:
        return 100.0 / (val + 100.0)
    if val < 0:
        return (-val) / (-val + 100.0)
    return None


def fetch_season_odds(end_year: int):
    url = f"https://www.basketball-reference.com/leagues/NBA_{end_year}_preseason_odds.html"
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urlopen(req, timeout=20) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    # Basketball-Reference sometimes wraps tables inside HTML comments; unwrap them cleanly
    uncommented = html.replace("<!--", "").replace("-->", "")

    # Locate the preseason odds table by id or by O/U column
    tbl_match = re.search(
        r'<table[^>]*id="NBA_\d+_preseason_odds"[^>]*>(.*?)</table>',
        uncommented,
        flags=re.S | re.I,
    )
    if not tbl_match:
        tbl_match = re.search(
            r'<table[^>]*>(?:(?!</table>).)*(?:win_loss_ou|over_under|W-L O/U)(?:(?!</table>).)*</table>',
            uncommented,
            flags=re.S | re.I,
        )
    if not tbl_match:
        tids = re.findall(r'<table[^>]*id="([^"]+)"', uncommented)
        dstats = sorted(set(re.findall(r'data-stat="([^"]+)"', uncommented)))
        raise ValueError(f"Could not find preseason odds table on {url} | tables={tids} | data-stats={dstats[:25]}")

    # Parse each row using flexible column detection via thead / data-stat
    tbody_m = re.search(r"<tbody>(.*?)</tbody>", tbl_match.group(0), flags=re.S | re.I)
    body = tbody_m.group(1) if tbody_m else tbl_match.group(0)

    # Identify column data-stat names from the first data row
    sample_tr = next((tr for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", body, flags=re.S | re.I) if 'class="thead"' not in tr and "<td" in tr), "")
    cell_stats = re.findall(r'data-stat="([^"]+)"', sample_tr)

    season_label = f"{end_year - 1}-{str(end_year)[-2:]}"
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", body, flags=re.S | re.I):
        if 'class="thead"' in tr or "over_header" in tr:
            continue
        cells = dict(re.findall(r'data-stat="([^"]+)"[^>]*>(.*?)</t[hd]>', tr, flags=re.S | re.I))
        if not cells:
            continue
        team_cell = next((cells[k] for k in ("team", "team_name", "franchise") if k in cells), None)
        odds_cell = next((cells[k] for k in ("odds", "champ_odds", "championship_odds") if k in cells), "")
        ou_cell = next((cells[k] for k in ("win_loss_ou", "over_under", "win_ou", "wins_over_under", "w_l_ou", "ou") if k in cells), None)
        if ou_cell is None:
            # Fallback: find any key containing "ou" or "over"
            ou_cell = next((v for k, v in cells.items() if "ou" in k.lower() or "over" in k.lower()), None)
        if not team_cell or not ou_cell:
            continue
        abbr_m = re.search(r"/teams/([A-Z]{3})/", team_cell)
        bref_abbr = abbr_m.group(1) if abbr_m else ""
        team_name = unescape(re.sub(r"<[^>]+>", "", team_cell)).strip()
        odds_txt = unescape(re.sub(r"<[^>]+>", "", odds_cell)).strip()
        ou_txt = unescape(re.sub(r"<[^>]+>", "", ou_cell)).strip()
        if not team_name or not ou_txt:
            continue
        try:
            win_ou = float(ou_txt)
        except ValueError:
            continue
        rows.append({
            "season": season_label,
            "end_year": end_year,
            "bref_abbr": bref_abbr,
            "espn_abbr": BREF_TO_ESPN.get(bref_abbr, bref_abbr),
            "team_name": team_name,
            "champ_odds": odds_txt,
            "champ_implied_prob": round(american_to_implied_prob(odds_txt), 5) if american_to_implied_prob(odds_txt) is not None else "",
            "win_ou": win_ou,
            "source_url": url,
        })

    if len(rows) != 30:
        raise ValueError(f"Expected 30 teams for {season_label}, parsed {len(rows)} (cell_stats={cell_stats}) from {url}")

    # Normalize win_ou to expected win percentage (handles 72-game 2020-21 season automatically!)
    avg_ou = sum(r["win_ou"] for r in rows) / len(rows)
    implied_sched_games = round(2.0 * avg_ou)  # ~82 for normal seasons, ~72 for 2020-21
    for r in rows:
        r["sched_games_implied"] = implied_sched_games
        r["win_ou_pct"] = round(r["win_ou"] / max(2.0 * avg_ou, 1.0), 4)
        r["win_ou_82"] = round(82.0 * r["win_ou_pct"], 2)

    return rows


def main():
    all_rows = []
    for idx, end_year in enumerate(range(2009, 2028)):
        if idx > 0:
            time.sleep(3.5)  # Polite rate limit for Basketball-Reference
        rows = fetch_season_odds(end_year)
        all_rows.extend(rows)
        ordered = sorted(rows, key=lambda r: -r["win_ou"])
        top2 = ", ".join(f"{r['bref_abbr']} {r['win_ou']:.1f}" for r in ordered[:2])
        bot2 = ", ".join(f"{r['bref_abbr']} {r['win_ou']:.1f}" for r in ordered[-2:])
        avg_ou = sum(r["win_ou"] for r in rows) / len(rows)
        print(
            f"  {rows[0]['season']} (30 teams, avg O/U={avg_ou:.1f}, ~{rows[0]['sched_games_implied']}g) | "
            f"Top: {top2} | Bot: {bot2}"
        )

    out_path = Path("data") / "vegas_preseason_odds.csv"
    fieldnames = [
        "season", "end_year", "bref_abbr", "espn_abbr", "team_name",
        "win_ou", "win_ou_82", "win_ou_pct", "sched_games_implied",
        "champ_odds", "champ_implied_prob", "source_url",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} verified preseason Vegas rows to {out_path.resolve()}")


if __name__ == "__main__":
    main()
