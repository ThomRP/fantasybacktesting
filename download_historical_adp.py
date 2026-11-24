"""Download verified pre-2016 historical NBA ADP (2008-09 through 2015-16) and SportsDataverse
player_core profiles (2007-2015) so the engine has 18 full historical seasons (2008-09..2025-26).

Sources (all 100% pre-Opening-Night, zero future leakage):
  - 2014-15 & 2015-16: Wayback Machine pre-Opening-Night snapshots of FantasyPros `overall.php`
      * 2014-15: 2014-10-24 (4 days before Oct 28, 2014 Opening Night) — Yahoo/ESPN/CBS/AVG (~215 rows)
      * 2015-16: 2015-10-25 (2 days before Oct 27, 2015 Opening Night) — Yahoo/ESPN/CBS/AVG (~230 rows)
  - 2008-09 through 2013-14: Official ESPN Fantasy Basketball v3 API
      `lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{end_year}/players?view=kona_player_info`
      Rank-ordered by `ownership.averageDraftPosition` (exact 1:1 match with FantasyPros's ESPN column,
      142-155 drafted players/season, exact ESPN athlete IDs).

Usage:
  .\\.venv\\Scripts\\python.exe download_historical_adp.py
"""
import csv
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from adp_names import clean_adp_player_name
from backtest_common import load_player_profiles

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# ESPN Fantasy slot ID -> standard position label
ESPN_POS_MAP = {1: "PG", 2: "SG", 3: "SF", 4: "PF", 5: "C"}
ESPN_SLOT_POS = {0: "PG", 1: "SG", 2: "SF", 3: "PF", 4: "C"}

# ESPN proTeamId -> standard team abbreviation
ESPN_TEAM_MAP = {
    1: "ATL", 2: "BOS", 3: "NO", 4: "CHI", 5: "CLE", 6: "DAL", 7: "DEN", 8: "DET",
    9: "GS", 10: "HOU", 11: "IND", 12: "LAC", 13: "LAL", 14: "MIA", 15: "MIL", 16: "MIN",
    17: "BKN", 18: "NY", 19: "ORL", 20: "PHI", 21: "PHX", 22: "POR", 23: "SAC", 24: "SA",
    25: "OKC", 26: "UTAH", 27: "WSH", 28: "TOR", 29: "MEM", 30: "CHA",
}


def fetch_wayback_fantasypros(start_year: int, ts: str):
    """Parse pre-Opening-Night Wayback FantasyPros ADP table cleanly (handling legacy unclosed <tr> HTML)."""
    url = f"https://web.archive.org/web/{ts}id_/https://www.fantasypros.com/nba/adp/overall.php"
    html = urlopen(Request(url, headers=HEADERS), timeout=30).read().decode("utf-8", errors="replace")
    # Use html5lib/lxml if available, else parse every <td> sequence cleanly from tbody
    soup = BeautifulSoup(html, "html.parser")
    target_tbl = None
    for tbl in soup.find_all("table"):
        ths = [th.get_text(" ", strip=True).lower() for th in tbl.find_all("th")]
        if any("player" in h for h in ths) and any(h in ("avg", "average") for h in ths):
            target_tbl = tbl
            break
    if target_tbl is None:
        raise ValueError(f"Could not find ADP table in Wayback snapshot {ts}")

    # Legacy 2014-2015 FP HTML often omits </tr>, causing html.parser to nest <tr> tags.
    # Extract flat leaf <td> cells directly in document order in chunks of 7: [Rank, Player, POS, Yahoo, ESPN, CBS, AVG]
    tbody = target_tbl.find("tbody") or target_tbl
    tds = [td for td in tbody.find_all("td") if not td.find("td")]
    rows = []
    i = 0
    seen_ranks = set()
    while i + 6 < len(tds):
        rk_txt = tds[i].get_text(" ", strip=True)
        if not rk_txt.isdigit():
            i += 1
            continue
        rk = int(rk_txt)
        if rk in seen_ranks or rk < 1 or rk > 500:
            i += 1
            continue
        p_cell = tds[i + 1]
        pos_txt = tds[i + 2].get_text(" ", strip=True)
        yah_txt = tds[i + 3].get_text(" ", strip=True)
        espn_txt = tds[i + 4].get_text(" ", strip=True)
        cbs_txt = tds[i + 5].get_text(" ", strip=True)
        avg_txt = tds[i + 6].get_text(" ", strip=True)
        if not re.fullmatch(r"\d+(?:\.\d+)?", avg_txt):
            i += 1
            continue

        link = p_cell.find("a")
        raw_name = link.get_text(" ", strip=True) if link else p_cell.get_text(" ", strip=True)
        pname = clean_adp_player_name(raw_name)
        purl = link.get("href", "") if link else ""
        # Extract team abbreviation from cell text (e.g. "( CLE )" or "NYK")
        cell_txt = p_cell.get_text(" ", strip=True)
        tm_m = re.search(r"\(\s*([A-Z]{2,4})\s*\)", cell_txt)
        if not tm_m:
            after = cell_txt[len(raw_name):].strip() if cell_txt.startswith(raw_name) else cell_txt
            tm_m = re.search(r"\b([A-Z]{2,3})\b", after)
        team_abbr = tm_m.group(1) if tm_m else ""
        # POS column in 2014/2015 is e.g. "F1", "G12", "C3" -> strip digits
        pos_clean = re.sub(r"\d+", "", pos_txt).strip()
        player_disp = f"{pname} ({team_abbr} - {pos_clean})" if (team_abbr and pos_clean) else (f"{pname} ({team_abbr})" if team_abbr else pname)

        seen_ranks.add(rk)
        rows.append({
            "adp_year": start_year,
            "player_name": pname,
            "fantasypros_player_url": purl,
            "Rank": rk,
            "Player": player_disp,
            "Yahoo": yah_txt,
            "ESPN": espn_txt,
            "CBS": cbs_txt,
            "AVG": avg_txt,
        })
        i += 7

    if len(rows) < 150:
        raise ValueError(f"Expected >=150 rows in Wayback FP {start_year} ({ts}), parsed only {len(rows)}")
    return rows


def fetch_espn_v3_adp(start_year: int):
    """Fetch official ESPN Fantasy Basketball v3 API historical ADP for season start_year -> start_year+1."""
    end_year = start_year + 1
    url = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{end_year}/players?view=kona_player_info"
    hdrs = dict(HEADERS)
    hdrs["x-fantasy-filter"] = json.dumps({"players": {"limit": 2500}})
    players = json.loads(urlopen(Request(url, headers=hdrs), timeout=25).read().decode("utf-8", errors="replace"))

    drafted = []
    for p in players:
        own = p.get("ownership") or {}
        raw_adp = own.get("averageDraftPosition")
        if raw_adp is None:
            continue
        raw_adp = float(raw_adp)
        if not (0.5 <= raw_adp < 139.95):
            continue
        # Normalize double spaces in older ESPN names (e.g. "Kevin  Durant")
        full_name = re.sub(r"\s+", " ", (p.get("fullName") or "")).strip()
        pname = clean_adp_player_name(full_name)
        if not pname:
            continue
        # Positions from defaultPositionId + eligibleSlots
        pos_list = []
        dp = ESPN_POS_MAP.get(p.get("defaultPositionId"))
        if dp:
            pos_list.append(dp)
        for sl in (p.get("eligibleSlots") or []):
            sp = ESPN_SLOT_POS.get(sl)
            if sp and sp not in pos_list:
                pos_list.append(sp)
        pos_str = ",".join(pos_list)
        team_abbr = ESPN_TEAM_MAP.get(p.get("proTeamId"), "")
        drafted.append((raw_adp, int(p.get("id") or 0), pname, team_abbr, pos_str))

    drafted.sort(key=lambda x: (x[0], x[1]))
    rows = []
    for rk, (raw_adp, aid, pname, team_abbr, pos_str) in enumerate(drafted, 1):
        player_disp = f"{pname} ({team_abbr} - {pos_str})" if (team_abbr and pos_str) else (f"{pname} ({team_abbr})" if team_abbr else pname)
        # Note: FantasyPros's `ESPN` column is exact integer rank (1..N) of this ESPN v3 ADP list,
        # which also removes the 10-team pick-130 ceiling compression on late-round picks (#100-#150).
        rows.append({
            "adp_year": start_year,
            "player_name": pname,
            "fantasypros_player_url": "",
            "Rank": rk,
            "Player": player_disp,
            "Yahoo": "",
            "ESPN": f"{rk:.1f}",
            "CBS": "",
            "AVG": f"{rk:.1f}",
            "espn_raw_adp": f"{raw_adp:.2f}",
            "athlete_id": str(aid),
        })
    return rows


def main():
    data = Path("data")

    print("=== 1. Caching SportsDataverse player_core profiles (2007..2015) ===")
    for end_year in range(2007, 2016):
        profs = load_player_profiles(data, end_year)
        print(f"  player_core_{end_year}.csv: {len(profs):4d} profiles")

    print("\n=== 2. Downloading ESPN v3 API Historical ADP (2008-09 .. 2013-14) ===")
    espn_fields = ["adp_year", "player_name", "fantasypros_player_url", "Rank", "Player", "Yahoo", "ESPN", "CBS", "AVG", "espn_raw_adp", "athlete_id"]
    for start_year in range(2008, 2014):
        season = f"{start_year}-{str(start_year + 1)[-2:]}"
        rows = fetch_espn_v3_adp(start_year)
        out = data / season / "adp.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=espn_fields)
            w.writeheader()
            w.writerows(rows)
        top3 = ", ".join(f"{r['player_name']} ({r['espn_raw_adp']})" for r in rows[:3])
        print(f"  {season}: {len(rows):3d} players -> {out} | Top 3: {top3}")

    print("\n=== 3. Downloading Wayback Pre-Opening-Night FantasyPros ADP (2014-15, 2015-16) ===")
    fp_fields = ["adp_year", "player_name", "fantasypros_player_url", "Rank", "Player", "Yahoo", "ESPN", "CBS", "AVG"]
    for start_year, ts in ((2014, "20141024085300"), (2015, "20151025164241")):
        season = f"{start_year}-{str(start_year + 1)[-2:]}"
        rows = fetch_wayback_fantasypros(start_year, ts)
        out = data / season / "adp.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fp_fields)
            w.writeheader()
            w.writerows(rows)
        top3 = ", ".join(f"{r['player_name']} ({r['AVG']})" for r in rows[:3])
        print(f"  {season} (snap {ts}): {len(rows):3d} players -> {out} | Top 3: {top3}")


if __name__ == "__main__":
    main()
