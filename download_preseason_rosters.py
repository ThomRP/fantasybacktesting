"""Download dated historical NBA transaction logs from Basketball-Reference (`NBA_{end_year}_transactions.html`)
into `data/bref_transactions/` and compile `data/preseason_rosters.csv` strictly cut off at `tx_date < opening_night`
(zero post-Game-1 data).
"""

import argparse
import csv
import datetime
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from backtest_common import name_key, player_box_scores, read_csv, regular_season_game

BREF_TO_STD = {
    "ATL": "ATL", "BOS": "BOS", "BRK": "BKN", "NJN": "BKN",
    "CHO": "CHA", "CHA": "CHA", "CHH": "CHA", "CHI": "CHI",
    "CLE": "CLE", "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GSW": "GSW", "GS": "GSW",
    "HOU": "HOU", "IND": "IND", "LAC": "LAC", "LAL": "LAL", "MEM": "MEM",
    "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NOP": "NOP", "NOH": "NOP", "NOK": "NOP", "NO": "NOP",
    "NYK": "NYK", "NY": "NYK", "OKC": "OKC", "SEA": "OKC", "ORL": "ORL", "PHI": "PHI",
    "PHO": "PHX", "PHX": "PHX", "POR": "POR", "SAC": "SAC", "SAS": "SAS", "SA": "SAS",
    "TOR": "TOR", "UTA": "UTA", "UTAH": "UTA", "WAS": "WAS", "WSH": "WAS",
}

NAME_ALIASES = {
    "enesfreedom": "eneskanter",
    "eneskanter": "enesfreedom",
    "kjmartin": "kenyonmartin",
    "kenyonmartin": "kjmartin",
}


def norm_team(ab):
    if not ab:
        return ""
    ab = ab.strip().upper()
    return BREF_TO_STD.get(ab, ab)


def parse_bref_date(s: str):
    s = (s or "").strip()
    if not s or "?" in s:
        return None
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def fetch_transactions_html(end_year: int) -> str:
    url = f"https://www.basketball-reference.com/leagues/NBA_{end_year}_transactions.html"
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urlopen(req, timeout=25) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_transactions_file(html_path: Path, min_date=None, max_date_excl=None):
    """Return chronological list of (date, action, player_key, team_std) strictly in [min_date, max_date_excl)."""
    if not html_path.is_file():
        return []
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    events = []
    ul = soup.select_one("ul.page_index")
    if not ul:
        return events
    for li in ul.find_all("li", recursive=False):
        span = li.find("span")
        dt = parse_bref_date(span.get_text(strip=True) if span else "")
        if dt is None:
            continue
        if min_date is not None and dt < min_date:
            continue
        if max_date_excl is not None and dt >= max_date_excl:
            continue
        day_signings, day_trades, day_outs = [], [], []
        for ptag in li.find_all("p"):
            raw_html = str(ptag)
            clean_html = re.sub(r"\([^()]*was later selected[^()]*\)", "", raw_html, flags=re.I)
            is_multi = bool(re.search(r"In a \d+-team trade", clean_html, flags=re.I))
            clauses = clean_html.split(";") if is_multi else [clean_html]
            for cl in clauses:
                csoup = BeautifulSoup(cl, "html.parser")
                txt = csoup.get_text(" ", strip=True).lower()
                from_all = csoup.find_all("a", attrs={"data-attr-from": True})
                to_all = csoup.find_all("a", attrs={"data-attr-to": True})
                t_from = norm_team(from_all[0]["data-attr-from"]) if from_all else ""
                t_to = norm_team(to_all[0]["data-attr-to"]) if to_all else ""

                if t_from and t_to and "traded" in txt:
                    parts = re.split(r'<a[^>]*data-attr-to="[^"]+"[^>]*>.*?</a>', cl, maxsplit=1, flags=re.S)
                    left_html = parts[0]
                    right_html = parts[1] if len(parts) > 1 else ""
                    for pa in BeautifulSoup(left_html, "html.parser").select('a[href^="/players/"]'):
                        day_trades.append((dt, "to", name_key(pa.get_text(strip=True)), t_to))
                    if re.search(r"\bfor\b", BeautifulSoup(right_html, "html.parser").get_text(" ", strip=True), flags=re.I):
                        for pa in BeautifulSoup(right_html, "html.parser").select('a[href^="/players/"]'):
                            day_trades.append((dt, "to", name_key(pa.get_text(strip=True)), t_from))
                elif t_to and any(w in txt for w in ("signed", "claimed", "converted", "matched", "acquired", "drafted", "picked up")):
                    for pa in csoup.select('a[href^="/players/"]'):
                        day_signings.append((dt, "to", name_key(pa.get_text(strip=True)), t_to))
                elif any(w in txt for w in ("waived", "released", "retired", "retirement")):
                    for pa in csoup.select('a[href^="/players/"]'):
                        day_outs.append((dt, "out", name_key(pa.get_text(strip=True)), t_from))
        events.extend(day_outs + day_signings + day_trades)
    return events


def compile_preseason_rosters_csv(data: Path, tx_dir: Path, out_csv: Path):
    season_dirs = sorted(
        (int(p.name[:4]), p)
        for p in data.iterdir()
        if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}", p.name) and (p / "games.csv").is_file()
    )
    s_info = {}
    for yr, folder in season_dirs:
        reg_dates = {}
        tid_to_abbr = {}
        for g in read_csv(folder / "games.csv"):
            if regular_season_game(g):
                reg_dates[g["game_id"]] = g["game_date"][:10]
                if g.get("home_id") and g.get("home_abbreviation"):
                    tid_to_abbr[str(g["home_id"])] = norm_team(g["home_abbreviation"])
                if g.get("away_id") and g.get("away_abbreviation"):
                    tid_to_abbr[str(g["away_id"])] = norm_team(g["away_abbreviation"])
        if not reg_dates:
            continue
        op_date = datetime.date.fromisoformat(min(reg_dates.values()))
        last_game = {}
        if (folder / "player_box_scores.csv").is_file():
            for gid, aid, name, _stats, _pos, _starter, tid in player_box_scores(folder, reg_dates):
                ab = tid_to_abbr.get(str(tid))
                if not ab:
                    continue
                gd_obj = datetime.date.fromisoformat(reg_dates[gid])
                k = name_key(name)
                if aid not in last_game or gd_obj >= last_game[aid][0]:
                    last_game[aid] = (gd_obj, ab, k, name)
        s_info[yr] = (op_date, last_game)

    tx_cache = {}
    for p in sorted(tx_dir.glob("NBA_*_transactions.html")):
        m = re.search(r"NBA_(\d+)_transactions\.html", p.name)
        if m:
            tx_cache[int(m.group(1))] = parse_transactions_file(p)

    rows_out = []
    for target_yr in sorted(s_info):
        prior_yr = target_yr - 1
        if prior_yr not in s_info:
            continue
        op_date, _ = s_info[target_yr]
        prior_op, prior_last = s_info[prior_yr]
        if (target_yr + 1) not in tx_cache:
            continue

        pre_team = {}
        prior_end_team = {}
        last_dt_map = {}
        last_tx_dt = {}
        meta_by_key = {}
        for aid, (dt_last, ab, k, disp) in prior_last.items():
            pre_team[k] = ab
            prior_end_team[k] = ab
            last_dt_map[k] = dt_last
            meta_by_key[k] = (aid, disp)
            if k in NAME_ALIASES:
                alt = NAME_ALIASES[k]
                pre_team[alt] = ab
                prior_end_team[alt] = ab
                last_dt_map[alt] = dt_last

        evs = [e for e in tx_cache.get(target_yr, []) if prior_op <= e[0] < op_date] + [
            e for e in tx_cache.get(target_yr + 1, []) if e[0] < op_date
        ]
        for dt, act, pk, tm in evs:
            assert dt < op_date, f"Leakage guard tripped: {dt} >= {op_date}"
            targets = [pk] + ([NAME_ALIASES[pk]] if pk in NAME_ALIASES else [])
            for tk in targets:
                if tk in last_dt_map and dt <= last_dt_map[tk]:
                    continue
                if act == "to":
                    pre_team[tk] = tm
                    last_tx_dt[tk] = dt.isoformat()
                elif act == "out":
                    if not tm or pre_team.get(tk) == tm:
                        pre_team[tk] = ""
                        last_tx_dt[tk] = dt.isoformat()

        season_str = f"{target_yr}-{(target_yr + 1) % 100:02d}"
        for k, (aid, disp) in sorted(meta_by_key.items()):
            rows_out.append([
                season_str,
                op_date.isoformat(),
                aid,
                k,
                disp,
                prior_end_team[k],
                pre_team.get(k, ""),
                last_tx_dt.get(k, ""),
            ])

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["season", "opening_night", "athlete_id", "player_key", "player", "prior_team", "oct_team", "last_tx_date"])
        w.writerows(rows_out)
    print(f"Compiled {out_csv} ({len(rows_out):,} pre-Opening-Night player-season rows)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-end-year", type=int, default=2009, help="first season end year (2009 for 2008-09)")
    parser.add_argument("--last-end-year", type=int, default=2027, help="last season end year (2027 for 2026-27)")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--delay", type=float, default=3.3)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = args.data / "bref_transactions"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.compile_only:
        to_fetch = [
            end_yr for end_yr in range(args.start_end_year, args.last_end_year + 1)
            if args.force or not (out_dir / f"NBA_{end_yr}_transactions.html").is_file()
            or (out_dir / f"NBA_{end_yr}_transactions.html").stat().st_size <= 5000
        ]
        for idx, end_yr in enumerate(to_fetch):
            out_path = out_dir / f"NBA_{end_yr}_transactions.html"
            print(f"  Fetching NBA_{end_yr}_transactions.html ...", flush=True)
            html = fetch_transactions_html(end_yr)
            out_path.write_text(html, encoding="utf-8")
            print(f"    Saved {out_path.name} ({len(html):,} chars)", flush=True)
            if idx + 1 < len(to_fetch):
                time.sleep(args.delay)

    compile_preseason_rosters_csv(args.data, out_dir, args.data / "preseason_rosters.csv")


if __name__ == "__main__":
    main()
