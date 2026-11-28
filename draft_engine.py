"""Unified Fantasy Basketball Draft Engine: Stage-1 Pre-ADP (Pure Stats + Live Injuries) & Stage-2 Residual (ADP-Aware).

Two-Stage Workflow:
  Stage 1 — Pre-ADP (--mode pure_stats, June–August before drafts open):
    * 100% ADP-free core model trained strictly on Year 0 / Year -1 box scores + age (veterans)
      and NBA draft pick + rookie age + Year-0 team context (rookies).
    * Optional --use-live-injuries pulls ESPN's public NBA injury JSON (return dates + season-enders)
      to cap availability directly inside AdjFPPG without needing ADP.
    * Supports --rookies-csv and --injuries-csv for offline/manual draft-day inputs.

  Stage 2 — Draft Season (--mode residual, September–October once ADP is live):
    * Runs the validated 8-season champion (`unified_equal`, default --lam 0.50):
      50/50 Huber + 3-part Structural Decomposition (MPG x FPPM x Availability) for veterans
      with a scale-invariant market-collapse hinge, plus the 11-feature Rookie Hype-Tax Huber model.
    * Also respects --use-live-injuries / --injuries-csv and --exclude-players for late-breaking news.

  Both stages recalculate every feature, target, and ESPN simulation game log under any custom
  points-league scoring via --scoring (default | espn | yahoo | sleeper | path.json | '{...}').
"""

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import mean, median
from urllib.request import Request, urlopen

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from adp_names import clean_adp_player_name
from backtest_common import (
    SCORING as DEFAULT_SCORING,
    adp_slot_points,
    load_birth_dates,
    load_player_profiles,
    name_key,
    parse_minutes,
    player_box_scores,
    read_csv,
    read_season,
    regular_season_game,
    write_csv,
)
from backtest_rookies import FIRST_ARCHIVE_YEAR, players_with_appearances


ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

BOARD_SIZE = 150
REF_GAMES = 65.0
WAIVER_FACTOR = 0.85
SEASON_DAYS = 175.0
SEASON_GAMES = 82.0

MIN_VET_TRAIN_GAMES = 15
MAX_VET_TRAIN_RANK = 260
MAX_ROOKIE_TRAIN_PICK = 45
MAX_RESIDUAL_VET_ADP = 175.0
MAX_RESIDUAL_ROOKIE_ADP = 200.0
COLLAPSE_WAIVER_RATIO = 0.38   # ~7.9 FPPG on ESPN default; scales automatically to any custom scoring
COLLAPSE_MIN_SLOT_DROP = 20.0
RIDGE_ALPHAS = np.logspace(-1, 3.5, 25)

SCORING_PRESETS = {
    "default": dict(DEFAULT_SCORING),
    "espn": {
        "points": 1.0,
        "three_point_field_goals_made": 1.0,
        "field_goals_attempted": -1.0,
        "field_goals_made": 2.0,
        "free_throws_attempted": -1.0,
        "free_throws_made": 1.0,
        "rebounds": 1.0,
        "assists": 2.0,
        "steals": 4.0,
        "blocks": 4.0,
        "turnovers": -2.0,
    },
    "yahoo": {
        "points": 1.0,
        "rebounds": 1.2,
        "assists": 1.5,
        "steals": 3.0,
        "blocks": 3.0,
        "turnovers": -1.0,
    },
    "sleeper": {
        "points": 0.5,
        "three_point_field_goals_made": 0.5,
        "rebounds": 1.0,
        "assists": 1.0,
        "steals": 2.0,
        "blocks": 2.0,
        "turnovers": -1.0,
    },
}

# --- Stage 1: 100% ADP-FREE Feature Sets ---
VET_STATS_FEATURES = (
    "prior_adj_fppg", "prior_fppg", "prior_mpg", "prior_fppm", "prior_games",
    "prior_start_pct", "prior_post_break_fppg_chg", "prior_post_break_mpg_chg",
    "prior_tp_pct", "age_at_prior_break", "age_under_24", "age_over_30",
    "age30_x_low_g", "young_x_post_mpg",
    "yoy_fppg_delta", "yoy_mpg_delta", "two_yr_weighted_fppg",
    "two_yr_games_avg", "no_prev2_season",
)

VET_STATS_MPG_FEATURES = (
    "prior_mpg", "prior_post_break_mpg_chg", "prior_games", "prior_start_pct",
    "prior_fppg", "age_at_prior_break", "age_under_24", "age_over_30",
    "yoy_mpg_delta", "two_yr_games_avg", "no_prev2_season",
)

VET_STATS_FPPM_FEATURES = (
    "prior_fppm", "prior_fppg", "prior_post_break_fppg_chg", "prior_mpg",
    "prior_tp_pct", "age_at_prior_break", "age_under_24", "age_over_30",
    "yoy_fppg_delta", "two_yr_weighted_fppg", "no_prev2_season",
)

VET_STATS_AVAIL_FEATURES = (
    "prior_games", "two_yr_games_avg", "prior_mpg", "prior_start_pct",
    "prior_fppg", "age_at_prior_break", "age_under_24", "age_over_30",
    "no_prev2_season",
)

ROOKIE_STATS_FEATURES = (
    "log_draft_pick", "is_top3_pick", "is_lottery_pick", "is_undrafted",
    "is_redshirt", "rookie_age", "height_inches",
    "team_top8_avg_age", "team_vegas_win_pct", "team_vet30_min_share",
    "team_prior_top150_count",
)

# --- Stage 2: ADP-Aware Residual Feature Sets (unified_equal champion) ---
VET_RESIDUAL_FEATURES = (
    "log_adp", "mkt_expected_adj_fppg", "prior_minus_mkt_capped", "market_collapse_excess",
    "age_at_prior_break", "age_under_24", "age_over_30",
    "prior_games", "prior_mpg", "prior_fppm", "prior_start_pct",
    "prior_post_break_mpg_chg", "prior_tp_pct",
    "age30_x_low_g", "young_x_post_mpg", "log_adp_x_age30",
    "yoy_fppg_delta", "yoy_mpg_delta", "two_yr_weighted_fppg",
    "two_yr_games_avg", "no_prev2_season", "adp_platform_std", "espn_vs_avg",
    "changed_team", "net_vacated_pos_mpg", "net_vacated_val",
)

VET_RES_MPG_FEATURES = (
    "log_adp", "mkt_expected_adj_fppg", "prior_minus_mkt_capped", "market_collapse_excess",
    "age_at_prior_break", "age_under_24", "age_over_30", "prior_games", "prior_mpg",
    "prior_start_pct", "prior_post_break_mpg_chg", "yoy_mpg_delta", "two_yr_games_avg", "no_prev2_season",
    "adp_platform_std", "espn_vs_avg",
    "changed_team", "net_vacated_pos_mpg", "net_vacated_val",
)

VET_RES_FPPM_FEATURES = (
    "log_adp", "mkt_expected_adj_fppg", "prior_minus_mkt_capped", "market_collapse_excess",
    "age_at_prior_break", "age_under_24", "age_over_30", "prior_fppm", "prior_mpg",
    "prior_tp_pct", "yoy_fppg_delta", "two_yr_weighted_fppg", "no_prev2_season",
    "adp_platform_std", "espn_vs_avg",
    "changed_team", "net_vacated_val",
)

VET_RES_AVAIL_FEATURES = (
    "log_adp", "mkt_expected_adj_fppg", "prior_minus_mkt_capped", "market_collapse_excess",
    "age_at_prior_break", "age_under_24", "age_over_30", "prior_games", "two_yr_games_avg",
    "prior_mpg", "prior_start_pct", "no_prev2_season", "adp_platform_std", "espn_vs_avg",
)

ROOKIE_RESIDUAL_FEATURES = (
    "log_draft_pick", "log_adp", "mkt_expected_adj_fppg", "adp_minus_pick_expected",
    "is_redshirt", "rookie_age", "team_top8_avg_age", "team_vegas_win_pct",
    "team_vet30_min_share", "team_prior_top150_count", "espn_vs_avg",
)

# --- Separable Advanced Impact & Usage Extension (opt-in via --advanced-stats) ---
VET_STATS_ADV_FEATURES = ("prior_bpm", "prior_vorp", "prior_on_off", "prior_usg_pct")
VET_STATS_MPG_ADV_FEATURES = ("prior_bpm", "prior_on_off", "prior_usg_pct")
VET_STATS_FPPM_ADV_FEATURES = ("prior_on_off", "prior_usg_pct")
VET_RES_MPG_ADV_FEATURES = ("prior_usg_pct",)
VET_RES_FPPM_ADV_FEATURES = ("prior_usg_pct",)

BACKTEST_STRATEGIES = (
    "market",
    "pure_stats",
    "pure_stats_news_cut",
    "blend_adp_50",
    "residual_unified",
    "residual_wait",
)
DEFAULT_WAIT_LAM = 0.85
LATE_LAM_RAMP_START = 100.0
LATE_LAM_RAMP_END = 150.0

# Standard ESPN Roster Settings (10 starters + 3 bench, 4-C primary cap)
VALID_POS = ("PG", "SG", "SF", "PF", "C")
SLOT_NAMES_10 = ("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL", "UTIL", "UTIL")
SLOT_MASK_MAP = {
    "PG": (1 << 0) | (1 << 5),
    "SG": (1 << 1) | (1 << 5),
    "SF": (1 << 2) | (1 << 6),
    "PF": (1 << 3) | (1 << 6),
    "C":  (1 << 4),
}
UTIL_BITS = (1 << 7) | (1 << 8) | (1 << 9)


def _safe_float(raw):
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def _safe_int(raw):
    val = _safe_float(raw)
    return int(round(val)) if val is not None else None


def _parse_iso_date(raw):
    if not raw:
        return None
    s = str(raw).strip()
    if len(s) >= 10:
        try:
            return date.fromisoformat(s[:10])
        except ValueError:
            pass
    return None


def season_label(start_year):
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def resolve_scoring(scoring_arg):
    """Parse --scoring preset name, JSON file path, or inline JSON dict."""
    if not scoring_arg or scoring_arg.lower() == "default":
        return "default", dict(SCORING_PRESETS["default"])
    low = scoring_arg.lower()
    if low in SCORING_PRESETS:
        return low, dict(SCORING_PRESETS[low])
    path = Path(scoring_arg)
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return path.stem, {str(k): float(v) for k, v in loaded.items()}
    try:
        loaded = json.loads(scoring_arg)
        if not isinstance(loaded, dict) or not loaded:
            raise ValueError("scoring JSON must be a non-empty object")
        return "custom", {str(k): float(v) for k, v in loaded.items()}
    except json.JSONDecodeError as err:
        raise ValueError(
            f"unknown --scoring {scoring_arg!r}. Use a preset ({', '.join(SCORING_PRESETS)}), "
            "a JSON file path, or a JSON dict like '{\"points\":1,\"rebounds\":1.2}'"
        ) from err


def score_box_stats(stats, scoring):
    total = 0.0
    for stat_key, weight in scoring.items():
        val = _safe_float(stats.get(stat_key))
        if val is not None:
            total += val * weight
    return total


def _pos_to_masks(pos_tuple):
    m7 = 0
    for p in pos_tuple:
        m7 |= SLOT_MASK_MAP.get(p, 0)
    if m7 == 0:
        m7 = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 5) | (1 << 6)
    return m7, m7 | UTIL_BITS


def _dfs_match(u, match_slot, masks, vis):
    m = masks[u]
    while m:
        lsb = m & -m
        s = lsb.bit_length() - 1
        m ^= lsb
        if not (vis[0] & lsb):
            vis[0] |= lsb
            prev = match_slot[s]
            if prev == -1 or _dfs_match(prev, match_slot, masks, vis):
                match_slot[s] = u
                return True
    return False


def max_matching_7(roster_indices, masks7):
    match_slot = [-1] * 7
    matched = 0
    for u in roster_indices:
        if _dfs_match(u, match_slot, masks7, [0]):
            matched += 1
            if matched == 7:
                break
    return matched


def assign_lineup_slots(roster_indices, masks10, prio):
    ordered = sorted(roster_indices, key=lambda idx: -prio[idx])
    match_slot = [-1] * 10
    filled = 0
    for idx in ordered:
        if filled < 10 and _dfs_match(idx, match_slot, masks10, [0]):
            filled += 1
    slot_of = {idx: "BE" for idx in roster_indices}
    for s_idx, p_idx in enumerate(match_slot):
        if p_idx != -1:
            slot_of[p_idx] = SLOT_NAMES_10[s_idx]
    return slot_of


def normalize_pos_tuple(raw_pos):
    if not raw_pos:
        return ("SG", "SF")
    tokens = []
    for tok in re.split(r"[,/\s]+", str(raw_pos)):
        t = tok.strip().upper()
        if t in VALID_POS and t not in tokens:
            tokens.append(t)
        elif t == "G":
            for gp in ("PG", "SG"):
                if gp not in tokens:
                    tokens.append(gp)
        elif t == "F":
            for fp in ("SF", "PF"):
                if fp not in tokens:
                    tokens.append(fp)
    return tuple(tokens) if tokens else ("SG", "SF")


def read_adp_entries(path):
    """Read optional ADP file with positions and cross-platform spreads (Yahoo/ESPN/CBS vs AVG)."""
    entries, seen = [], set()
    if not path.is_file():
        return entries
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in (reader.fieldnames or [])}
        name_col = cols.get("playername") or cols.get("player")
        avg_col = next((cols[k] for k in ("avg", "average", "adp", "averagedraftposition", "consensusadp") if k in cols), None)
        if not name_col or not avg_col:
            return entries
        raw_cols = [cols[k] for k in ("player", "playername") if k in cols]
        for row in reader:
            raw_text = next(
                (row.get(c) or "" for c in raw_cols if re.search(r"\([^()]+\s+-\s+[^()]+\)", row.get(c) or "")),
                row.get(name_col) or "",
            )
            cleaned = clean_adp_player_name(raw_text)
            adp = _safe_float(row.get(avg_col))
            if not cleaned or adp is None or adp <= 0:
                continue
            key = name_key(cleaned)
            if key in seen:
                continue
            seen.add(key)

            espn = _safe_float(row.get(cols["espn"])) if "espn" in cols else None
            platforms = [_safe_float(row.get(cols[p])) for p in ("yahoo", "espn", "cbs") if p in cols]
            valid = [v for v in platforms if v is not None and v > 0]
            std = float(np.std(valid, ddof=1)) if len(valid) >= 2 else None
            espn_vs_avg = (adp - espn) if (espn is not None and espn > 0) else None

            adp_team_abbr = None
            pos_tuple = None
            m = re.search(r"\(([^()\s-]+)\s+-\s+([^()]+)\)", raw_text)
            if m:
                adp_team_abbr = m.group(1).strip().upper()
                pos_tuple = normalize_pos_tuple(m.group(2))

            entries.append({
                "key": key,
                "player": cleaned,
                "adp": adp,
                "adp_platform_std": std,
                "espn_vs_avg": espn_vs_avg,
                "adp_team_abbr": adp_team_abbr,
                "adp_positions": pos_tuple,
            })
    entries.sort(key=lambda r: (r["adp"], r["player"].casefold()))
    return entries


# ==============================================================================
# Live ESPN Injury Fetcher & Availability Cap Layer (0% ADP)
# ==============================================================================

def _extract_athlete_id_from_espn_obj(athlete_obj):
    if not isinstance(athlete_obj, dict):
        return None
    direct = str(athlete_obj.get("id") or "").strip()
    if direct.isdigit():
        return direct
    for link in athlete_obj.get("links") or []:
        href = str(link.get("href") or "")
        m = re.search(r"/id/(\d+)", href)
        if m:
            return m.group(1)
    return None


def parse_espn_injuries_payload(payload):
    """Parse ESPN NBA injuries JSON into a normalized list of active injury records (skipping healthy Active notes)."""
    items = []
    if isinstance(payload, dict):
        top = payload.get("injuries") or []
    elif isinstance(payload, list):
        top = payload
    else:
        return items

    for entry in top:
        if not isinstance(entry, dict):
            continue
        if isinstance(entry.get("injuries"), list):
            team_name = entry.get("displayName") or entry.get("team", {}).get("displayName") or ""
            for inj in entry["injuries"]:
                if isinstance(inj, dict):
                    items.append((team_name, inj))
        elif "athlete" in entry or "status" in entry:
            items.append(("", entry))

    records = []
    for team_name, inj in items:
        athlete = inj.get("athlete") or {}
        details = inj.get("details") or {}
        status = str(inj.get("status") or details.get("fantasyStatus", {}).get("description") or "").strip()
        short_c = str(inj.get("shortComment") or "").strip()
        long_c = str(inj.get("longComment") or "").strip()
        inj_type = str(details.get("type") or "").strip()
        inj_detail = str(details.get("detail") or "").strip()
        return_date = _parse_iso_date(details.get("returnDate") or inj.get("returnDate"))

        # ESPN's feed includes healthy roster notes under status="Active" with no injury detail/return date
        if status.lower() == "active" and not inj_type and return_date is None:
            continue

        aid = _extract_athlete_id_from_espn_obj(athlete)
        names = set()
        for col in ("displayName", "fullName", "shortName"):
            val = str(athlete.get(col) or "").strip()
            if val:
                names.add(val)
        display_name = str(athlete.get("displayName") or athlete.get("fullName") or "").strip()
        if not display_name and not aid:
            continue

        records.append({
            "athlete_id": aid,
            "player": display_name,
            "name_keys": {name_key(n) for n in names},
            "team": team_name,
            "status": status,
            "injury_type": inj_type,
            "injury_detail": inj_detail,
            "return_date": return_date,
            "comment": f"{short_c} {long_c}".strip(),
        })
    return records


def fetch_espn_live_injuries(cache_path=None, timeout=20):
    """Fetch live NBA injuries JSON from ESPN public API (with optional local JSON cache fallback)."""
    req = Request(
        ESPN_INJURIES_URL,
        headers={"User-Agent": "Mozilla/5.0 (fantasy-draft-engine/1.0)"},
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw_bytes = resp.read()
        payload = json.loads(raw_bytes.decode("utf-8"))
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(raw_bytes)
        return parse_espn_injuries_payload(payload), "live_espn_api"
    except Exception as err:
        if cache_path is not None and cache_path.is_file():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return parse_espn_injuries_payload(payload), f"cached_espn_json ({err})"
        raise OSError(f"failed to fetch live ESPN injuries ({err})") from err


def load_injuries_csv(path):
    """Load manual/offline injury CSV with columns: player, status, return_date (YYYY-MM-DD), notes."""
    records = []
    if not path or not path.is_file():
        return records
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("player") or row.get("name") or "").strip()
            if not name:
                continue
            status = (row.get("status") or "Out").strip()
            ret = _parse_iso_date(row.get("return_date") or row.get("returnDate"))
            notes = (row.get("notes") or row.get("comment") or row.get("injury_type") or "").strip()
            records.append({
                "athlete_id": (row.get("athlete_id") or "").strip() or None,
                "player": name,
                "name_keys": {name_key(name)},
                "team": (row.get("team") or "").strip(),
                "status": status,
                "injury_type": (row.get("injury_type") or "").strip(),
                "injury_detail": (row.get("injury_detail") or "").strip(),
                "return_date": ret,
                "comment": notes,
            })
    return records


def evaluate_injury_availability_cap(inj_rec, target_year, opening_night=None, as_of_date=None):
    """Translate an injury record into (avail_cap_or_None, est_missed_games, label).

    Anchors remaining-season games to ``ref_start = max(opening_night, as_of_date or date.today())``:
    - Season-ending injury (Out for Season / return >= March 15) -> avail_cap = 0.0
    - Delayed return past ref_start (>= 5 est. games missed) -> cap remaining games to
      (rem_season_games - est_missed) * rehab_factor / rem_ref_games
    - Minor / pre-ref_start / DTD -> no cap (None), informational status label only
    """
    op_date = opening_night or date(target_year, 10, 21)
    today = as_of_date if as_of_date is not None else date.today()
    ref_start = max(op_date, today)
    season_end_cutoff = date(target_year + 1, 3, 15)

    elapsed_days = max(0, (ref_start - op_date).days)
    rem_season_days = max(0.0, SEASON_DAYS - elapsed_days)
    rem_season_frac = rem_season_days / SEASON_DAYS
    rem_season_games = SEASON_GAMES * rem_season_frac
    rem_ref_games = REF_GAMES * rem_season_frac

    status_l = (inj_rec.get("status") or "").lower()
    text_l = " ".join([
        status_l,
        (inj_rec.get("injury_type") or "").lower(),
        (inj_rec.get("injury_detail") or "").lower(),
        (inj_rec.get("comment") or "").lower(),
    ])
    ret_date = inj_rec.get("return_date")

    season_ender_phrases = (
        "out for season", "out for the season", "season-ending", "season ending",
        "miss the entire", f"miss the {target_year}-{str(target_year + 1)[-2:]}",
        "will not play this season", "ruled out for the season",
    )
    severe_tear_phrases = ("torn acl", "acl tear", "torn achilles", "achilles tear", "achilles rupture", "ruptured achilles")
    major_rehab_phrases = ("achilles", "acl", "meniscus", "surgery", "fracture", "stress", "patellar", "mcl", "pcl")

    is_out_like = any(k in status_l for k in ("out", "ir", "injured reserve", "suspension", "suspended"))
    is_season_ender = (
        any(p in text_l for p in season_ender_phrases)
        or (ret_date is not None and ret_date >= season_end_cutoff)
        or (is_out_like and ret_date is None and any(p in text_l for p in severe_tear_phrases))
    )
    if is_season_ender:
        ret_s = f" ret {ret_date.isoformat()}" if ret_date is not None else ""
        return 0.0, int(round(rem_season_games)), f"OUT_SEASON({inj_rec.get('status') or 'Out'}{ret_s})"

    if ret_date is not None and ret_date > ref_start:
        days_missed = (ret_date - ref_start).days
        est_missed = int(round((days_missed / SEASON_DAYS) * SEASON_GAMES))
        if est_missed >= 5 and rem_ref_games > 0.0:
            rem_games = max(0.0, rem_season_games - est_missed)
            rehab_mult = 0.85 if any(p in text_l for p in major_rehab_phrases) else 0.95
            avail_cap = min(1.0, (rem_games * rehab_mult) / rem_ref_games)
            return avail_cap, est_missed, f"INJ_CAP(ret {ret_date.isoformat()}, ~{est_missed}g miss)"

    if is_out_like:
        return None, 0, f"INJ_NOTE({inj_rec.get('status')})"
    if status_l:
        return None, 0, f"INJ_NOTE({inj_rec.get('status')})"
    return None, 0, ""


def apply_injury_report_layer(rows, projections, proj_mpg, proj_fppm, proj_avail, injury_records, target_year, opening_night=None, as_of_date=None):
    """Apply live ESPN / CSV injury return-date availability caps directly onto AdjFPPG projections."""
    by_id = {}
    by_key = {}
    for rec in injury_records:
        if rec.get("athlete_id"):
            by_id[str(rec["athlete_id"])] = rec
        for k in rec.get("name_keys", ()):
            by_key[k] = rec

    adjusted = np.array(projections, dtype=float)
    labels = [""] * len(rows)
    capped_count = 0

    for i, r in enumerate(rows):
        inj = by_id.get(str(r.get("athlete_id") or "")) or by_key.get(name_key(r["player"]))
        if inj is None:
            continue
        avail_cap, _, label = evaluate_injury_availability_cap(
            inj, target_year, opening_night=opening_night, as_of_date=as_of_date,
        )
        labels[i] = label
        if avail_cap is None:
            continue

        waiver = float(r.get("prior_waiver_fppg") or 20.0)
        if avail_cap <= 0.0:
            adjusted[i] = min(adjusted[i], waiver)
            capped_count += 1
            continue

        # Estimate healthy per-game scoring rate to recompute AdjFPPG under the capped availability
        if math.isfinite(proj_mpg[i]) and math.isfinite(proj_fppm[i]):
            healthy_rate = float(proj_mpg[i]) * float(proj_fppm[i])
        elif r.get("prior_fppg") is not None:
            healthy_rate = float(r["prior_fppg"])
        elif math.isfinite(proj_avail[i]) and proj_avail[i] > 0.15:
            healthy_rate = waiver + (adjusted[i] - waiver) / float(proj_avail[i])
        else:
            healthy_rate = adjusted[i]
        healthy_rate = max(healthy_rate, waiver)
        capped_adj = avail_cap * healthy_rate + (1.0 - avail_cap) * waiver
        if capped_adj < adjusted[i]:
            adjusted[i] = capped_adj
            capped_count += 1

    return adjusted, labels, capped_count


# ==============================================================================
# Season Loading & Feature Builders (Custom Scoring Supported End-to-End)
# ==============================================================================

def parse_draft_info_from_profiles(profiles):
    """Extract draft pick, DOB, height, experience, team_id, position, and names from ESPN player_core."""
    if not profiles:
        return {}
    rd2_sel = [
        s for p in profiles.values()
        if _safe_int(p.get("draft_round")) == 2 and (s := _safe_int(p.get("draft_selection"))) is not None and s > 0
    ]
    selection_is_overall = any(s > 30 for s in rd2_sel)

    parsed = {}
    for aid, p in profiles.items():
        dy = _safe_int(p.get("draft_year"))
        rd = _safe_int(p.get("draft_round"))
        sel = _safe_int(p.get("draft_selection"))
        overall = None
        for col in ("draft_overall", "overall_pick", "draft_pick"):
            v = _safe_int(p.get(col))
            if v is not None and 1 <= v <= 100:
                overall = v
                break
        if overall is None and sel is not None and sel > 0:
            overall = sel if (selection_is_overall or rd is None or rd <= 1) else 30 * (rd - 1) + sel
        if overall is None:
            txt = str(p.get("draft_display_text") or p.get("draft_text") or "")
            m_rp = re.search(r"(?:rd|round)\s*(\d+)[^0-9]+(?:pk|pick)\s*(\d+)", txt, flags=re.I)
            if m_rp:
                rn, pn = int(m_rp.group(1)), int(m_rp.group(2))
                overall = pn if (pn > 30 or rn <= 1) else 30 * (rn - 1) + pn

        dob = _parse_iso_date(p.get("date_of_birth"))
        height = _safe_float(p.get("height"))
        if height is not None and height <= 0:
            height = None
        exp = _safe_int(p.get("experience_years") or p.get("experience"))
        tid = str(p.get("team_id") or p.get("draft_team_id") or "").strip() or None
        pos = (p.get("position_abbreviation") or p.get("position") or "").strip().upper() or None
        names = {
            (p.get(c) or "").strip()
            for c in ("display_name", "full_name", "athlete_display_name", "name")
            if (p.get(c) or "").strip()
        }
        display_name = next(
            ((p.get(c) or "").strip() for c in ("display_name", "full_name", "athlete_display_name", "name") if (p.get(c) or "").strip()),
            aid,
        )
        parsed[aid] = {
            "display_name": display_name,
            "draft_year": dy,
            "overall_pick": overall,
            "dob": dob,
            "height": height,
            "experience": exp,
            "team_id": tid,
            "position": pos,
            "names": names,
        }
    return parsed


def compute_advanced_box_metrics(folder, game_dates):
    """Compute leak-free local impact & usage metrics (BPM, VORP, shrunk On-Off, USG%) from regular-season box scores."""
    team_game_stats = {}
    team_totals = defaultdict(lambda: {"pts": 0.0, "opp_pts": 0.0, "poss": 0.0, "games": 0})
    tbox_path = folder / "team_box_scores.csv"
    if not tbox_path.is_file():
        return {}
    for row in read_csv(tbox_path):
        gid = row.get("game_id")
        if gid not in game_dates:
            continue
        tid = str(row.get("team_id") or "")
        if not tid:
            continue
        fga = _safe_float(row.get("field_goals_attempted")) or 0.0
        fta = _safe_float(row.get("free_throws_attempted")) or 0.0
        tov = _safe_float(row.get("turnovers")) or 0.0
        orb = _safe_float(row.get("offensive_rebounds")) or 0.0
        pts = _safe_float(row.get("team_score")) or 0.0
        opts = _safe_float(row.get("opponent_team_score")) or 0.0
        poss = max(1.0, fga + 0.44 * fta - orb + tov)
        team_game_stats[(gid, tid)] = {
            "fga": fga, "fta": fta, "tov": tov, "margin": pts - opts, "poss": poss,
        }
        ts = team_totals[tid]
        ts["pts"] += pts
        ts["opp_pts"] += opts
        ts["poss"] += poss
        ts["games"] += 1

    raw_entries = []
    team_game_min = defaultdict(float)
    with (folder / "player_box_scores.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            gid = row.get("game_id")
            if gid not in game_dates:
                continue
            aid = str(row.get("athlete_id") or "").strip()
            tid = str(row.get("team_id") or "").strip()
            mins = parse_minutes(row.get("minutes"))
            if not aid or not tid or mins is None or mins <= 0:
                continue
            team_game_min[(gid, tid)] += mins
            raw_entries.append((gid, aid, tid, mins, row))

    p_rows = defaultdict(lambda: {
        "min": 0.0, "pts": 0.0, "fga": 0.0, "tpm": 0.0, "fta": 0.0,
        "orb": 0.0, "drb": 0.0, "ast": 0.0, "stl": 0.0, "blk": 0.0, "tov": 0.0, "pf": 0.0,
        "on_pm": 0.0, "on_pm_min": 0.0, "off_pm": 0.0, "off_pm_min": 0.0,
        "tm_fga": 0.0, "tm_fta": 0.0, "tm_tov": 0.0, "tm_min": 0.0, "tm_poss": 0.0,
        "team_counts": Counter(), "pos_counts": Counter(),
    })
    for gid, aid, tid, mins, row in raw_entries:
        rec = p_rows[aid]
        rec["min"] += mins
        rec["pts"] += _safe_float(row.get("points")) or 0.0
        rec["fga"] += _safe_float(row.get("field_goals_attempted")) or 0.0
        rec["tpm"] += _safe_float(row.get("three_point_field_goals_made")) or 0.0
        rec["fta"] += _safe_float(row.get("free_throws_attempted")) or 0.0
        rec["orb"] += _safe_float(row.get("offensive_rebounds")) or 0.0
        rec["drb"] += _safe_float(row.get("defensive_rebounds")) or 0.0
        rec["ast"] += _safe_float(row.get("assists")) or 0.0
        rec["stl"] += _safe_float(row.get("steals")) or 0.0
        rec["blk"] += _safe_float(row.get("blocks")) or 0.0
        rec["tov"] += _safe_float(row.get("turnovers")) or 0.0
        rec["pf"] += _safe_float(row.get("fouls")) or 0.0
        rec["team_counts"][tid] += 1
        pos = (row.get("athlete_position_abbreviation") or "").strip().upper()
        if pos:
            rec["pos_counts"][pos] += 1
        tgs = team_game_stats.get((gid, tid))
        tgm = max(team_game_min.get((gid, tid), 240.0), 5.0 * mins)
        if tgs is not None:
            rec["tm_fga"] += tgs["fga"]
            rec["tm_fta"] += tgs["fta"]
            rec["tm_tov"] += tgs["tov"]
            rec["tm_min"] += tgm
            rec["tm_poss"] += tgs["poss"] * (mins / (tgm / 5.0))
            pm_raw = (row.get("plus_minus") or "").strip().replace("+", "")
            pm = _safe_float(pm_raw)
            if pm is not None:
                game_len = tgm / 5.0
                rec["on_pm"] += pm
                rec["on_pm_min"] += mins
                rec["off_pm"] += (tgs["margin"] - pm)
                rec["off_pm_min"] += max(0.0, game_len - mins)

    pos_w = {"PG": 1.0, "G": 1.5, "SG": 2.0, "SF": 3.0, "F": 3.5, "PF": 4.0, "C": 5.0}
    raw_bpm_by_id = {}
    team_bpm_num, team_bpm_den = defaultdict(float), defaultdict(float)
    for aid, r in p_rows.items():
        poss = max(r["tm_poss"], 1.0)
        if r["pos_counts"]:
            pos_est = sum(pos_w.get(k, 3.0) * c for k, c in r["pos_counts"].items()) / sum(r["pos_counts"].values())
        else:
            pos_est = 3.0
        pf = (pos_est - 1.0) / 4.0
        s = 100.0 / poss
        raw_bpm = (
            0.860 * (r["pts"] * s)
            + (-0.850 * (1.0 - pf) - 0.720 * pf) * (r["fga"] * s)
            + (-0.336 * (1.0 - pf) - 0.372 * pf) * (r["fta"] * s)
            + 0.389 * (r["tpm"] * s)
            + (0.413 * (1.0 - pf) + 0.181 * pf) * (r["orb"] * s)
            + (0.191 * (1.0 - pf) + 0.181 * pf) * (r["drb"] * s)
            + (1.368 * (1.0 - pf) + 1.008 * pf) * (r["stl"] * s)
            + 0.703 * (r["blk"] * s)
            + (0.580 * (1.0 - pf) + 1.034 * pf) * (r["ast"] * s)
            - 0.964 * (r["tov"] * s)
            - 0.375 * (r["pf"] * s)
            - 4.5
        )
        raw_bpm_by_id[aid] = raw_bpm
        main_tid = r["team_counts"].most_common(1)[0][0]
        team_bpm_num[main_tid] += raw_bpm * r["min"]
        team_bpm_den[main_tid] += r["min"]

    team_adj = {}
    for tid, ts in team_totals.items():
        net100 = 100.0 * (ts["pts"] - ts["opp_pts"]) / max(ts["poss"], 1.0)
        team_adj[tid] = net100 - (team_bpm_num[tid] / max(team_bpm_den[tid], 1.0))

    out = {}
    for aid, r in p_rows.items():
        mins = max(r["min"], 1.0)
        main_tid = r["team_counts"].most_common(1)[0][0]
        on_m, off_m = r["on_pm_min"], r["off_pm_min"]
        if on_m > 0 and off_m > 0:
            eff_m = (on_m * off_m) / (on_m + off_m)
            on_off = (eff_m / (eff_m + 500.0)) * (((r["on_pm"] / on_m) - (r["off_pm"] / off_m)) * 36.0)
        else:
            on_off = 0.0
        shrink = mins / (mins + 250.0)
        bpm = shrink * (raw_bpm_by_id[aid] + team_adj.get(main_tid, 0.0)) + (1.0 - shrink) * (-2.0)
        tm_g = max(team_totals[main_tid]["games"], 1)
        vorp = (bpm + 2.0) * (mins / (tm_g * 240.0)) * (82.0 / tm_g)
        tm_used = max(r["tm_fga"] + 0.44 * r["tm_fta"] + r["tm_tov"], 1.0)
        usg = min(48.0, max(5.0, 100.0 * ((r["fga"] + 0.44 * r["fta"] + r["tov"]) * (r["tm_min"] / 5.0)) / (mins * tm_used)))
        out[aid] = {
            "bpm": round(bpm, 3),
            "vorp": round(vorp, 3),
            "on_off": round(on_off, 3),
            "usg_pct": round(usg, 3),
        }
    return out


def load_season_custom(folder, scoring, prior_births=None):
    """Load one completed regular season under ANY custom scoring dictionary in a single pass."""
    if folder is None or not (folder / "games.csv").is_file() or not (folder / "player_box_scores.csv").is_file():
        return None
    try:
        all_star_date, _ = read_season(folder, split_at_all_star=True, include_box_scores=False)
    except Exception:
        return None

    schedule = list(read_csv(folder / "games.csv"))
    reg_games = [g for g in schedule if regular_season_game(g)]
    if not reg_games:
        return None

    game_dates = {g["game_id"]: g["game_date"] for g in reg_games}
    dates_sorted = sorted(set(game_dates.values()))
    date_to_idx = {d: i for i, d in enumerate(dates_sorted)}

    wins, team_gp, abbr_to_tid = Counter(), Counter(), {}
    for g in reg_games:
        hid, aid = str(g["home_id"]), str(g["away_id"])
        team_gp[hid] += 1
        team_gp[aid] += 1
        if g.get("home_abbreviation"):
            abbr_to_tid[g["home_abbreviation"].strip().upper()] = hid
        if g.get("away_abbreviation"):
            abbr_to_tid[g["away_abbreviation"].strip().upper()] = aid
        hw = str(g.get("home_winner") or "").strip().lower()
        aw = str(g.get("away_winner") or "").strip().lower()
        if hw == "true":
            wins[hid] += 1
        elif aw == "true":
            wins[aid] += 1
        else:
            hs, as_ = _safe_float(g.get("home_score")), _safe_float(g.get("away_score"))
            if hs is not None and as_ is not None:
                if hs > as_:
                    wins[hid] += 1
                elif as_ > hs:
                    wins[aid] += 1

    players = {}
    daily = defaultdict(dict)
    first_seen_team = {}
    team_player_min = defaultdict(lambda: defaultdict(float))
    as_iso = all_star_date.isoformat() if all_star_date is not None else "9999-12-31"

    for gid, aid, name, stats, pos, starter, tid in player_box_scores(folder, game_dates):
        gdate = game_dates[gid]
        pts = score_box_stats(stats, scoring)
        mins = parse_minutes(stats.get("minutes"))
        rec = players.get(aid)
        if rec is None:
            rec = {
                "athlete_id": aid, "names": set(), "display_name": name,
                "games": 0, "total_points": 0.0, "total_minutes": 0.0,
                "tpm": 0.0, "tpa": 0.0,
                "starts": 0, "unknown_starts": 0,
                "pre_games": 0, "pre_points": 0.0, "pre_min": 0.0, "pre_min_games": 0,
                "post_games": 0, "post_points": 0.0, "post_min": 0.0, "post_min_games": 0,
                "pos_counts": Counter(),
            }
            players[aid] = rec
        rec["names"].add(name)
        rec["display_name"] = name
        rec["games"] += 1
        rec["total_points"] += pts
        if mins is not None and mins > 0:
            rec["total_minutes"] += mins
        rec["tpm"] += _safe_float(stats.get("three_point_field_goals_made")) or 0.0
        rec["tpa"] += _safe_float(stats.get("three_point_field_goals_attempted")) or 0.0
        st = str(starter).strip().lower()
        if st == "true":
            rec["starts"] += 1
        elif st != "false":
            rec["unknown_starts"] += 1

        if gdate < as_iso:
            rec["pre_games"] += 1
            rec["pre_points"] += pts
            if mins is not None and mins > 0:
                rec["pre_min"] += mins
                rec["pre_min_games"] += 1
        else:
            rec["post_games"] += 1
            rec["post_points"] += pts
            if mins is not None and mins > 0:
                rec["post_min"] += mins
                rec["post_min_games"] += 1

        if pos:
            rec["pos_counts"][pos.strip().upper()] += 1
        di = date_to_idx[gdate]
        daily[aid][di] = daily[aid].get(di, 0.0) + pts
        if tid:
            ts = str(tid)
            if aid not in first_seen_team or gdate < first_seen_team[aid][0]:
                first_seen_team[aid] = (gdate, ts)
            if mins is not None and mins > 0:
                team_player_min[ts][aid] += mins

    qual_raw = sorted((p["total_points"] / p["games"] for p in players.values() if p["games"] >= 20), reverse=True)
    if not qual_raw:
        qual_raw = sorted((p["total_points"] / p["games"] for p in players.values() if p["games"] > 0), reverse=True)
    repl_fppg = qual_raw[min(149, len(qual_raw) - 1)] if qual_raw else 20.0
    waiver_fppg = WAIVER_FACTOR * repl_fppg

    adj_by_id, raw_by_id = {}, {}
    for aid, p in players.items():
        g = p["games"]
        raw = p["total_points"] / g
        eff_g = min(float(g), REF_GAMES)
        adj = (eff_g * raw + (REF_GAMES - eff_g) * waiver_fppg) / REF_GAMES
        p["fppg"] = raw
        p["adj_fppg"] = adj
        p["mpg"] = (p["total_minutes"] / g) if p["total_minutes"] > 0 else None
        p["fppm"] = (p["total_points"] / p["total_minutes"]) if p["total_minutes"] > 0 else None
        p["tp_pct"] = ((p["tpm"] + 50.0 * 0.355) / (p["tpa"] + 50.0)) if p["tpa"] >= 10 else None
        p["start_pct"] = (100.0 * p["starts"] / g) if p["unknown_starts"] == 0 else None
        p["post_fppg_chg"] = (
            p["post_points"] / p["post_games"] - p["pre_points"] / p["pre_games"]
            if p["pre_games"] >= 5 and p["post_games"] >= 5 else None
        )
        p["post_mpg_chg"] = (
            p["post_min"] / p["post_min_games"] - p["pre_min"] / p["pre_min_games"]
            if p["pre_min_games"] >= 5 and p["post_min_games"] >= 5 else None
        )
        p["primary_pos"] = normalize_pos_tuple(p["pos_counts"].most_common(1)[0][0]) if p["pos_counts"] else ("SG", "SF")
        adj_by_id[aid] = adj
        raw_by_id[aid] = raw

    adv_metrics = compute_advanced_box_metrics(folder, game_dates)
    for aid, p in players.items():
        am = adv_metrics.get(aid)
        if am is not None:
            p.update(am)

    adj_lb = sorted(adj_by_id.values(), reverse=True)
    adj_rank_by_score = {}
    for place, sc in enumerate(adj_lb, start=1):
        adj_rank_by_score.setdefault(sc, place)
    raw_lb = sorted(raw_by_id.values(), reverse=True)
    raw_rank_by_score = {}
    for place, sc in enumerate(raw_lb, start=1):
        raw_rank_by_score.setdefault(sc, place)
    for aid, p in players.items():
        p["adj_rank"] = adj_rank_by_score[p["adj_fppg"]]
        p["raw_rank"] = raw_rank_by_score[p["fppg"]]

    births = prior_births or {}
    team_ctx = {}
    for tid, p_mins in team_player_min.items():
        if sum(p_mins.values()) <= 0:
            continue
        ranked_rot = sorted(p_mins.items(), key=lambda kv: -kv[1])
        top8_ids = {aid for aid, _ in ranked_rot[:8]}
        age_min = top8_age = top8_min = vet30_min = 0.0
        top150_cnt = 0
        for aid, mins in ranked_rot:
            if aid in players and players[aid]["adj_rank"] <= 150 and mins >= 500:
                top150_cnt += 1
            dob = births.get(aid)
            if dob is not None and all_star_date is not None:
                age = (all_star_date - dob).days / 365.2425
                age_min += mins
                if age >= 30.0:
                    vet30_min += mins
                if aid in top8_ids:
                    top8_age += age * mins
                    top8_min += mins
        team_ctx[tid] = {
            "win_pct": wins[tid] / team_gp[tid] if team_gp[tid] > 0 else None,
            "top8_avg_age": top8_age / top8_min if top8_min > 0 else None,
            "vet30_min_share": vet30_min / age_min if age_min > 0 else None,
            "top150_count": top150_cnt,
        }

    return {
        "all_star_date": all_star_date,
        "players": players,
        "waiver_fppg": waiver_fppg,
        "adj_lb": adj_lb,
        "adj_rank_by_score": adj_rank_by_score,
        "raw_rank_by_score": raw_rank_by_score,
        "n_days": len(dates_sorted),
        "daily": daily,
        "first_team": {aid: t for aid, (_, t) in first_seen_team.items()},
        "team_ctx": team_ctx,
        "abbr_to_tid": abbr_to_tid,
    }


def make_vet_feature_row(aid, prior_s, prev2_s, prior_births, target_season, prior_name, next_s=None, roster_turnover=None):
    """Build a returning-veteran feature row from Year 0 & Year -1 box scores + age (0% ADP) + optional pre-Opening-Night roster turnover."""
    pr = prior_s["players"][aid]
    p2 = prev2_s["players"].get(aid) if prev2_s is not None else None
    all_star = prior_s["all_star_date"]
    dob = prior_births.get(aid)
    age = (
        all_star.year - dob.year - ((all_star.month, all_star.day) < (dob.month, dob.day))
        if (dob is not None and all_star is not None) else None
    )

    age_under_24 = max(0.0, 24.0 - age) if age is not None else None
    age_over_30 = max(0.0, age - 30.0) if age is not None else None
    a30_val = float(age_over_30 or 0.0)
    au24_val = float(age_under_24 or 0.0)
    pg_val = float(pr["games"]) if pr.get("games") is not None else REF_GAMES
    pmpg_chg = float(pr.get("post_mpg_chg") or 0.0)
    rt = (roster_turnover.get(aid) or roster_turnover.get(name_key(pr["display_name"]))) if roster_turnover else None

    row = {
        "prior_season": prior_name,
        "draft_season": target_season,
        "athlete_id": aid,
        "player": pr["display_name"],
        "player_type": "returning_vet",
        "positions": pr["primary_pos"],
        "prior_waiver_fppg": prior_s["waiver_fppg"],
        "prior_adj_fppg": pr["adj_fppg"],
        "prior_adj_rank": pr["adj_rank"],
        "prior_raw_rank": pr["raw_rank"],
        "prior_fppg": pr["fppg"],
        "prior_mpg": pr["mpg"],
        "prior_fppm": pr["fppm"],
        "prior_games": pr["games"],
        "prior_start_pct": pr["start_pct"],
        "prior_post_break_fppg_chg": pr["post_fppg_chg"],
        "prior_post_break_mpg_chg": pr["post_mpg_chg"],
        "prior_tp_pct": pr["tp_pct"],
        "prior_bpm": pr.get("bpm"),
        "prior_vorp": pr.get("vorp"),
        "prior_on_off": pr.get("on_off"),
        "prior_usg_pct": pr.get("usg_pct"),
        "age_at_prior_break": age,
        "age_under_24": age_under_24,
        "age_over_30": age_over_30,
        "age30_x_low_g": a30_val * max(0.0, (REF_GAMES - pg_val) / REF_GAMES),
        "young_x_post_mpg": au24_val * max(0.0, pmpg_chg),
        "yoy_fppg_delta": (pr["fppg"] - p2["fppg"]) if p2 else None,
        "yoy_mpg_delta": (pr["mpg"] - p2["mpg"]) if (p2 and pr["mpg"] is not None and p2["mpg"] is not None) else None,
        "two_yr_weighted_fppg": (0.65 * pr["fppg"] + 0.35 * p2["fppg"]) if p2 else pr["fppg"],
        "two_yr_games_avg": (0.5 * (pr["games"] + p2["games"])) if p2 else float(pr["games"]),
        "no_prev2_season": 0.0 if p2 else (1.0 if prev2_s is not None else None),
        "changed_team": rt["changed_team"] if rt is not None else (0.0 if roster_turnover is not None else None),
        "net_vacated_pos_mpg": rt["net_vacated_pos_mpg"] if rt is not None else (0.0 if roster_turnover is not None else None),
        "net_vacated_val": rt["net_vacated_val"] if rt is not None else (0.0 if roster_turnover is not None else None),
        "draft_pick": None, "log_draft_pick": None, "is_top3_pick": None, "is_lottery_pick": None,
        "is_undrafted": None, "is_redshirt": None, "rookie_age": None, "height_inches": None,
        "team_top8_avg_age": None, "team_prior_win_pct": None, "team_vet30_min_share": None, "team_prior_top150_count": None,
        # Optional ADP fields (filled only if ADP is attached)
        "adp": None, "log_adp": None, "mkt_expected_adj_fppg": None,
        "prior_minus_mkt_capped": None, "market_collapse_excess": 0.0,
        "adp_platform_std": None, "espn_vs_avg": None,
    }
    if next_s is not None:
        attach_next_outcome(row, aid, next_s)
    return row


def load_preseason_rosters(data):
    """Load strictly pre-Opening-Night (`last_tx_date < opening_night`) roster snapshots (`data/preseason_rosters.csv`)."""
    path = Path(data) / "preseason_rosters.csv"
    if not path.is_file():
        return None
    by_season = defaultdict(dict)
    for row in read_csv(path):
        s = (row.get("season") or "").strip()
        op = (row.get("opening_night") or "").strip()
        tx = (row.get("last_tx_date") or "").strip()
        if not s:
            continue
        if tx and op and tx >= op:
            continue  # hard zero-leakage guard
        rec = {
            "prior_team": (row.get("prior_team") or "").strip(),
            "oct_team": (row.get("oct_team") or "").strip(),
        }
        aid = (row.get("athlete_id") or "").strip()
        pk = (row.get("player_key") or "").strip()
        if aid:
            by_season[s][aid] = rec
        if pk:
            by_season[s][pk] = rec
    return dict(by_season) if by_season else None


def compute_season_roster_turnover(target_season, prior_s, preseason_rosters):
    """Compute leak-free offseason `changed_team`, `net_vacated_pos_mpg`, and `net_vacated_val` for `target_season`."""
    if not preseason_rosters or target_season not in preseason_rosters:
        return None
    s_rosters = preseason_rosters[target_season]
    waiver = float(prior_s["waiver_fppg"])

    def _pgrp(ptup):
        p0 = ptup[0] if ptup else "SG"
        return "C" if p0 == "C" else ("G" if p0 in ("PG", "SG") else "F")

    prior_rot = {}
    for aid, p in prior_s["players"].items():
        k = name_key(p["display_name"])
        rm = s_rosters.get(aid) or s_rosters.get(k)
        if rm is None:
            continue
        mpg82 = (p["total_minutes"] / 82.0) if p["total_minutes"] > 0 else 0.0
        val_w = max(0.0, float(p["adj_fppg"]) - waiver)
        prior_rot[aid] = {
            "key": k,
            "mpg82": mpg82,
            "val": val_w,
            "pos": _pgrp(p["primary_pos"]),
            "prior_tm": rm["prior_team"],
            "oct_tm": rm["oct_team"],
        }

    vac_pos_mpg = defaultdict(float)
    vac_val = defaultdict(float)
    oct_by_tm = defaultdict(list)
    for aid, info in prior_rot.items():
        ptm, otm = info["prior_tm"], info["oct_tm"]
        if ptm and otm != ptm:
            vac_pos_mpg[(ptm, info["pos"])] += info["mpg82"]
            vac_val[ptm] += info["val"]
        if otm:
            oct_by_tm[otm].append((aid, info))

    out = {}
    for aid, info in prior_rot.items():
        ptm, otm = info["prior_tm"], info["oct_tm"]
        changed = 1.0 if (otm and ptm and otm != ptm) else 0.0
        if otm:
            mates = [m for maid, m in oct_by_tm[otm] if maid != aid]
            add_pos = sum(m["mpg82"] for m in mates if m["prior_tm"] != otm and m["pos"] == info["pos"])
            add_val = sum(m["val"] for m in mates if m["prior_tm"] != otm)
            net_pos = vac_pos_mpg[(otm, info["pos"])] - add_pos
            net_v = vac_val[otm] - add_val
        else:
            net_pos = net_v = 0.0
        rec = {
            "changed_team": changed,
            "net_vacated_pos_mpg": round(net_pos, 3),
            "net_vacated_val": round(net_v, 3),
        }
        out[aid] = rec
        out[info["key"]] = rec
    return out


def load_vegas_preseason_odds(data):
    """Load verified preseason NBA Vegas Win Totals (`data/vegas_preseason_odds.csv`) if present."""
    path = Path(data) / "vegas_preseason_odds.csv"
    if not path.is_file():
        return {}
    aliases = {
        "GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS",
        "UTAH": "UTA", "WSH": "WAS", "BKN": "BRK", "CHA": "CHO", "PHX": "PHO",
    }
    odds = {}
    for row in read_csv(path):
        s = (row.get("season") or "").strip()
        ab = (row.get("espn_abbr") or "").strip().upper()
        wpct = _safe_float(row.get("win_ou_pct"))
        if not s or not ab or wpct is None:
            continue
        odds[(s, ab)] = wpct
        if ab in aliases:
            odds[(s, aliases[ab])] = wpct
    return odds


def make_rookie_feature_row(
    *, aid, player_name, prof, prior_year, prior_name, target_season, prior_s,
    team_id=None, adp_team_abbr=None, pos_override=None, next_s=None, vegas_odds=None,
):
    """Build an incoming-rookie feature row from NBA draft pick, bio, Year-0 team context, and preseason Vegas win%."""
    prof = prof or {}
    tid = team_id or prof.get("team_id")
    if tid is None and adp_team_abbr:
        tid = prior_s["abbr_to_tid"].get(adp_team_abbr.strip().upper())
    tctx = prior_s["team_ctx"].get(str(tid)) if tid is not None else None

    # Resolve team abbreviation for preseason Vegas win total (falls back cleanly to team_prior_win_pct if absent)
    team_abbr = adp_team_abbr.strip().upper() if adp_team_abbr else None
    if tid is not None:
        for src in (next_s, prior_s):
            if src is not None:
                found = next((ab.strip().upper() for ab, t_val in src["abbr_to_tid"].items() if str(t_val) == str(tid)), None)
                if found:
                    team_abbr = found
                    break
    prior_win_pct = round(tctx["win_pct"], 4) if (tctx and tctx["win_pct"] is not None) else None
    vegas_wpct = (vegas_odds or {}).get((target_season, team_abbr)) if team_abbr else None
    eff_vegas_wpct = round(float(vegas_wpct), 4) if vegas_wpct is not None else prior_win_pct

    overall_pick = prof.get("overall_pick")
    eff_pick = float(overall_pick) if overall_pick is not None else 65.0
    dy = prof.get("draft_year")
    dob = prof.get("dob")
    draft_date_ref = date(prior_year + 1, 10, 15)
    rookie_age = (draft_date_ref - dob).days / 365.2425 if dob is not None else prof.get("rookie_age")

    row = {
        "prior_season": prior_name,
        "draft_season": target_season,
        "athlete_id": aid,
        "player": player_name,
        "player_type": "rookie",
        "positions": pos_override or normalize_pos_tuple(prof.get("position")),
        "prior_waiver_fppg": prior_s["waiver_fppg"],
        "prior_adj_fppg": None, "prior_adj_rank": None, "prior_raw_rank": None,
        "prior_fppg": None, "prior_mpg": None, "prior_fppm": None, "prior_games": None,
        "prior_start_pct": None, "prior_post_break_fppg_chg": None, "prior_post_break_mpg_chg": None,
        "prior_tp_pct": None,
        "age_at_prior_break": None, "age_under_24": None, "age_over_30": None,
        "yoy_fppg_delta": None, "yoy_mpg_delta": None, "two_yr_weighted_fppg": None,
        "two_yr_games_avg": None, "no_prev2_season": None,
        "draft_pick": overall_pick,
        "log_draft_pick": math.log(max(eff_pick, 1.0)),
        "is_top3_pick": 1.0 if (overall_pick is not None and overall_pick <= 3) else 0.0,
        "is_lottery_pick": 1.0 if (overall_pick is not None and overall_pick <= 14) else 0.0,
        "is_undrafted": 1.0 if overall_pick is None else 0.0,
        "is_redshirt": (
            float(prof["is_redshirt"]) if prof.get("is_redshirt") is not None
            else ((1.0 if dy < prior_year + 1 else 0.0) if dy is not None else None)
        ),
        "rookie_age": round(rookie_age, 2) if rookie_age is not None else None,
        "height_inches": prof.get("height"),
        "team_top8_avg_age": round(tctx["top8_avg_age"], 2) if (tctx and tctx["top8_avg_age"] is not None) else None,
        "team_prior_win_pct": prior_win_pct,
        "team_vegas_win_pct": eff_vegas_wpct,
        "team_vet30_min_share": round(tctx["vet30_min_share"], 4) if (tctx and tctx["vet30_min_share"] is not None) else None,
        "team_prior_top150_count": tctx["top150_count"] if tctx else None,
        "adp": None, "log_adp": None, "mkt_expected_adj_fppg": None,
        "prior_minus_mkt_capped": None, "market_collapse_excess": 0.0,
        "adp_platform_std": None, "espn_vs_avg": None,
    }
    if next_s is not None:
        attach_next_outcome(row, aid, next_s)
    return row


def attach_adp_fields(row, entry, prior_s, next_s=None):
    """Attach ADP + scale-invariant market-collapse hinge fields onto a candidate row (Stage 2 / evaluation)."""
    adp = entry["adp"]
    mkt_exp = adp_slot_points(prior_s["adj_lb"], adp)
    waiver = float(prior_s["waiver_fppg"])
    collapse_thresh = max(3.5, COLLAPSE_WAIVER_RATIO * waiver)

    row["adp"] = adp
    row["log_adp"] = math.log(max(adp, 1.0))
    row["mkt_expected_adj_fppg"] = mkt_exp
    row["adp_platform_std"] = entry.get("adp_platform_std")
    row["espn_vs_avg"] = entry.get("espn_vs_avg")
    if entry.get("adp_positions"):
        row["positions"] = entry["adp_positions"]

    if row["player_type"] == "returning_vet" and row.get("prior_fppg") is not None:
        raw_gap = float(row["prior_fppg"]) - mkt_exp
        p_rank = row.get("prior_raw_rank") or row.get("prior_adj_rank") or 500
        slot_drop = adp - p_rank
        is_collapse = slot_drop >= COLLAPSE_MIN_SLOT_DROP and raw_gap > collapse_thresh
        row["prior_minus_mkt_capped"] = max(-1.25 * collapse_thresh, min(collapse_thresh, raw_gap))
        row["market_collapse_excess"] = (raw_gap - collapse_thresh) if is_collapse else 0.0
        a30 = float(row.get("age_over_30") or 0.0)
        au24 = float(row.get("age_under_24") or 0.0)
        pg = float(row["prior_games"]) if row.get("prior_games") is not None else REF_GAMES
        pmpg_chg = float(row.get("prior_post_break_mpg_chg") or 0.0)
        row["age30_x_low_g"] = a30 * max(0.0, (REF_GAMES - pg) / REF_GAMES)
        row["young_x_post_mpg"] = au24 * max(0.0, pmpg_chg)
        row["log_adp_x_age30"] = row["log_adp"] * a30
    else:
        row["prior_minus_mkt_capped"] = None
        row["market_collapse_excess"] = 0.0
        row["age30_x_low_g"] = None
        row["young_x_post_mpg"] = None
        row["log_adp_x_age30"] = None

    if next_s is not None and "next_adj_fppg" in row:
        slot_adj = adp_slot_points(next_s["adj_lb"], adp)
        row["fppg_above_adp"] = round(row["next_adj_fppg"] - slot_adj, 3)
        row["beat_adp"] = row["next_rank"] < adp


def attach_next_outcome(row, aid, next_s):
    after = next_s["players"].get(aid) if aid is not None else None
    waiver = next_s["waiver_fppg"]
    if after is not None:
        g, adj, raw, tot = after["games"], after["adj_fppg"], after["fppg"], after["total_points"]
        rk, mpg, fppm = after["adj_rank"], after["mpg"], after["fppm"]
    else:
        g, adj, raw, tot = 0, waiver, 0.0, 0.0
        rk = sum(1 for v in next_s["adj_lb"] if v > waiver) + 1
        mpg = fppm = None
    row.update({
        "next_games": g,
        "next_avail_frac": min(float(g), REF_GAMES) / REF_GAMES,
        "next_raw_fppg": round(raw, 3),
        "next_adj_fppg": round(adj, 3),
        "next_total_fpts": round(tot, 2),
        "next_rank": rk,
        "next_mpg": mpg,
        "next_fppm": fppm,
    })


def to_matrix(rows, features):
    return np.array(
        [[np.nan if row.get(f) is None else float(row[f]) for f in features] for row in rows],
        dtype=float,
    )


def active_features(training, features):
    return [f for f in features if len({row[f] for row in training if row.get(f) is not None}) > 1]


def _make_spline_ridge(feat_list, spline_targets):
    s_cols = [i for i, f in enumerate(feat_list) if f in spline_targets]
    o_cols = [i for i in range(len(feat_list)) if i not in s_cols]
    transformers = []
    if s_cols:
        transformers.append(("splines", make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            SplineTransformer(n_knots=4, degree=2, include_bias=False),
            StandardScaler(),
        ), s_cols))
    if o_cols:
        transformers.append(("linear", make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
        ), o_cols))
    return make_pipeline(ColumnTransformer(transformers), RidgeCV(alphas=RIDGE_ALPHAS))


# ==============================================================================
# Stage 1 Model: 100% ADP-Free Pure Stats (50/50 Huber + Decomposed + Rookies)
# ==============================================================================

def fit_predict_pure_stats(tr_vets, tr_rookies, evaluated_all, prior_waiver, use_advanced_stats=False):
    """Fit 100% ADP-FREE models and return absolute AdjFPPG projections + MPG/FPPM/Avail sub-projections."""
    ev_vet_idx = [i for i, r in enumerate(evaluated_all) if r["player_type"] == "returning_vet"]
    ev_rk_idx = [i for i, r in enumerate(evaluated_all) if r["player_type"] == "rookie"]
    ev_abs_idx = [i for i, r in enumerate(evaluated_all) if r["player_type"] == "returning_absence"]

    ev_vets = [evaluated_all[i] for i in ev_vet_idx]
    ev_rk = [evaluated_all[i] for i in ev_rk_idx]

    n = len(evaluated_all)
    proj_stats = np.full(n, prior_waiver, dtype=float)
    proj_mpg = np.full(n, np.nan, dtype=float)
    proj_fppm = np.full(n, np.nan, dtype=float)
    proj_avail = np.full(n, np.nan, dtype=float)
    vet_coefs, rk_coefs = {}, {}
    spline_set = {"prior_mpg", "prior_fppm", "prior_fppg", "prior_games", "age_at_prior_break"}

    vet_feats_cfg = VET_STATS_FEATURES + (VET_STATS_ADV_FEATURES if use_advanced_stats else ())
    mpg_feats_cfg = VET_STATS_MPG_FEATURES + (VET_STATS_MPG_ADV_FEATURES if use_advanced_stats else ())
    fppm_feats_cfg = VET_STATS_FPPM_FEATURES + (VET_STATS_FPPM_ADV_FEATURES if use_advanced_stats else ())

    if tr_vets and ev_vets:
        y_vet = np.array([float(r["next_adj_fppg"]) for r in tr_vets], dtype=float)
        h_feats = active_features(tr_vets, vet_feats_cfg)
        X_tr_h = to_matrix(tr_vets, h_feats)
        X_ev_h = to_matrix(ev_vets, h_feats)
        h_pipe = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            HuberRegressor(alpha=25.0, epsilon=1.35, max_iter=1000),
        )
        h_pipe.fit(X_tr_h, y_vet)
        ev_h = h_pipe.predict(X_ev_h)
        vet_coefs = dict(zip(h_feats, map(float, h_pipe.named_steps["huberregressor"].coef_)))

        rate_rows = [r for r in tr_vets if r.get("next_mpg") is not None and r.get("next_fppm") is not None]
        mpg_f = active_features(rate_rows, mpg_feats_cfg)
        fppm_f = active_features(rate_rows, fppm_feats_cfg)
        av_f = active_features(tr_vets, VET_STATS_AVAIL_FEATURES)

        X_m, y_m = to_matrix(rate_rows, mpg_f), np.array([float(r["next_mpg"]) for r in rate_rows], dtype=float)
        X_f, y_f = to_matrix(rate_rows, fppm_f), np.array([float(r["next_fppm"]) for r in rate_rows], dtype=float)
        X_a, y_a = to_matrix(tr_vets, av_f), np.array([float(r["next_avail_frac"]) for r in tr_vets], dtype=float)

        groups = np.array([r["draft_season"] for r in tr_vets])
        n_s = len(set(groups))
        oof_m = np.zeros(len(tr_vets), dtype=float)
        oof_f = np.zeros(len(tr_vets), dtype=float)
        oof_a = np.zeros(len(tr_vets), dtype=float)
        if n_s >= 2:
            for tr_i, va_i in GroupKFold(n_splits=min(n_s, 5)).split(tr_vets, y_a, groups):
                tr_sub = [tr_vets[i] for i in tr_i]
                va_sub = [tr_vets[i] for i in va_i]
                tr_r = [r for r in tr_sub if r.get("next_mpg") is not None and r.get("next_fppm") is not None]
                oof_m[va_i] = np.clip(_make_spline_ridge(mpg_f, spline_set).fit(to_matrix(tr_r, mpg_f), [float(r["next_mpg"]) for r in tr_r]).predict(to_matrix(va_sub, mpg_f)), 5.0, 42.0)
                oof_f[va_i] = np.clip(_make_spline_ridge(fppm_f, spline_set).fit(to_matrix(tr_r, fppm_f), [float(r["next_fppm"]) for r in tr_r]).predict(to_matrix(va_sub, fppm_f)), 0.1, 5.0)
                oof_a[va_i] = np.clip(_make_spline_ridge(av_f, spline_set).fit(to_matrix(tr_sub, av_f), [float(r["next_avail_frac"]) for r in tr_sub]).predict(to_matrix(va_sub, av_f)), 0.0, 1.0)
        else:
            oof_m = np.clip(_make_spline_ridge(mpg_f, spline_set).fit(X_m, y_m).predict(to_matrix(tr_vets, mpg_f)), 5.0, 42.0)
            oof_f = np.clip(_make_spline_ridge(fppm_f, spline_set).fit(X_f, y_f).predict(to_matrix(tr_vets, fppm_f)), 0.1, 5.0)
            oof_a = np.clip(_make_spline_ridge(av_f, spline_set).fit(X_a, y_a).predict(X_a), 0.0, 1.0)

        tr_w = np.array([float(r["prior_waiver_fppg"]) for r in tr_vets])
        tr_struct = oof_a * (oof_m * oof_f) + (1.0 - oof_a) * tr_w

        ev_m = np.clip(_make_spline_ridge(mpg_f, spline_set).fit(X_m, y_m).predict(to_matrix(ev_vets, mpg_f)), 5.0, 42.0)
        ev_f = np.clip(_make_spline_ridge(fppm_f, spline_set).fit(X_f, y_f).predict(to_matrix(ev_vets, fppm_f)), 0.1, 5.0)
        ev_a = np.clip(_make_spline_ridge(av_f, spline_set).fit(X_a, y_a).predict(to_matrix(ev_vets, av_f)), 0.0, 1.0)
        ev_w = np.array([float(r["prior_waiver_fppg"]) for r in ev_vets])
        ev_struct = ev_a * (ev_m * ev_f) + (1.0 - ev_a) * ev_w

        tr_pa = np.array([float(r["prior_adj_fppg"]) for r in tr_vets])
        ev_pa = np.array([float(r["prior_adj_fppg"]) for r in ev_vets])
        tr_a30 = np.array([float(r["age_over_30"] or 0.0) for r in tr_vets])
        ev_a30 = np.array([float(r["age_over_30"] or 0.0) for r in ev_vets])
        meta = make_pipeline(StandardScaler(), HuberRegressor(alpha=15.0, max_iter=1000))
        meta.fit(np.column_stack([tr_struct, tr_pa, tr_a30]), y_vet)
        ev_d = meta.predict(np.column_stack([ev_struct, ev_pa, ev_a30]))

        if len(tr_vets) >= 80:
            gb = HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.03,
                max_iter=100,
                max_depth=2,
                min_samples_leaf=35,
                l2_regularization=20.0,
                max_bins=64,
                random_state=42,
            )
            gb.fit(X_tr_h, y_vet)
            ev_g = gb.predict(X_ev_h)
            ev_ens = 0.375 * ev_h + 0.375 * ev_d + 0.25 * ev_g
        else:
            ev_ens = 0.5 * ev_h + 0.5 * ev_d
        for pos, idx in enumerate(ev_vet_idx):
            proj_stats[idx] = ev_ens[pos]
            proj_mpg[idx] = ev_m[pos]
            proj_fppm[idx] = ev_f[pos]
            proj_avail[idx] = ev_a[pos]

    if tr_rookies and ev_rk:
        y_rk = np.array([float(r["next_adj_fppg"]) for r in tr_rookies], dtype=float)
        r_feats = active_features(tr_rookies, ROOKIE_STATS_FEATURES)
        if len(tr_rookies) >= 12 and r_feats:
            r_pipe = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                StandardScaler(),
                HuberRegressor(alpha=30.0, epsilon=1.35, max_iter=1000),
            )
            r_pipe.fit(to_matrix(tr_rookies, r_feats), y_rk)
            ev_r = r_pipe.predict(to_matrix(ev_rk, r_feats))
            rk_coefs = dict(zip(r_feats, map(float, r_pipe.named_steps["huberregressor"].coef_)))
        else:
            ev_r = np.full(len(ev_rk), float(np.median(y_rk)), dtype=float)
        for pos, idx in enumerate(ev_rk_idx):
            proj_stats[idx] = ev_r[pos]

    for idx in ev_abs_idx:
        rec = evaluated_all[idx]
        p2_adj = rec.get("prev2_adj_fppg")
        proj_stats[idx] = (0.78 * float(p2_adj) + 0.22 * prior_waiver) if p2_adj is not None else prior_waiver

    return {
        "pure_stats": proj_stats,
        "proj_mpg": proj_mpg,
        "proj_fppm": proj_fppm,
        "proj_avail": proj_avail,
        "vet_coefs": vet_coefs,
        "rookie_coefs": rk_coefs,
    }


def apply_optional_adp_layer(rows, pure_stats_preds, adp_weight=0.0, apply_collapse_guard=False):
    """Optional Stage-1 Layer-2 blend with ADP + offseason injury-collapse alerts."""
    n = len(rows)
    final_proj = np.array(pure_stats_preds, dtype=float)
    alerts = [""] * n

    for i, r in enumerate(rows):
        mkt = r.get("mkt_expected_adj_fppg")
        adp = r.get("adp")
        if mkt is None or adp is None:
            continue
        p_rank = r.get("prior_raw_rank") or r.get("prior_adj_rank")
        waiver = float(r.get("prior_waiver_fppg") or 20.0)
        gap_thresh = max(3.5, COLLAPSE_WAIVER_RATIO * waiver)
        stats_gap = final_proj[i] - mkt
        raw_gap = (float(r["prior_fppg"]) - mkt) if r.get("prior_fppg") is not None else stats_gap
        eff_gap = max(stats_gap, raw_gap)
        is_collapse = (
            r["player_type"] == "returning_vet"
            and p_rank is not None
            and (adp - p_rank) >= COLLAPSE_MIN_SLOT_DROP
            and eff_gap > gap_thresh
            and stats_gap > (0.15 * waiver)
        )
        if is_collapse:
            alerts[i] = "CHECK_INJURY"
            if apply_collapse_guard:
                final_proj[i] = mkt - 0.50 * (eff_gap - gap_thresh)
                continue
        if adp_weight > 0.0:
            if r["player_type"] == "returning_absence" and r.get("prev2_adj_fppg") is None:
                final_proj[i] = mkt
            else:
                final_proj[i] = (1.0 - adp_weight) * final_proj[i] + adp_weight * mkt
    return final_proj, alerts


# ==============================================================================
# Stage 2 Model: ADP-Aware Residual Champion (`unified_equal`, lam=0.50)
# ==============================================================================

def _effective_residual_lam(adp, lam_early, lam_late):
    """Smoothly ramp residual weight from `lam_early` (ADP <= 100) to `lam_late` (ADP >= 150)."""
    if adp is None or lam_late <= lam_early:
        return float(lam_early)
    t = max(0.0, min(1.0, (float(adp) - LATE_LAM_RAMP_START) / (LATE_LAM_RAMP_END - LATE_LAM_RAMP_START)))
    return float(lam_early) + t * (float(lam_late) - float(lam_early))


def _recency_weights(rows, halflife):
    """Normalized exponential recency weights (mean=1.0) anchored to latest training season in `rows`."""
    if halflife is None or halflife <= 0 or not rows:
        return None
    years = np.array([int(str(r["draft_season"])[:4]) for r in rows], dtype=float)
    ref_year = float(np.max(years))
    w = np.power(0.5, np.maximum(0.0, ref_year - years) / float(halflife))
    return w * (len(w) / max(float(np.sum(w)), 1e-9))


def fit_predict_residual_unified(training_adp_rows, evaluated_all, prior_waiver, lam=0.50, lam_late=None, use_advanced_stats=False, recency_halflife=None):
    """Fit Stage-2 `unified_equal` (Huber + Decomposed + shallow L1 GBDT vets + Rookie Huber) on custom-scored ADP pools."""
    tr_vets = [
        r for r in training_adp_rows
        if r["player_type"] == "returning_vet" and r.get("adp") is not None and r["adp"] <= MAX_RESIDUAL_VET_ADP
    ]
    tr_rk = [
        r for r in training_adp_rows
        if r["player_type"] == "rookie" and r.get("adp") is not None and r["adp"] <= MAX_RESIDUAL_ROOKIE_ADP
    ]
    tr_abs = [
        r for r in training_adp_rows
        if r["player_type"] == "returning_absence" and r.get("adp") is not None and r["adp"] <= MAX_RESIDUAL_VET_ADP
    ]

    ev_vet_idx = [i for i, r in enumerate(evaluated_all) if r["player_type"] == "returning_vet" and r.get("adp") is not None]
    ev_rk_idx = [i for i, r in enumerate(evaluated_all) if r["player_type"] == "rookie" and r.get("adp") is not None]
    ev_abs_idx = [i for i, r in enumerate(evaluated_all) if r["player_type"] == "returning_absence" and r.get("adp") is not None]

    ev_vets = [evaluated_all[i] for i in ev_vet_idx]
    ev_rk = [evaluated_all[i] for i in ev_rk_idx]

    n = len(evaluated_all)
    raw_residuals = np.zeros(n, dtype=float)
    proj_mpg = np.full(n, np.nan, dtype=float)
    proj_fppm = np.full(n, np.nan, dtype=float)
    proj_avail = np.full(n, np.nan, dtype=float)
    vet_coefs, rk_coefs = {}, {}
    spline_set = {"log_adp", "mkt_expected_adj_fppg", "age_at_prior_break"}

    # 1. Returning Veterans: 50/50 huber_clean + decomposed_clean predicting fppg_above_adp
    vet_peer_center = 0.0
    if tr_vets and ev_vets:
        y_vet = np.array([float(r["fppg_above_adp"]) for r in tr_vets], dtype=float)
        w_vet = _recency_weights(tr_vets, recency_halflife)
        h_feats = active_features(tr_vets, VET_RESIDUAL_FEATURES)
        X_tr_h = to_matrix(tr_vets, h_feats)
        X_ev_h = to_matrix(ev_vets, h_feats)
        h_pipe = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            HuberRegressor(alpha=25.0, epsilon=1.35, max_iter=1000),
        )
        if w_vet is not None:
            h_pipe.fit(X_tr_h, y_vet, huberregressor__sample_weight=w_vet)
        else:
            h_pipe.fit(X_tr_h, y_vet)
        ev_h = h_pipe.predict(X_ev_h)
        vet_coefs = dict(zip(h_feats, map(float, h_pipe.named_steps["huberregressor"].coef_)))

        rate_rows = [r for r in tr_vets if r.get("next_mpg") is not None and r.get("next_fppm") is not None]
        w_rate = _recency_weights(rate_rows, recency_halflife)
        mpg_f = active_features(rate_rows, VET_RES_MPG_FEATURES + (VET_RES_MPG_ADV_FEATURES if use_advanced_stats else ()))
        fppm_f = active_features(rate_rows, VET_RES_FPPM_FEATURES + (VET_RES_FPPM_ADV_FEATURES if use_advanced_stats else ()))
        av_f = active_features(tr_vets, VET_RES_AVAIL_FEATURES)

        X_m, y_m = to_matrix(rate_rows, mpg_f), np.array([float(r["next_mpg"]) for r in rate_rows], dtype=float)
        X_f, y_f = to_matrix(rate_rows, fppm_f), np.array([float(r["next_fppm"]) for r in rate_rows], dtype=float)
        X_a, y_a = to_matrix(tr_vets, av_f), np.array([float(r["next_avail_frac"]) for r in tr_vets], dtype=float)

        def _fit_ridge(pipe, X_, y_, sw_):
            return pipe.fit(X_, y_, ridgecv__sample_weight=sw_) if sw_ is not None else pipe.fit(X_, y_)

        groups = np.array([r["draft_season"] for r in tr_vets])
        n_s = len(set(groups))
        oof_m = np.zeros(len(tr_vets), dtype=float)
        oof_f = np.zeros(len(tr_vets), dtype=float)
        oof_a = np.zeros(len(tr_vets), dtype=float)
        if n_s >= 2:
            for tr_i, va_i in GroupKFold(n_splits=min(n_s, 5)).split(tr_vets, y_a, groups):
                tr_sub = [tr_vets[i] for i in tr_i]
                va_sub = [tr_vets[i] for i in va_i]
                tr_r = [r for r in tr_sub if r.get("next_mpg") is not None and r.get("next_fppm") is not None]
                sw_r = _recency_weights(tr_r, recency_halflife)
                sw_a = _recency_weights(tr_sub, recency_halflife)
                oof_m[va_i] = np.clip(_fit_ridge(_make_spline_ridge(mpg_f, spline_set), to_matrix(tr_r, mpg_f), [float(r["next_mpg"]) for r in tr_r], sw_r).predict(to_matrix(va_sub, mpg_f)), 5.0, 42.0)
                oof_f[va_i] = np.clip(_fit_ridge(_make_spline_ridge(fppm_f, spline_set), to_matrix(tr_r, fppm_f), [float(r["next_fppm"]) for r in tr_r], sw_r).predict(to_matrix(va_sub, fppm_f)), 0.1, 5.0)
                oof_a[va_i] = np.clip(_fit_ridge(_make_spline_ridge(av_f, spline_set), to_matrix(tr_sub, av_f), [float(r["next_avail_frac"]) for r in tr_sub], sw_a).predict(to_matrix(va_sub, av_f)), 0.0, 1.0)
        else:
            oof_m = np.clip(_fit_ridge(_make_spline_ridge(mpg_f, spline_set), X_m, y_m, w_rate).predict(to_matrix(tr_vets, mpg_f)), 5.0, 42.0)
            oof_f = np.clip(_fit_ridge(_make_spline_ridge(fppm_f, spline_set), X_f, y_f, w_rate).predict(to_matrix(tr_vets, fppm_f)), 0.1, 5.0)
            oof_a = np.clip(_fit_ridge(_make_spline_ridge(av_f, spline_set), X_a, y_a, w_vet).predict(X_a), 0.0, 1.0)

        tr_w = np.array([float(r["prior_waiver_fppg"]) for r in tr_vets])
        tr_struct = oof_a * (oof_m * oof_f) + (1.0 - oof_a) * tr_w

        ev_m = np.clip(_fit_ridge(_make_spline_ridge(mpg_f, spline_set), X_m, y_m, w_rate).predict(to_matrix(ev_vets, mpg_f)), 5.0, 42.0)
        ev_f = np.clip(_fit_ridge(_make_spline_ridge(fppm_f, spline_set), X_f, y_f, w_rate).predict(to_matrix(ev_vets, fppm_f)), 0.1, 5.0)
        ev_a = np.clip(_fit_ridge(_make_spline_ridge(av_f, spline_set), X_a, y_a, w_vet).predict(to_matrix(ev_vets, av_f)), 0.0, 1.0)
        ev_w = np.array([float(r["prior_waiver_fppg"]) for r in ev_vets])
        ev_struct = ev_a * (ev_m * ev_f) + (1.0 - ev_a) * ev_w

        tr_mkt = np.array([float(r["mkt_expected_adj_fppg"]) for r in tr_vets])
        ev_mkt = np.array([float(r["mkt_expected_adj_fppg"]) for r in ev_vets])
        tr_la = np.array([float(r["log_adp"]) for r in tr_vets])
        ev_la = np.array([float(r["log_adp"]) for r in ev_vets])
        tr_c = np.array([float(r["market_collapse_excess"]) for r in tr_vets])
        ev_c = np.array([float(r["market_collapse_excess"]) for r in ev_vets])

        meta = make_pipeline(StandardScaler(), HuberRegressor(alpha=10.0, max_iter=1000))
        X_meta = np.column_stack([tr_struct - tr_mkt, (tr_struct - tr_mkt) * tr_la, tr_la, tr_c])
        if w_vet is not None:
            meta.fit(X_meta, y_vet, huberregressor__sample_weight=w_vet)
        else:
            meta.fit(X_meta, y_vet)
        ev_d = meta.predict(np.column_stack([ev_struct - ev_mkt, (ev_struct - ev_mkt) * ev_la, ev_la, ev_c]))

        if len(tr_vets) >= 80:
            gb = HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.03,
                max_iter=100,
                max_depth=2,
                min_samples_leaf=35,
                l2_regularization=20.0,
                max_bins=64,
                random_state=42,
            )
            if w_vet is not None:
                gb.fit(X_tr_h, y_vet, sample_weight=w_vet)
            else:
                gb.fit(X_tr_h, y_vet)
            ev_g = gb.predict(X_ev_h)
            ev_eq = 0.375 * ev_h + 0.375 * ev_d + 0.25 * ev_g
        else:
            ev_eq = 0.5 * ev_h + 0.5 * ev_d
        vet_peer_center = float(np.median(ev_eq))
        for pos, idx in enumerate(ev_vet_idx):
            raw_residuals[idx] = ev_eq[pos]
            proj_mpg[idx] = ev_m[pos]
            proj_fppm[idx] = ev_f[pos]
            proj_avail[idx] = ev_a[pos]

    # 2. Incoming Rookies: lean 11-feature rookie_full_huber with zero-leakage Rookie Hype Tax
    rk_flat_bias = 0.0
    if tr_rk and ev_rk:
        y_rk = np.array([float(r["fppg_above_adp"]) for r in tr_rk], dtype=float)
        rk_flat_bias = float(np.median(y_rk))
        tr_work = [r.copy() for r in tr_rk]
        ev_work = [r.copy() for r in ev_rk]
        tr_valid = [r for r in tr_work if r.get("log_draft_pick") is not None]
        if len(tr_valid) >= 8:
            p_mod = HuberRegressor(alpha=5.0, max_iter=500).fit(
                np.array([[float(r["log_draft_pick"])] for r in tr_valid], dtype=float),
                np.array([float(r["next_adj_fppg"]) for r in tr_valid], dtype=float),
            )
            fb = float(np.median([float(r["next_adj_fppg"]) for r in tr_valid]))
            for grp in (tr_work, ev_work):
                for r in grp:
                    pe = float(p_mod.predict([[float(r["log_draft_pick"])]])[0]) if r.get("log_draft_pick") is not None else fb
                    r["adp_minus_pick_expected"] = float(r["mkt_expected_adj_fppg"]) - pe
        else:
            for grp in (tr_work, ev_work):
                for r in grp:
                    r["adp_minus_pick_expected"] = None

        r_feats = active_features(tr_work, ROOKIE_RESIDUAL_FEATURES)
        if len(tr_work) >= 12 and r_feats:
            r_pipe = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                StandardScaler(),
                HuberRegressor(alpha=30.0, epsilon=1.35, max_iter=1000),
            )
            w_rk = _recency_weights(tr_work, recency_halflife)
            if w_rk is not None:
                r_pipe.fit(to_matrix(tr_work, r_feats), y_rk, huberregressor__sample_weight=w_rk)
            else:
                r_pipe.fit(to_matrix(tr_work, r_feats), y_rk)
            ev_r = r_pipe.predict(to_matrix(ev_work, r_feats))
            rk_coefs = dict(zip(r_feats, map(float, r_pipe.named_steps["huberregressor"].coef_)))
        else:
            ev_r = np.full(len(ev_rk), rk_flat_bias, dtype=float)
        for pos, idx in enumerate(ev_rk_idx):
            raw_residuals[idx] = ev_r[pos]

    if len(tr_abs) >= 5:
        abs_med = float(np.median([float(r["fppg_above_adp"]) for r in tr_abs]))
        abs_shrink = len(tr_abs) / (len(tr_abs) + 10.0)
        abs_pred = abs_shrink * abs_med + (1.0 - abs_shrink) * vet_peer_center
    else:
        abs_pred = vet_peer_center
    for idx in ev_abs_idx:
        raw_residuals[idx] = abs_pred

    eff_lam_late = max(float(lam), DEFAULT_WAIT_LAM) if lam_late is None else float(lam_late)
    clip_val = max(4.0, COLLAPSE_WAIVER_RATIO * prior_waiver)
    final_proj = np.full(n, prior_waiver, dtype=float)
    scaled_edges = np.zeros(n, dtype=float)
    for i, r in enumerate(evaluated_all):
        mkt = r.get("mkt_expected_adj_fppg")
        if mkt is not None:
            row_lam = _effective_residual_lam(r.get("adp"), lam, eff_lam_late)
            adj = row_lam * max(-clip_val, min(clip_val, float(raw_residuals[i])))
            scaled_edges[i] = adj
            final_proj[i] = float(mkt) + adj
        else:
            final_proj[i] = float(r.get("prior_adj_fppg") or prior_waiver)

    return {
        "residual_proj": final_proj,
        "scaled_edges": scaled_edges,
        "raw_residuals": raw_residuals,
        "proj_mpg": proj_mpg,
        "proj_fppm": proj_fppm,
        "proj_avail": proj_avail,
        "vet_coefs": vet_coefs,
        "rookie_coefs": rk_coefs,
        "rookie_flat_bias": rk_flat_bias,
    }


# ==============================================================================
# Historical Dataset Builders & ESPN Backtest Simulator
# ==============================================================================

def load_all_completed_seasons(data, scoring):
    seasons_dirs = {int(p.name[:4]): p for p in data.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}", p.name)}
    season_data, births_cache, profiles_cache = {}, {}, {}
    for year in sorted(seasons_dirs):
        try:
            births_cache[year] = load_birth_dates(data, year + 1)
        except Exception:
            births_cache[year] = {}
        try:
            profiles_cache[year] = parse_draft_info_from_profiles(load_player_profiles(data, year + 1))
        except Exception:
            profiles_cache[year] = {}
        s_obj = load_season_custom(seasons_dirs[year], scoring, prior_births=births_cache[year])
        if s_obj is not None:
            season_data[year] = s_obj
    return seasons_dirs, season_data, births_cache, profiles_cache


def build_eval_adp_pool(data, year, seasons_dirs, season_data, births_cache, profiles_cache, played_before_prior, history_available, max_pool_adp=225.0, vegas_odds=None, preseason_rosters=None):
    """Build full-pool ADP candidate rows for year -> year+1 with both pure-stats and ADP-residual features."""
    prior_s = season_data[year]
    next_s = season_data.get(year + 1)
    prev2_s = season_data.get(year - 1)
    prior_name = season_label(year)
    target_name = season_label(year + 1)
    adp_path = seasons_dirs[year + 1] / "adp.csv"
    entries = read_adp_entries(adp_path)
    next_profs = profiles_cache.get(year + 1, {})
    eff_rosters = preseason_rosters if preseason_rosters is not None else load_preseason_rosters(data)
    rt_map = compute_season_roster_turnover(target_name, prior_s, eff_rosters)

    ids_prior = defaultdict(set)
    for aid, p in prior_s["players"].items():
        for n in p["names"]:
            ids_prior[name_key(n)].add(aid)
    ids_prev2 = defaultdict(set)
    if prev2_s is not None:
        for aid, p in prev2_s["players"].items():
            for n in p["names"]:
                ids_prev2[name_key(n)].add(aid)
    ids_next = defaultdict(set)
    if next_s is not None:
        for aid, p in next_s["players"].items():
            for n in p["names"]:
                ids_next[name_key(n)].add(aid)
    ids_prof = defaultdict(set)
    for aid, prof in next_profs.items():
        for n in prof["names"]:
            ids_prof[name_key(n)].add(aid)

    rows, seen_ids = [], set()
    for entry in entries:
        if len(rows) >= int(max_pool_adp) and entry["adp"] > max_pool_adp:
            continue
        key = entry["key"]
        aid = None
        player_type = "returning_absence"

        p_ids = ids_prior[key]
        if len(p_ids) == 1 and next(iter(p_ids)) not in seen_ids:
            (aid,) = p_ids
            player_type = "returning_vet"
        else:
            cand_ids = ids_next.get(key, set())
            if len(cand_ids) == 1 and next(iter(cand_ids)) not in seen_ids:
                (aid,) = cand_ids
            else:
                pc = [c for c in ids_prof.get(key, set()) if c not in seen_ids]
                if len(pc) > 1:
                    rec_c = [c for c in pc if next_profs[c]["draft_year"] in (year + 1, year)]
                    if len(rec_c) == 1:
                        pc = rec_c
                if len(pc) == 1:
                    aid = pc[0]
            if aid is not None and aid not in played_before_prior:
                prof = next_profs.get(aid, {})
                dy, exp = prof.get("draft_year"), prof.get("experience")
                is_rk = True
                if not history_available and ((dy is not None and dy < year - 1) or (exp is not None and exp > 1)):
                    is_rk = False
                elif next_s is not None and aid not in next_s["players"] and dy is not None and dy < year - 1:
                    is_rk = False
                if is_rk:
                    player_type = "rookie"
            elif aid is None and len(ids_prev2.get(key, set())) == 1:
                (aid,) = ids_prev2[key]
                player_type = "returning_absence"

        if aid is not None:
            seen_ids.add(aid)

        if player_type == "returning_vet":
            row = make_vet_feature_row(aid, prior_s, prev2_s, births_cache.get(year, {}), target_name, prior_name, next_s=next_s, roster_turnover=rt_map)
            row["player"] = entry["player"]
        elif player_type == "rookie":
            prof = next_profs.get(aid, {}) if aid is not None else {}
            tid = (next_s["first_team"].get(aid) if (next_s is not None and aid is not None) else None) or prof.get("team_id")
            row = make_rookie_feature_row(
                aid=aid or f"unmatched_{key}",
                player_name=entry["player"],
                prof=prof,
                prior_year=year,
                prior_name=prior_name,
                target_season=target_name,
                prior_s=prior_s,
                team_id=tid,
                adp_team_abbr=entry["adp_team_abbr"],
                pos_override=entry["adp_positions"],
                next_s=next_s,
                vegas_odds=vegas_odds,
            )
        else:
            p2_rec = prev2_s["players"].get(aid) if (prev2_s is not None and aid is not None) else None
            row = {
                "prior_season": prior_name, "draft_season": target_name,
                "athlete_id": aid or f"unmatched_{key}", "player": entry["player"],
                "player_type": "returning_absence",
                "positions": entry["adp_positions"] or (p2_rec["primary_pos"] if p2_rec else ("SG", "SF")),
                "prior_waiver_fppg": prior_s["waiver_fppg"],
                "prior_adj_fppg": None, "prior_adj_rank": None, "prior_raw_rank": None,
                "prior_fppg": None, "prior_mpg": None, "prior_fppm": None, "prior_games": None,
                "prev2_adj_fppg": p2_rec["adj_fppg"] if p2_rec else None,
                "draft_pick": None, "market_collapse_excess": 0.0,
            }
            if next_s is not None:
                attach_next_outcome(row, aid, next_s)

        row["market_rank"] = len(rows) + 1
        attach_adp_fields(row, entry, prior_s, next_s=next_s)
        rows.append(row)
    return rows


def build_all_training_datasets(data, seasons_dirs, season_data, births_cache, profiles_cache, max_pool_adp=225.0):
    """Build BOTH Stage-1 (0% ADP) and Stage-2 (ADP-pool) historical training datasets in one chronological pass."""
    played_before = set()
    archive_through = FIRST_ARCHIVE_YEAR - 1
    history_available = True

    pure_vet_by_s = defaultdict(list)
    pure_rk_by_s = defaultdict(list)
    adp_pool_by_s = {}
    vegas_odds = load_vegas_preseason_odds(data)
    preseason_rosters = load_preseason_rosters(data)

    for year in sorted(season_data):
        if history_available and year >= FIRST_ARCHIVE_YEAR:
            try:
                for ending_year in range(archive_through + 1, year + 2):
                    played_before.update(players_with_appearances(data, ending_year))
                archive_through = year + 1
            except Exception:
                history_available = False
        played_before.update(season_data[year]["players"].keys())

        if (year + 1) not in season_data:
            continue
        prior_s = season_data[year]
        next_s = season_data[year + 1]
        prev2_s = season_data.get(year - 1)
        prior_name = season_label(year)
        target_name = season_label(year + 1)

        # Stage-1 pure-stats veteran rows (0% ADP)
        for aid, p in prior_s["players"].items():
            if p["games"] >= MIN_VET_TRAIN_GAMES and p["adj_rank"] <= MAX_VET_TRAIN_RANK:
                pure_vet_by_s[target_name].append(
                    make_vet_feature_row(aid, prior_s, prev2_s, births_cache.get(year, {}), target_name, prior_name, next_s=next_s)
                )

        # Stage-1 pure-stats rookie rows (0% ADP)
        next_profs = profiles_cache.get(year + 1, {})
        for aid, prof in next_profs.items():
            if aid in played_before:
                continue
            pick = prof.get("overall_pick")
            dy = prof.get("draft_year")
            if pick is None or pick > MAX_ROOKIE_TRAIN_PICK:
                continue
            if dy is not None and dy not in (year + 1, year):
                continue
            tid = next_s["first_team"].get(aid) or prof.get("team_id")
            pure_rk_by_s[target_name].append(
                make_rookie_feature_row(
                    aid=aid, player_name=prof["display_name"], prof=prof,
                    prior_year=year, prior_name=prior_name, target_season=target_name,
                    prior_s=prior_s, team_id=tid, next_s=next_s, vegas_odds=vegas_odds,
                )
            )

        # Stage-2 / Evaluation ADP pool (for seasons with adp.csv)
        if (seasons_dirs[year + 1] / "adp.csv").is_file():
            adp_pool_by_s[target_name] = build_eval_adp_pool(
                data, year, seasons_dirs, season_data, births_cache, profiles_cache,
                set(played_before), history_available, max_pool_adp=max_pool_adp,
                vegas_odds=vegas_odds, preseason_rosters=preseason_rosters,
            )

    return pure_vet_by_s, pure_rk_by_s, adp_pool_by_s, played_before, history_available


def score_roster_espn(roster, bundle):
    """Exact day-by-day ESPN 10-starter (PG,SG,SF,PF,C,G,F,3xUTIL) + 3 Bench scorer under custom scoring."""
    raw_total = sum(bundle["tot_fpts"][i] for i in roster)
    key = tuple(sorted(roster))
    cached = bundle["roster_cache"].get(key)
    if cached is not None:
        return cached

    prio = bundle["prio"]
    ordered = sorted(roster, key=lambda idx: -prio[idx])
    counts = bundle["played_mat"][ordered].sum(axis=0)
    crowded_days = np.flatnonzero(counts >= 5).tolist()
    played_lists, fpts_lists, masks10 = bundle["played_lists"], bundle["fpts_lists"], bundle["masks10"]

    benched_pts = 0.0
    benched_g = 0
    for d in crowded_days:
        free_bits = 0x3FF
        match_slot = None
        filled = 0
        for idx in ordered:
            if not played_lists[idx][d]:
                continue
            if filled == 10:
                benched_g += 1
                benched_pts += fpts_lists[idx][d]
                continue
            m = masks10[idx]
            if match_slot is None:
                avail = m & free_bits
                if avail:
                    free_bits ^= (avail & -avail)
                    filled += 1
                    continue
                match_slot = [-1] * 10
                filled = 0
                for j in ordered:
                    if not played_lists[j][d]:
                        continue
                    if j == idx:
                        break
                    if _dfs_match(j, match_slot, masks10, [0]):
                        filled += 1
            if filled < 10 and _dfs_match(idx, match_slot, masks10, [0]):
                filled += 1
            else:
                benched_g += 1
                benched_pts += fpts_lists[idx][d]

    res = (raw_total - benched_pts, raw_total, benched_g, benched_pts)
    bundle["roster_cache"][key] = res
    return res


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def estimate_survival_prob(player_adp, threshold_adp, sigma, margin=0.0):
    """Probability a player with ``player_adp`` survives K opponent picks past ``threshold_adp``."""
    eff_sigma = max(0.5, float(sigma))
    return _norm_cdf((float(player_adp) - (float(threshold_adp) + float(margin))) / eff_sigma)


def choose_reach_or_wait_pick(
    hero_board,
    vals,
    adps,
    taken,
    roster,
    cur_c,
    cur_m,
    rem_after,
    masks7,
    is_c,
    max_centers,
    picks_until_next,
    surv_sigma,
    round_num,
    surv_margin=1.0,
    min_wait_gain=0.05,
):
    """ADP-aware 2-pick survival optimizer: choose legal pick i maximizing V(i) + E[max_{j!=i, j survives K} V(j)].

    - Enforces ESPN's 4-Center primary cap and 7-starter matching feasibility for both now and next turn.
    - In the final round or back-to-back turns (K == 0), reduces cleanly to greedy top legal value.
    """
    legal = []
    for idx in hero_board:
        if taken[idx] or (is_c[idx] and cur_c >= max_centers):
            continue
        if cur_m + rem_after >= 7:
            legal.append((idx, max_matching_7(roster + [idx], masks7) if cur_m < 7 else 7))
        else:
            cm = max_matching_7(roster + [idx], masks7)
            if cm + rem_after >= 7:
                legal.append((idx, cm))
        if len(legal) >= 35:
            break

    if not legal:
        choice = next(
            (idx for idx in hero_board if not taken[idx] and not (is_c[idx] and cur_c >= max_centers)),
            next(idx for idx in hero_board if not taken[idx]),
        )
        return choice, max_matching_7(roster + [choice], masks7)

    if picks_until_next is None or picks_until_next <= 0 or len(legal) == 1:
        return legal[0]

    avail_pairs = sorted((adps[i], i) for i in range(len(taken)) if not taken[i])
    n_av = len(avail_pairs)
    k = picks_until_next
    if k >= n_av:
        return legal[0]

    eff_sig = max(0.5, float(surv_sigma))
    eff_margin = float(surv_margin)

    best_ev = None
    greedy_ev = None
    best_pair = legal[0]
    for pos_i, (idx_i, new_m_i) in enumerate(legal[:12]):
        val_i = vals[idx_i]
        rem_adps = [a for a, idx in avail_pairs if idx != idx_i]
        if k >= len(rem_adps):
            t_k = rem_adps[-1]
        else:
            t_k = 0.5 * (rem_adps[k - 1] + rem_adps[k])
        next_c_full = (cur_c + (1 if is_c[idx_i] else 0)) >= max_centers

        ev_next = 0.0
        rem_prob = 1.0
        last_val = None
        for idx_j, _ in legal:
            if idx_j == idx_i or (next_c_full and is_c[idx_j]):
                continue
            last_val = vals[idx_j]
            s_j = estimate_survival_prob(adps[idx_j], t_k, eff_sig, margin=eff_margin)
            if s_j <= 1e-6:
                continue
            ev_next += rem_prob * s_j * vals[idx_j]
            rem_prob *= (1.0 - s_j)
            if rem_prob < 1e-5:
                break
        if rem_prob > 1e-6 and last_val is not None:
            ev_next += rem_prob * last_val

        tot_ev = val_i + ev_next
        if pos_i == 0:
            greedy_ev = tot_ev
            best_ev = tot_ev
            best_pair = (idx_i, new_m_i)
        elif tot_ev > max(best_ev + 1e-9, greedy_ev + float(min_wait_gain)):
            best_ev = tot_ev
            best_pair = (idx_i, new_m_i)

    return best_pair


def run_espn_backtest_sim(season_eval_bundles, *, lam=0.50, wait_lam=DEFAULT_WAIT_LAM, n_teams=12, n_rounds=13, trials_per_slot=12, adp_var=2.0, max_centers=4, seed=None):
    """Run 12-team x 13-round ESPN snake draft simulation comparing Stage-1 and Stage-2 strategies."""
    if seed is None:
        seed = random.SystemRandom().randint(1, 999_999)
    rng = random.Random(seed)

    seasons = sorted(season_eval_bundles)
    pick_order = [
        (rnd + 1, team)
        for rnd in range(n_rounds)
        for team in (range(n_teams - 1, -1, -1) if rnd % 2 == 1 else range(n_teams))
    ]
    next_k_by_pick = []
    for p_idx, (_, team) in enumerate(pick_order):
        nxt = next((j for j in range(p_idx + 1, len(pick_order)) if pick_order[j][1] == team), None)
        next_k_by_pick.append((nxt - p_idx - 1) if nxt is not None else None)
    surv_sigma = max(2.5, 0.7 * adp_var)
    surv_margin = max(1.0, 0.4 * adp_var)

    records_split = defaultdict(list)
    records_season = defaultdict(list)
    sample_slot = rng.randrange(n_teams)
    sample_trial = rng.randrange(trials_per_slot)
    sample_season = seasons[-1]
    sample_info = None

    for s in seasons:
        b = season_eval_bundles[s]
        split = b["split"]
        players = b["players"]
        adps = [p["adp"] for p in players]
        straight_order = list(range(len(players)))
        hero_orders = {}
        hero_vals = {}
        for strat in BACKTEST_STRATEGIES:
            if strat == "market":
                hero_orders[strat] = straight_order
                hero_vals[strat] = [p["mkt_expected_adj_fppg"] for p in players]
            else:
                vals = [p[f"{strat}_proj"] for p in players]
                scored = [(vals[idx], -p["market_rank"], idx) for idx, p in enumerate(players)]
                scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
                hero_orders[strat] = [idx for _, _, idx in scored]
                hero_vals[strat] = vals

        for slot in range(n_teams):
            for trial in range(trials_per_slot):
                opp_noisy = []
                for _ in range(n_teams):
                    noisy = [(players[i]["adp"] + rng.uniform(-adp_var, adp_var), i) for i in range(len(players))]
                    noisy.sort(key=lambda t: (t[0], t[1]))
                    opp_noisy.append([i for _, i in noisy])

                for strat in BACKTEST_STRATEGIES:
                    taken = [False] * len(players)
                    rosters = [[] for _ in range(n_teams)]
                    c_cnt = [0] * n_teams
                    m_sz = [0] * n_teams
                    hero_picks = []
                    masks7, is_c = b["masks7"], b["is_center"]

                    for p_idx, (rnd, team) in enumerate(pick_order):
                        pick_num = p_idx + 1
                        roster = rosters[team]
                        rem_after = n_rounds - len(roster) - 1
                        cur_c, cur_m = c_cnt[team], m_sz[team]

                        if team == slot and strat == "residual_wait":
                            choice, new_m = choose_reach_or_wait_pick(
                                hero_orders[strat], hero_vals[strat], adps, taken,
                                roster, cur_c, cur_m, rem_after, masks7, is_c, max_centers,
                                next_k_by_pick[p_idx], surv_sigma, rnd, surv_margin=surv_margin,
                            )
                        else:
                            board = hero_orders[strat] if team == slot else (straight_order if rnd == 1 else opp_noisy[team])
                            choice, new_m = None, cur_m
                            for idx in board:
                                if taken[idx] or (is_c[idx] and cur_c >= max_centers):
                                    continue
                                if cur_m + rem_after >= 7:
                                    choice = idx
                                    new_m = max_matching_7(roster + [idx], masks7) if cur_m < 7 else 7
                                    break
                                cm = max_matching_7(roster + [idx], masks7)
                                if cm + rem_after >= 7:
                                    choice, new_m = idx, cm
                                    break
                            if choice is None:
                                choice = next(
                                    (idx for idx in board if not taken[idx] and not (is_c[idx] and cur_c >= max_centers)),
                                    next(idx for idx in board if not taken[idx]),
                                )
                                new_m = max_matching_7(roster + [choice], masks7)
                        taken[choice] = True
                        roster.append(choice)
                        if is_c[choice]:
                            c_cnt[team] += 1
                        m_sz[team] = new_m
                        if team == slot:
                            hero_picks.append((rnd, pick_num, choice))

                    afs, bgs = [], []
                    for t in range(n_teams):
                        af, _, bg, _ = score_roster_espn(rosters[t], b)
                        afs.append(af)
                        bgs.append(bg)

                    hero_af = afs[slot]
                    field_afs = [afs[t] for t in range(n_teams) if t != slot]
                    field_avg = sum(field_afs) / len(field_afs)
                    better = sum(1 for x in field_afs if x > hero_af)
                    tied = sum(1 for x in field_afs if x == hero_af)
                    rank = 1.0 + better + 0.5 * tied
                    rec = {
                        "hero_af": hero_af, "field_af": field_avg, "diff": hero_af - field_avg,
                        "rank": rank, "won": (1.0 / (1 + tied)) if better == 0 else 0.0,
                        "top4": 1.0 if rank <= 4.0 else 0.0,
                        "beat_field": 1.0 if hero_af > field_avg else (0.5 if hero_af == field_avg else 0.0),
                        "hero_rk": sum(1 for i in rosters[slot] if players[i]["player_type"] == "rookie"),
                        "field_rk": sum(1 for t in range(n_teams) if t != slot for i in rosters[t] if players[i]["player_type"] == "rookie") / (n_teams - 1),
                        "hero_0g": sum(1 for i in rosters[slot] if players[i]["next_games"] == 0),
                        "field_0g": sum(1 for t in range(n_teams) if t != slot for i in rosters[t] if players[i]["next_games"] == 0) / (n_teams - 1),
                        "hero_bg": bgs[slot], "hero_c": c_cnt[slot],
                    }
                    records_split[(split, strat)].append(rec)
                    records_split[("all", strat)].append(rec)
                    records_season[(s, split, strat)].append(rec)
                    if s == sample_season and slot == sample_slot and trial == sample_trial and strat == "residual_unified":
                        sample_info = (s, slot + 1, b, hero_orders[strat], hero_picks, rosters[slot], rec)

    def agg(recs, fields):
        h_avg = mean(r["hero_af"] for r in recs)
        f_avg = mean(r["field_af"] for r in recs)
        d = h_avg - f_avg
        return {
            **fields,
            "drafts": len(recs),
            "hero_espn_fpts": round(h_avg, 1),
            "field_espn_fpts": round(f_avg, 1),
            "espn_diff": round(d, 1),
            "espn_lift_pct": round(100.0 * d / f_avg, 2) if f_avg else None,
            "avg_rank": round(mean(r["rank"] for r in recs), 2),
            "win_pct": round(100.0 * mean(r["won"] for r in recs), 1),
            "top4_pct": round(100.0 * mean(r["top4"] for r in recs), 1),
            "beat_field_pct": round(100.0 * mean(r["beat_field"] for r in recs), 1),
            "hero_rk": round(mean(r["hero_rk"] for r in recs), 2),
            "field_rk": round(mean(r["field_rk"] for r in recs), 2),
            "hero_0g": round(mean(r["hero_0g"] for r in recs), 2),
            "field_0g": round(mean(r["field_0g"] for r in recs), 2),
        }

    sim_summary, sim_seasons = [], []
    for split in ("cv", "holdout", "all"):
        for strat in BACKTEST_STRATEGIES:
            if (split, strat) in records_split:
                sim_summary.append(agg(records_split[(split, strat)], {"split": split, "strategy": strat}))
    for s in seasons:
        sp = season_eval_bundles[s]["split"]
        for strat in BACKTEST_STRATEGIES:
            sim_seasons.append(agg(records_season[(s, sp, strat)], {"season": s, "split": sp, "strategy": strat}))

    print(f"\n================ {n_teams}-TEAM ESPN DRAFT SIMULATOR (Stage 1 Pure Stats vs. Stage 2 Residual lam={lam:.2f} / Wait lam={wait_lam:.2f}) ================")
    print(f"Rules: {n_teams}T x {n_rounds}R | PG,SG,SF,PF,C,G,F,3xUTIL + 3 BE | Max {max_centers} Primary C | R1 straight ADP, R2-13 Uniform[-{adp_var:g},+{adp_var:g}] | Wait(sig={surv_sigma:.1f}, m={surv_margin:.1f})")
    print(f"Trials: {trials_per_slot} x {n_teams} slots = {trials_per_slot * n_teams} drafts/season | Seed: {seed}")

    for split_label, title in (("cv", "CV SEASONS"), ("holdout", "HOLDOUT SEASONS"), ("all", "ALL SEASONS COMBINED")):
        rows = [r for r in sim_summary if r["split"] == split_label]
        if not rows:
            continue
        print(f"\n=== ESPN Simulation — {title} ({rows[0]['drafts']} drafts/strategy) ===")
        print(
            f"  {'Strategy':20s} | {'Hero ESPN':>9s} {'Field ESPN':>10s} {'Diff':>8s} {'Lift%':>6s} | "
            f"{'Rank':>5s} {'Win%':>5s} {'Top4%':>6s} {'BeatFld':>7s} | {'Rk(H/F)':>9s} {'0g(H/F)':>9s}"
        )
        print("  " + "-" * 111)
        for r in rows:
            print(
                f"  {r['strategy']:20s} | "
                f"{r['hero_espn_fpts']:9.1f} {r['field_espn_fpts']:10.1f} {r['espn_diff']:+8.1f} {r['espn_lift_pct']:+5.2f}% | "
                f"{r['avg_rank']:5.2f} {r['win_pct']:4.1f}% {r['top4_pct']:5.1f}% {r['beat_field_pct']:6.1f}% | "
                f"{r['hero_rk']:4.2f}/{r['field_rk']:4.2f} {r['hero_0g']:4.2f}/{r['field_0g']:4.2f}"
            )

    print(f"\n=== Season-by-Season ESPN FPTS Diff (Market vs. Residual Greedy lam={lam:.2f} vs. Residual+Wait lam={wait_lam:.2f}) ===")
    print(f"  {'Season':7s} {'Split':7s} | {'Market':>8s} | {'Residual':>9s} {'Rk':>4s} {'Top4%':>6s} | {'ResidWait':>9s} {'Rk':>4s} {'Win%':>5s} {'Top4%':>6s} {'0g(H/F)':>9s}")
    print("  " + "-" * 96)
    by_sm = {(r["season"], r["strategy"]): r for r in sim_seasons}
    for s in seasons:
        m = by_sm[(s, "market")]
        ru = by_sm[(s, "residual_unified")]
        rw = by_sm[(s, "residual_wait")]
        print(
            f"  {s:7s} {ru['split']:7s} | {m['espn_diff']:+8.1f} | "
            f"{ru['espn_diff']:+9.1f} {ru['avg_rank']:4.2f} {ru['top4_pct']:5.1f}% | "
            f"{rw['espn_diff']:+9.1f} {rw['avg_rank']:4.2f} {rw['win_pct']:4.1f}% {rw['top4_pct']:5.1f}% {rw['hero_0g']:4.2f}/{rw['field_0g']:4.2f}"
        )

    if sample_info:
        s, slot_num, b, hero_order, picks, roster, rec = sample_info
        players = b["players"]
        rank_of = {idx: r for r, idx in enumerate(hero_order, start=1)}
        slot_of = assign_lineup_slots(roster, b["masks10"], b["prio"])
        tag_map = {"returning_vet": "VET", "rookie": "ROOK", "returning_absence": "RET"}
        print(f"\n=== Sample {s} Roster — Hero (residual_unified, lam={lam:.2f}) at Random Slot #{slot_num}/{n_teams} ===")
        print(
            f"  ESPN Active FPTS: {rec['hero_af']:,.1f} vs Field Avg {rec['field_af']:,.1f} ({rec['diff']:+,.1f}) | "
            f"Finish: #{rec['rank']:.1f}/{n_teams} | Primary C: {rec['hero_c']}/{max_centers} | 0g Busts: {rec['hero_0g']}"
        )
        print(f"  {'Rd':>2s} {'Pk':>3s} {'Slot':>4s} | {'Player':23s} {'Type':>4s} {'Pos':>8s} | {'ADP':>5s} {'HeroRk':>6s} | {'Gm':>3s} {'FPPG':>5s} {'Total':>7s}")
        print("  " + "-" * 86)
        for rnd, pick_num, idx in picks:
            p = players[idx]
            print(
                f"  {rnd:2d} {pick_num:3d} {slot_of[idx]:>4s} | {p['player'][:23]:23s} {tag_map[p['player_type']]:>4s} {p['pos_str'][:8]:>8s} | "
                f"{p['adp']:5.1f} {rank_of[idx]:6d} | {p['next_games']:3d} {p['next_raw_fppg']:5.1f} {p['next_total_fpts']:7,.0f}"
            )

    return sim_summary, sim_seasons


def run_backtest(data, scoring_name, scoring, lam=0.50, wait_lam=DEFAULT_WAIT_LAM, holdout_seasons=2, cv_folds=6, max_pool_adp=225.0, skip_sim=False, trials_per_slot=12, adp_var=2.0, seed=None, use_advanced_stats=False, min_train_season="2016-17", recency_halflife=None, report_suffix="", n_teams=12):
    seasons_dirs, season_data, births_cache, profiles_cache = load_all_completed_seasons(data, scoring)
    pure_vet_by_s, pure_rk_by_s, adp_pool_by_s, _, _ = build_all_training_datasets(
        data, seasons_dirs, season_data, births_cache, profiles_cache, max_pool_adp=max_pool_adp,
    )

    eff_min_s = None if (not min_train_season or str(min_train_season).lower() == "all") else str(min_train_season)
    train_adp_seasons = sorted(s for s in adp_pool_by_s if eff_min_s is None or s >= eff_min_s)
    # Full 12-team evaluation seasons require >=156 ADP candidates (2014-15+ multi-platform FantasyPros)
    eval_seasons = [s for s in train_adp_seasons if len(adp_pool_by_s[s]) >= 156]
    if len(eval_seasons) <= holdout_seasons:
        eval_seasons = train_adp_seasons
    if len(eval_seasons) <= holdout_seasons:
        raise ValueError(f"need more completed draft seasons than --holdout-seasons ({len(eval_seasons)} found)")

    development, holdout = eval_seasons[:-holdout_seasons], eval_seasons[-holdout_seasons:]
    # Require at least 2 prior training seasons before the first evaluated CV fold
    cv_eligible = [s for s in development if sum(1 for ts in train_adp_seasons if ts < s) >= 2]
    cv_seasons = cv_eligible[-cv_folds:] if cv_eligible else development[1:]
    scored_seasons = sorted(set(cv_seasons) | set(holdout))

    board_rows_by_split = defaultdict(list)
    sim_bundles = {}
    final_meta = {}

    for target_s in scored_seasons:
        target_year = int(target_s[:4])
        prior_year = target_year - 1
        split = "holdout" if target_s in holdout else "cv"
        cutoff = development[-1] if split == "holdout" else target_s

        # Strictly earlier completed seasons for training (zero future leakage)
        tr_pure_vets = [r for s, grp in pure_vet_by_s.items() if (eff_min_s is None or s >= eff_min_s) and (s <= cutoff if split == "holdout" else s < cutoff) for r in grp]
        tr_pure_rks = [r for s, grp in pure_rk_by_s.items() if (eff_min_s is None or s >= eff_min_s) and (s <= cutoff if split == "holdout" else s < cutoff) for r in grp]
        tr_adp_rows = [r for s, grp in adp_pool_by_s.items() if (eff_min_s is None or s >= eff_min_s) and (s <= cutoff if split == "holdout" else s < cutoff) for r in grp]

        cands = [r.copy() for r in adp_pool_by_s[target_s]]
        prior_waiver = season_data[prior_year]["waiver_fppg"]

        # Stage 1: Pure Stats + Optional News/Blend
        res_pure = fit_predict_pure_stats(tr_pure_vets, tr_pure_rks, cands, prior_waiver, use_advanced_stats=use_advanced_stats)
        pure_preds = res_pure["pure_stats"]
        news_cut_preds, _ = apply_optional_adp_layer(cands, pure_preds, adp_weight=0.0, apply_collapse_guard=True)
        blend50_preds, _ = apply_optional_adp_layer(cands, pure_preds, adp_weight=0.50, apply_collapse_guard=True)

        # Stage 2: ADP-Aware Residual Champion (`unified_equal`, lam) + Reach-vs-Wait (`wait_lam`)
        res_resid = fit_predict_residual_unified(tr_adp_rows, cands, prior_waiver, lam=lam, use_advanced_stats=use_advanced_stats, recency_halflife=recency_halflife)
        resid_preds = res_resid["residual_proj"]
        clip_val = max(4.0, COLLAPSE_WAIVER_RATIO * prior_waiver)
        raw_res = res_resid["raw_residuals"]

        for i, row in enumerate(cands):
            row["split"] = split
            row["pure_stats_proj"] = round(float(pure_preds[i]), 3)
            row["pure_stats_news_cut_proj"] = round(float(news_cut_preds[i]), 3)
            row["blend_adp_50_proj"] = round(float(blend50_preds[i]), 3)
            row["residual_unified_proj"] = round(float(resid_preds[i]), 3)
            mkt_i = row.get("mkt_expected_adj_fppg")
            wait_p = (
                float(mkt_i) + wait_lam * max(-clip_val, min(clip_val, float(raw_res[i])))
                if mkt_i is not None else float(resid_preds[i])
            )
            row["residual_wait_proj"] = float(wait_p)

        board_rows_by_split[split].extend(cands)
        if split == "holdout":
            final_meta = {
                "n_pure_vets": len(tr_pure_vets), "n_pure_rks": len(tr_pure_rks),
                "n_adp_rows": len(tr_adp_rows), "res_vet_coefs": res_resid["vet_coefs"],
                "rk_bias": res_resid["rookie_flat_bias"],
            }

        next_s = season_data[target_year]
        n_p, n_days = len(cands), next_s["n_days"]
        played_mat = np.zeros((n_p, max(n_days, 1)), dtype=np.uint8)
        fpts_mat = np.zeros((n_p, max(n_days, 1)), dtype=np.float32)
        masks7, masks10, is_center, prio, tot_fpts = [], [], [], [], []
        for idx, p in enumerate(cands):
            ptup = p["positions"]
            p["pos_str"] = ",".join(ptup)
            m7, m10 = _pos_to_masks(ptup)
            masks7.append(m7)
            masks10.append(m10)
            is_center.append(ptup[0] == "C")
            prio.append(p["next_raw_fppg"] if p["next_games"] > 0 else -1.0)
            day_map = next_s["daily"].get(p["athlete_id"], {})
            for d, pts in day_map.items():
                played_mat[idx, d] = 1
                fpts_mat[idx, d] = pts
            tot_fpts.append(float(sum(day_map.values())))

        if len(cands) >= n_teams * 13:
            sim_bundles[target_s] = {
                "split": split, "players": cands,
                "played_mat": played_mat, "played_lists": played_mat.tolist(),
                "fpts_lists": fpts_mat.tolist(), "masks7": masks7, "masks10": masks10,
                "is_center": is_center, "prio": prio, "tot_fpts": tot_fpts, "roster_cache": {},
            }

    print(f"================ UNIFIED DRAFT ENGINE BACKTEST (Scoring: {scoring_name}, Residual lam={lam:.2f}) ================")
    print(f"Development drafts: {', '.join(development)} | Holdout drafts: {', '.join(holdout)}")
    if final_meta:
        cap_w = final_meta["res_vet_coefs"].get("prior_minus_mkt_capped")
        col_w = final_meta["res_vet_coefs"].get("market_collapse_excess")
        hinge = f" | Injury hinge capped={cap_w:+.2f}, collapse={col_w:+.2f}" if (cap_w is not None and col_w is not None) else ""
        print(f"Stage-1 train: {final_meta['n_pure_vets']} vets + {final_meta['n_pure_rks']} rookies | Stage-2 rookie bias: {final_meta['rk_bias']:+.2f}{hinge}")

    board_summary = []
    for split_label in ("cv", "holdout"):
        rows = board_rows_by_split.get(split_label, [])
        if not rows:
            continue
        by_s = defaultdict(list)
        for r in rows:
            by_s[r["draft_season"]].append(r)
        s_list = sorted(by_s)
        print(f"\n=== GLOBAL Top-150 Board — {split_label.upper()} ({len(s_list)} seasons, Scoring: {scoring_name}) ===")
        for strat in BACKTEST_STRATEGIES:
            items, added, dropped = [], [], []
            for s in s_list:
                grp = by_s[s]
                ordered = (
                    sorted(grp, key=lambda r: r["market_rank"]) if strat == "market"
                    else sorted(grp, key=lambda r: (-r[f"{strat}_proj"], r["market_rank"]))
                )
                top150, out = ordered[:BOARD_SIZE], ordered[BOARD_SIZE:]
                for rk, it in enumerate(top150, start=1):
                    items.append((rk, it))
                    if it["market_rank"] > BOARD_SIZE:
                        added.append(it)
                for it in out:
                    if it["market_rank"] <= BOARD_SIZE:
                        dropped.append(it)
            cap = sum(1 for _, it in items if it["next_rank"] <= BOARD_SIZE)
            avg_adj = mean(it["next_adj_fppg"] for _, it in items)
            mae = mean(abs(rk - it["next_rank"]) for rk, it in items)
            z0 = sum(1 for _, it in items if it["next_games"] == 0)
            rk_n = sum(1 for _, it in items if it["player_type"] == "rookie")
            proj_mae = (
                mean(abs(it[f"{strat}_proj"] - it["next_adj_fppg"]) for _, it in items) if strat != "market"
                else mean(abs(it["mkt_expected_adj_fppg"] - it["next_adj_fppg"]) for _, it in items)
            )
            swap_s = (
                f"Swaps({len(added):2d}): {mean(x['next_adj_fppg'] for x in added) - mean(x['next_adj_fppg'] for x in dropped):+.2f} FPPG"
                if added and dropped else "Swaps( 0): baseline"
            )
            print(
                f"  {strat:20s} | Top150: {cap:3d}/{len(items)} ({100*cap/len(items):5.2f}%) | "
                f"AvgAdj: {avg_adj:5.2f} | ProjMAE: {proj_mae:4.2f} | RankMAE: {mae:5.2f} | Rk:{rk_n:2d} 0g:{z0:2d} | {swap_s}"
            )
            board_summary.append({
                "split": split_label, "strategy": strat, "seasons": len(s_list),
                "top150_captured": cap, "board_slots": len(items), "avg_adj_fppg": round(avg_adj, 3),
                "proj_mae_adj_fppg": round(proj_mae, 3), "rank_mae": round(mae, 2),
                "rookies_on_board": rk_n, "zero_game_busts": z0,
            })

    tag = f"{scoring_name}_{report_suffix}" if report_suffix else scoring_name
    write_csv(data / f"draft_engine_board_summary_{tag}.csv", board_summary)

    if not skip_sim and sim_bundles:
        sim_summary, sim_seasons = run_espn_backtest_sim(
            sim_bundles, lam=lam, wait_lam=wait_lam, n_teams=n_teams, trials_per_slot=trials_per_slot, adp_var=adp_var, seed=seed,
        )
        write_csv(data / f"draft_engine_sim_summary_{tag}.csv", sim_summary)
        write_csv(data / f"draft_engine_sim_by_season_{tag}.csv", sim_seasons)

    print(f"\nSaved backtest reports under {data}/draft_engine_*_{tag}.csv")


# ==============================================================================
# Live Draft Board Generator (Stage 1 Pre-ADP & Stage 2 Residual + Live Injuries)
# ==============================================================================

def load_optional_rookies_csv(path, prior_year, prior_s, target_season):
    """Score a new draft class before ESPN publishes player_core (cols: player,draft_pick,team,age,pos,is_redshirt)."""
    rows = []
    if not path or not path.is_file():
        return rows
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("player") or row.get("name") or "").strip()
            if not name:
                continue
            prof = {
                "overall_pick": _safe_int(row.get("draft_pick") or row.get("pick")),
                "rookie_age": _safe_float(row.get("age") or row.get("rookie_age")),
                "height": _safe_float(row.get("height") or row.get("height_inches")),
                "is_redshirt": _safe_float(row.get("is_redshirt")),
                "position": row.get("pos") or row.get("position"),
            }
            rows.append(
                make_rookie_feature_row(
                    aid=f"rookie_csv_{name_key(name)}", player_name=name, prof=prof,
                    prior_year=prior_year, prior_name=season_label(prior_year),
                    target_season=target_season, prior_s=prior_s,
                    adp_team_abbr=row.get("team") or row.get("team_abbr"),
                    pos_override=normalize_pos_tuple(row.get("pos") or row.get("position")) if (row.get("pos") or row.get("position")) else None,
                )
            )
    return rows


def generate_live_board(
    data,
    target_season,
    scoring_name,
    scoring,
    mode="auto",
    lam=0.50,
    adp_weight=0.0,
    collapse_guard=False,
    use_live_injuries=False,
    injuries_csv=None,
    opening_night_str=None,
    as_of_date_str=None,
    exclude_players="",
    rookies_csv=None,
    board_limit=225,
    use_advanced_stats=False,
    min_train_season="2016-17",
    recency_halflife=None,
):
    """Generate a ready-to-use ranked draft board in either Stage-1 (`pure_stats`) or Stage-2 (`residual`) mode."""
    if not re.fullmatch(r"\d{4}-\d{2}", target_season):
        raise ValueError(f"--generate-board expects YYYY-YY format (got {target_season!r})")
    target_year = int(target_season[:4])
    prior_year = target_year - 1

    seasons_dirs, season_data, births_cache, profiles_cache = load_all_completed_seasons(data, scoring)
    if prior_year not in season_data:
        raise ValueError(f"missing completed prior season {season_label(prior_year)} in {data}")

    pure_vet_by_s, pure_rk_by_s, adp_pool_by_s, played_before, history_available = build_all_training_datasets(
        data, seasons_dirs, season_data, births_cache, profiles_cache,
    )
    eff_min_s = None if (not min_train_season or str(min_train_season).lower() == "all") else str(min_train_season)
    prior_seasons_pure = sorted(s for s in pure_vet_by_s if s < target_season and (eff_min_s is None or s >= eff_min_s))
    prior_seasons_adp = sorted(s for s in adp_pool_by_s if s < target_season and (eff_min_s is None or s >= eff_min_s))
    if not prior_seasons_pure:
        raise ValueError(f"no completed training seasons before {target_season}")

    adp_path = data / target_season / "adp.csv"
    adp_entries = read_adp_entries(adp_path)
    adp_by_key = {e["key"]: e for e in adp_entries}
    has_adp = bool(adp_entries)

    if mode == "auto":
        eff_mode = "residual" if (has_adp and prior_seasons_adp) else "pure_stats"
    else:
        eff_mode = mode
    if eff_mode == "residual" and not has_adp:
        raise ValueError(f"--mode residual requires {adp_path}. Use --mode pure_stats before ADP is released.")
    if eff_mode == "residual" and not prior_seasons_adp:
        raise ValueError(f"--mode residual needs at least 1 completed prior season with adp.csv before {target_season}")

    prior_s = season_data[prior_year]
    prev2_s = season_data.get(prior_year - 1)
    prior_name = season_label(prior_year)
    excluded_keys = {name_key(p) for p in exclude_players.split(",") if p.strip()} if exclude_players else set()

    # Load optional injury records (live ESPN API and/or local CSV)
    injury_records = []
    injury_source_notes = []
    if use_live_injuries:
        cache_file = data / target_season / "espn_injuries_cache.json"
        live_recs, src_tag = fetch_espn_live_injuries(cache_path=cache_file)
        injury_records.extend(live_recs)
        injury_source_notes.append(f"{src_tag} ({len(live_recs)} active entries)")
    if injuries_csv:
        csv_recs = load_injuries_csv(Path(injuries_csv))
        injury_records.extend(csv_recs)
        injury_source_notes.append(f"{Path(injuries_csv).name} ({len(csv_recs)} rows)")
    opening_night = _parse_iso_date(opening_night_str) if opening_night_str else date(target_year, 10, 21)
    if opening_night_str and opening_night is None:
        raise ValueError(f"invalid --opening-night date {opening_night_str!r} (expected YYYY-MM-DD)")
    as_of_date = _parse_iso_date(as_of_date_str) if as_of_date_str else None
    if as_of_date_str and as_of_date is None:
        raise ValueError(f"invalid --as-of-date {as_of_date_str!r} (expected YYYY-MM-DD)")

    # Load target-season profiles if available (and fallback prior profiles to rescue returning_absence vets)
    try:
        target_profs = parse_draft_info_from_profiles(load_player_profiles(data, target_year + 1))
    except Exception:
        target_profs = {}
    prior_profs = profiles_cache.get(prior_year, {})
    vegas_odds = load_vegas_preseason_odds(data)
    rt_live = compute_season_roster_turnover(target_season, prior_s, load_preseason_rosters(data))

    # Build candidates
    cands = []
    seen_ids, seen_keys = set(), set()

    if eff_mode == "residual":
        # Anchor candidate universe to target season's adp.csv (full pool: vets + rookies + returning absences)
        ids_prior = defaultdict(set)
        for aid, p in prior_s["players"].items():
            for n in p["names"]:
                ids_prior[name_key(n)].add(aid)
        ids_prev2 = defaultdict(set)
        if prev2_s is not None:
            for aid, p in prev2_s["players"].items():
                for n in p["names"]:
                    ids_prev2[name_key(n)].add(aid)
        ids_tprof = defaultdict(set)
        for aid, prof in target_profs.items():
            for n in prof["names"]:
                ids_tprof[name_key(n)].add(aid)
        ids_pprof = defaultdict(set)
        for aid, prof in prior_profs.items():
            for n in prof["names"]:
                ids_pprof[name_key(n)].add(aid)

        csv_rk_by_key = {
            name_key(r["player"]): r
            for r in load_optional_rookies_csv(Path(rookies_csv), prior_year, prior_s, target_season)
        } if rookies_csv else {}

        for entry in adp_entries:
            if len(cands) >= board_limit and entry["adp"] > board_limit:
                continue
            key = entry["key"]
            if key in excluded_keys:
                continue
            aid = None
            ptype = "returning_absence"
            p_ids = ids_prior[key]
            if len(p_ids) == 1 and next(iter(p_ids)) not in seen_ids:
                (aid,) = p_ids
                ptype = "returning_vet"
            else:
                pc = [c for c in ids_tprof.get(key, set()) if c not in seen_ids]
                if len(pc) > 1:
                    rec_c = [c for c in pc if target_profs[c]["draft_year"] in (target_year, prior_year)]
                    if len(rec_c) == 1:
                        pc = rec_c
                if len(pc) == 1:
                    aid = pc[0]
                    if aid not in played_before:
                        ptype = "rookie"
                elif key in csv_rk_by_key:
                    ptype = "rookie"
                else:
                    # Check prior profiles / Year -1 to distinguish injured returning stars from incoming rookies
                    prev_c = [c for c in ids_pprof.get(key, set()) if c in played_before]
                    if len(prev_c) == 1:
                        aid = prev_c[0]
                        ptype = "returning_absence"
                    elif len(ids_prev2.get(key, set())) == 1:
                        (aid,) = ids_prev2[key]
                        ptype = "returning_absence"
                    else:
                        ptype = "rookie"

            if aid is not None:
                seen_ids.add(aid)
            seen_keys.add(key)

            if ptype == "returning_vet":
                row = make_vet_feature_row(aid, prior_s, prev2_s, births_cache.get(prior_year, {}), target_season, prior_name, roster_turnover=rt_live)
                row["player"] = entry["player"]
            elif ptype == "rookie":
                if key in csv_rk_by_key:
                    row = csv_rk_by_key[key].copy()
                    row["player"] = entry["player"]
                else:
                    prof = target_profs.get(aid, {}) if aid is not None else {}
                    row = make_rookie_feature_row(
                        aid=aid or f"unmatched_{key}", player_name=entry["player"], prof=prof,
                        prior_year=prior_year, prior_name=prior_name, target_season=target_season,
                        prior_s=prior_s, adp_team_abbr=entry["adp_team_abbr"], pos_override=entry["adp_positions"],
                        vegas_odds=vegas_odds,
                    )
            else:
                p2_rec = prev2_s["players"].get(aid) if (prev2_s is not None and aid is not None) else None
                row = {
                    "prior_season": prior_name, "draft_season": target_season,
                    "athlete_id": aid or f"unmatched_{key}", "player": entry["player"],
                    "player_type": "returning_absence",
                    "positions": entry["adp_positions"] or (p2_rec["primary_pos"] if p2_rec else ("SG", "SF")),
                    "prior_waiver_fppg": prior_s["waiver_fppg"],
                    "prior_adj_fppg": None, "prior_adj_rank": None, "prior_raw_rank": None,
                    "prior_fppg": None, "prior_mpg": None, "prior_fppm": None, "prior_games": 0,
                    "prev2_adj_fppg": p2_rec["adj_fppg"] if p2_rec else None,
                    "draft_pick": None, "market_collapse_excess": 0.0,
                }
            row["market_rank"] = len(cands) + 1
            attach_adp_fields(row, entry, prior_s)
            cands.append(row)
    else:
        # Stage 1 (`pure_stats`): build complete candidate pool WITHOUT requiring adp.csv!
        for aid, p in sorted(prior_s["players"].items(), key=lambda kv: kv[1]["adj_rank"]):
            key = name_key(p["display_name"])
            if key in excluded_keys or (p["adj_rank"] > 280 and key not in adp_by_key):
                continue
            cands.append(make_vet_feature_row(aid, prior_s, prev2_s, births_cache.get(prior_year, {}), target_season, prior_name))
            seen_ids.add(aid)
            seen_keys.add(key)

        if prev2_s is not None:
            for aid, p2 in sorted(prev2_s["players"].items(), key=lambda kv: kv[1]["adj_rank"]):
                if aid in seen_ids or p2["adj_rank"] > 120:
                    continue
                key = name_key(p2["display_name"])
                if key in excluded_keys or key in seen_keys:
                    continue
                if has_adp and key not in adp_by_key:
                    continue
                if not has_adp and p2["adj_rank"] > 60:
                    continue
                cands.append({
                    "prior_season": prior_name, "draft_season": target_season,
                    "athlete_id": aid, "player": p2["display_name"],
                    "player_type": "returning_absence", "positions": p2["primary_pos"],
                    "prior_waiver_fppg": prior_s["waiver_fppg"],
                    "prior_adj_fppg": None, "prior_adj_rank": None, "prior_raw_rank": None,
                    "prior_fppg": None, "prior_mpg": None, "prior_fppm": None, "prior_games": 0,
                    "prev2_adj_fppg": p2["adj_fppg"], "draft_pick": None, "market_collapse_excess": 0.0,
                })
                seen_ids.add(aid)
                seen_keys.add(key)

        if rookies_csv:
            for r_row in load_optional_rookies_csv(Path(rookies_csv), prior_year, prior_s, target_season):
                k = name_key(r_row["player"])
                if k not in excluded_keys and k not in seen_keys:
                    cands.append(r_row)
                    seen_keys.add(k)

        for aid, prof in target_profs.items():
            if aid in seen_ids or aid in played_before:
                continue
            k = name_key(prof["display_name"])
            if k in excluded_keys or k in seen_keys:
                continue
            pick, dy = prof.get("overall_pick"), prof.get("draft_year")
            if (pick is not None and pick <= 45 and (dy is None or dy in (target_year, prior_year))) or (k in adp_by_key):
                cands.append(make_rookie_feature_row(
                    aid=aid, player_name=prof["display_name"], prof=prof,
                    prior_year=prior_year, prior_name=prior_name, target_season=target_season,
                    prior_s=prior_s,
                    adp_team_abbr=adp_by_key[k]["adp_team_abbr"] if k in adp_by_key else None,
                    pos_override=adp_by_key[k]["adp_positions"] if k in adp_by_key else None,
                    vegas_odds=vegas_odds,
                ))
                seen_ids.add(aid)
                seen_keys.add(k)

        for r in cands:
            e = adp_by_key.get(name_key(r["player"]))
            if e is not None:
                attach_adp_fields(r, e, prior_s)

    # Always compute Layer-1 pure_stats for transparency (even in residual mode)
    tr_pure_vets = [r for s in prior_seasons_pure for r in pure_vet_by_s[s]]
    tr_pure_rks = [r for s in prior_seasons_pure for r in pure_rk_by_s[s]]
    pure_res = fit_predict_pure_stats(tr_pure_vets, tr_pure_rks, cands, prior_s["waiver_fppg"], use_advanced_stats=use_advanced_stats)
    pure_preds = pure_res["pure_stats"]

    if eff_mode == "residual":
        tr_adp_rows = [r for s in prior_seasons_adp for r in adp_pool_by_s[s]]
        res_out = fit_predict_residual_unified(tr_adp_rows, cands, prior_s["waiver_fppg"], lam=lam, use_advanced_stats=use_advanced_stats, recency_halflife=recency_halflife)
        model_preds = res_out["residual_proj"]
        proj_mpg, proj_fppm, proj_avail = res_out["proj_mpg"], res_out["proj_fppm"], res_out["proj_avail"]
        alerts = ["CHECK_INJURY" if r.get("market_collapse_excess", 0.0) > 0 else "" for r in cands]
        edge_raw = res_out["scaled_edges"]
        peer_center = median(edge_raw[i] for i, r in enumerate(cands) if r.get("market_rank", 999) <= BOARD_SIZE) if cands else 0.0
        edges_vs_peers = edge_raw - peer_center
        mode_tag = f"residual_lam{int(round(100 * lam))}"
    else:
        eff_adp_w = adp_weight if has_adp else 0.0
        eff_guard = (collapse_guard or eff_adp_w > 0.0) and has_adp
        model_preds, alerts = apply_optional_adp_layer(cands, pure_preds, adp_weight=eff_adp_w, apply_collapse_guard=eff_guard)
        proj_mpg, proj_fppm, proj_avail = pure_res["proj_mpg"], pure_res["proj_fppm"], pure_res["proj_avail"]
        edges_vs_peers = np.array([
            (float(pure_preds[i]) - r["mkt_expected_adj_fppg"]) if r.get("mkt_expected_adj_fppg") is not None else np.nan
            for i, r in enumerate(cands)
        ], dtype=float)
        mode_tag = "pure_stats" if eff_adp_w == 0.0 else f"pure_stats_blend{int(round(100 * eff_adp_w))}"

    # Apply live ESPN / CSV injury availability caps on top of whichever mode was chosen
    inj_labels = [""] * len(cands)
    inj_capped = 0
    final_preds = np.array(model_preds, dtype=float)
    if injury_records:
        final_preds, inj_labels, inj_capped = apply_injury_report_layer(
            cands, final_preds, proj_mpg, proj_fppm, proj_avail,
            injury_records, target_year, opening_night=opening_night, as_of_date=as_of_date,
        )

    ranked = sorted(range(len(cands)), key=lambda i: (-final_preds[i], -pure_preds[i], cands[i]["player"]))
    board_rows = []
    for rank, idx in enumerate(ranked[:board_limit], start=1):
        r = cands[idx]
        pos_tup = r["positions"]
        adp_val = r.get("adp")
        note = " | ".join(x for x in (alerts[idx], inj_labels[idx]) if x)
        board_rows.append({
            "rank": rank,
            "in_top_150": rank <= BOARD_SIZE,
            "player": r["player"],
            "type": r["player_type"],
            "positions": ",".join(pos_tup),
            "counts_toward_4c_cap": pos_tup[0] == "C",
            "proj_adj_fppg": round(float(final_preds[idx]), 2),
            "pure_stats_adj_fppg": round(float(pure_preds[idx]), 2),
            "edge_vs_peers": round(float(edges_vs_peers[idx]), 2) if math.isfinite(edges_vs_peers[idx]) else None,
            "proj_mpg": round(float(proj_mpg[idx]), 1) if math.isfinite(proj_mpg[idx]) else None,
            "proj_fppm": round(float(proj_fppm[idx]), 3) if math.isfinite(proj_fppm[idx]) else None,
            "proj_avail_pct": round(100.0 * float(proj_avail[idx]), 1) if math.isfinite(proj_avail[idx]) else None,
            "adp": adp_val,
            "slots_ahead_of_adp": round(adp_val - rank, 1) if adp_val is not None else None,
            "injury_or_news_flag": note,
            "prior_games": r.get("prior_games"),
            "prior_fppg": round(r["prior_fppg"], 2) if r.get("prior_fppg") is not None else None,
            "draft_pick": r.get("draft_pick"),
        })

    out_path = data / f"draft_board_{target_season}_{mode_tag}_{scoring_name}.csv"
    write_csv(out_path, board_rows)

    print(f"\n================ {target_season} DRAFT ENGINE BOARD ({mode_tag.upper()}, Scoring: {scoring_name}) ================")
    if eff_mode == "residual":
        print(f"Stage-2 Residual Champion (unified_equal, lam={lam:.2f}): trained on {len(prior_seasons_adp)} prior ADP drafts ({prior_seasons_adp[0]}..{prior_seasons_adp[-1]})")
    else:
        print(f"Stage-1 Pre-ADP Pure Stats (0% ADP): trained on {len(prior_seasons_pure)} completed seasons ({len(tr_pure_vets)} vets, {len(tr_pure_rks)} rookies)")
    if injury_source_notes:
        inj_ref = max(opening_night, as_of_date or date.today())
        print(
            f"Injury Layer: {', '.join(injury_source_notes)} | "
            f"Ref start: {inj_ref.isoformat()} (Opening Night: {opening_night.isoformat()}) | "
            f"Projections capped: {inj_capped}"
        )
    if excluded_keys:
        print(f"Manual exclusions: {exclude_players}")

    print(f"\n=== Top 25 Overall ({mode_tag}) ===")
    print(f"  {'Rk':>3s} | {'Player':22s} {'Typ':>4s} {'Pos':>8s} {'4C':>2s} | {'Proj':>5s} {'Stat':>5s} {'MPG':>4s} {'Avl%':>4s} | {'ADP':>5s} {'Diff':>6s} {'Flags'}")
    print("  " + "-" * 97)
    tag_map = {"returning_vet": "VET", "rookie": "ROOK", "returning_absence": "RET"}
    for r in board_rows[:25]:
        mpg_s = f"{r['proj_mpg']:4.1f}" if r["proj_mpg"] is not None else "   -"
        av_s = f"{r['proj_avail_pct']:4.0f}" if r["proj_avail_pct"] is not None else "   -"
        adp_s = f"{r['adp']:5.1f}" if r["adp"] is not None else "    -"
        diff_s = f"{r['slots_ahead_of_adp']:+6.1f}" if r["slots_ahead_of_adp"] is not None else "     -"
        c_tag = "C" if r["counts_toward_4c_cap"] else ""
        print(
            f"  {r['rank']:3d} | {r['player'][:22]:22s} {tag_map[r['type']]:>4s} {r['positions'][:8]:>8s} {c_tag:>2s} | "
            f"{r['proj_adj_fppg']:5.2f} {r['pure_stats_adj_fppg']:5.2f} {mpg_s} {av_s} | {adp_s} {diff_s} {r['injury_or_news_flag']}"
        )

    capped_rows = [r for r in board_rows if "OUT_SEASON" in r["injury_or_news_flag"] or "INJ_CAP" in r["injury_or_news_flag"]]
    if capped_rows:
        print(f"\n=== Injury-Capped Players ({len(capped_rows)} adjusted from return-date estimates) ===")
        for r in capped_rows[:12]:
            adp_s = f"ADP {r['adp']:5.1f}" if r["adp"] is not None else "Pre-ADP"
            print(f"  Rk #{r['rank']:3d} ({adp_s}) | {r['player']:23s} | Final {r['proj_adj_fppg']:5.2f} (Healthy Stat {r['pure_stats_adj_fppg']:5.2f}) | {r['injury_or_news_flag']}")

    if has_adp:
        targets = sorted(
            [r for r in board_rows[:BOARD_SIZE] if r["adp"] is not None and r["edge_vs_peers"] is not None and r["edge_vs_peers"] > 0 and "OUT_SEASON" not in r["injury_or_news_flag"] and "CHECK_INJURY" not in r["injury_or_news_flag"]],
            key=lambda r: (-r["edge_vs_peers"], -(r["slots_ahead_of_adp"] or 0.0)),
        )[:10]
        if targets:
            print(f"\n=== Top 10 Draft Targets vs. Current ADP ({mode_tag}) ===")
            for r in targets:
                print(
                    f"  Rk #{r['rank']:3d} (ADP {r['adp']:5.1f}, {r['slots_ahead_of_adp']:+5.1f} slots) | "
                    f"{r['player']:23s} | Proj {r['proj_adj_fppg']:5.2f} | Edge {r['edge_vs_peers']:+.2f}"
                )

        fades = sorted(
            [r for r in board_rows if r["adp"] is not None and r["adp"] <= BOARD_SIZE and r["edge_vs_peers"] is not None and r["edge_vs_peers"] < 0],
            key=lambda r: (r["edge_vs_peers"], r["slots_ahead_of_adp"] or 0.0),
        )[:10]
        if fades:
            print(f"\n=== Top 10 Draft Fades vs. Current ADP ({mode_tag}) ===")
            for r in fades:
                cut = " [CUT]" if not r["in_top_150"] else ""
                print(
                    f"  Rk #{r['rank']:3d} (ADP {r['adp']:5.1f}, {r['slots_ahead_of_adp']:+5.1f} slots) | "
                    f"{r['player']:23s} | Proj {r['proj_adj_fppg']:5.2f} | Edge {r['edge_vs_peers']:+.2f}{cut}"
                )

    rks = [r for r in board_rows if r["type"] == "rookie"][:10]
    if rks:
        print(f"\n=== Top Projected Incoming Rookies ({mode_tag}) ===")
        for r in rks:
            pk = f"Pk#{r['draft_pick']}" if r["draft_pick"] is not None else "UDFA"
            adp_s = f"ADP {r['adp']:5.1f}" if r["adp"] is not None else "Pre-ADP"
            print(f"  Rk #{r['rank']:3d} | {r['player']:23s} ({pk:5s}) | Proj {r['proj_adj_fppg']:5.2f} | {adp_s}")

    print(f"\nSaved board to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"), help="season data directory (default: data)")
    parser.add_argument("--scoring", default="default", help="scoring preset (default|espn|yahoo|sleeper), JSON path, or inline JSON")
    parser.add_argument("--backtest", action="store_true", help="run full backtest + 12-team ESPN simulator comparing Stage 1 & Stage 2")
    parser.add_argument("--generate-board", metavar="YYYY-YY", help="generate ranked draft board for a season")
    parser.add_argument("--mode", choices=("auto", "pure_stats", "residual"), default="auto", help="board mode: auto (residual if adp.csv exists else pure_stats), pure_stats, or residual")
    parser.add_argument("--lam", type=float, default=0.50, help="Stage-2 residual scale lambda (default: 0.50)")
    parser.add_argument("--wait-lam", type=float, default=DEFAULT_WAIT_LAM, help=f"Stage-2 Reach-vs-Wait EV residual scale lambda (default: {DEFAULT_WAIT_LAM})")
    parser.add_argument("--adp-weight", type=float, default=0.0, help="optional Stage-1 ADP blend weight in [0, 1] (default: 0.0)")
    parser.add_argument("--collapse-guard", action="store_true", help="auto-fade CHECK_INJURY alerts in pure_stats mode")
    parser.add_argument("--use-live-injuries", action="store_true", help="fetch live ESPN NBA injury report & cap availability from return dates")
    parser.add_argument("--injuries-csv", type=Path, default=None, help="optional local injury CSV (player,status,return_date,notes)")
    parser.add_argument("--opening-night", default=None, help="override Opening Night date (YYYY-MM-DD) for injury games-missed math")
    parser.add_argument("--as-of-date", default=None, help="override reference date (YYYY-MM-DD) for mid-season injury games-missed math (default: today)")
    parser.add_argument("--exclude-players", default="", help="comma-separated confirmed out players to remove completely")
    parser.add_argument("--rookies-csv", type=Path, default=None, help="optional CSV of incoming rookies (player,draft_pick,team,age,pos)")
    parser.add_argument("--holdout-seasons", type=int, default=2, help="latest draft seasons held out in --backtest (default: 2)")
    parser.add_argument("--cv-folds", type=int, default=6, help="max chronological CV folds in --backtest (default: 6)")
    parser.add_argument("--trials-per-slot", type=int, default=12, help="Monte Carlo drafts per slot in --backtest (default: 12)")
    parser.add_argument("--teams", type=int, default=12, help="number of teams in simulated draft leagues (default: 12)")
    parser.add_argument("--adp-var", type=float, default=2.0, help="villain R2-13 Uniform ADP noise bound (default: 2.0)")
    parser.add_argument("--skip-sim", action="store_true", help="skip the 12-team ESPN simulator in --backtest")
    parser.add_argument("--advanced-stats", action="store_true", help="opt-in: enable local BPM/VORP/On-Off/USG%% extension")
    parser.add_argument("--min-train-season", default="2016-17", help="earliest draft season to include in training (default: 2016-17; use 2014-15 or 2008-09/all for expanded history)")
    parser.add_argument("--recency-halflife", type=float, default=None, help="opt-in: exponential recency half-life in seasons (e.g. 6.0); default None = equal weights")
    parser.add_argument("--report-suffix", default="", help="optional suffix appended to saved backtest CSV filenames (e.g. 13yr)")
    parser.add_argument("--seed", type=int, default=None, help="random seed for simulator")
    args = parser.parse_args()

    if not args.backtest and not args.generate_board:
        args.backtest = True
    if not 0.0 <= args.adp_weight <= 1.0 or args.lam < 0:
        parser.error("--adp-weight must be in [0, 1] and --lam >= 0")
    try:
        scoring_name, scoring = resolve_scoring(args.scoring)
        use_adv = bool(args.advanced_stats)
        if args.generate_board:
            generate_live_board(
                data=args.data,
                target_season=args.generate_board,
                scoring_name=scoring_name,
                scoring=scoring,
                mode=args.mode,
                lam=args.lam,
                adp_weight=args.adp_weight,
                collapse_guard=args.collapse_guard,
                use_live_injuries=args.use_live_injuries,
                injuries_csv=args.injuries_csv,
                opening_night_str=args.opening_night,
                as_of_date_str=args.as_of_date,
                exclude_players=args.exclude_players,
                rookies_csv=args.rookies_csv,
                use_advanced_stats=use_adv,
                min_train_season=args.min_train_season,
                recency_halflife=args.recency_halflife,
            )
        else:
            run_backtest(
                data=args.data,
                scoring_name=scoring_name,
                scoring=scoring,
                lam=args.lam,
                wait_lam=args.wait_lam,
                holdout_seasons=args.holdout_seasons,
                cv_folds=args.cv_folds,
                skip_sim=args.skip_sim,
                trials_per_slot=args.trials_per_slot,
                adp_var=args.adp_var,
                seed=args.seed,
                use_advanced_stats=use_adv,
                min_train_season=args.min_train_season,
                recency_halflife=args.recency_halflife,
                report_suffix=args.report_suffix,
                n_teams=args.teams,
            )
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(1, f"Draft engine failed: {error}\n")


if __name__ == "__main__":
    main()
