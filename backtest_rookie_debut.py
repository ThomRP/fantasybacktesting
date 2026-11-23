"""Backtest incoming NBA rookies against ADP using draft pick, rookie age, team age, win %, and logjam."""

import argparse
import csv
import math
import random
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from adp_names import clean_adp_player_name
from backtest_common import (
    adp_slot_points,
    load_birth_dates,
    load_player_profiles,
    name_key,
    parse_minutes,
    player_box_scores,
    read_adp,
    read_csv,
    read_season,
    regular_season_game,
    write_csv,
)
from backtest_rookies import FIRST_ARCHIVE_YEAR, players_with_appearances


REF_GAMES = 65.0
WAIVER_FACTOR = 0.85
RIDGE_ALPHAS = np.logspace(0.0, 3.5, 20)

PICK_FEATURES = (
    "log_draft_pick",
    "log_adp",
    "mkt_expected_adj_fppg",
    "adp_minus_pick_expected",
    "is_redshirt",
)

FULL_ROOKIE_FEATURES = (
    "log_draft_pick",
    "log_adp",
    "mkt_expected_adj_fppg",
    "adp_minus_pick_expected",
    "is_redshirt",
    "rookie_age",
    "team_top8_avg_age",
    "team_prior_win_pct",
    "team_vet30_min_share",
    "team_prior_top150_count",
    "team_same_pos_min_share",
    "espn_vs_avg",
)

MODEL_NAMES = (
    "rookie_flat_bias",
    "pick_and_adp",
    "rookie_full_huber",
    "rookie_full_ridge",
)


def _safe_float(raw):
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def _safe_int(raw):
    val = _safe_float(raw)
    return int(round(val)) if val is not None else None


def compute_season_adj_leaderboard(players):
    """Availability-adjusted FPPG leaderboard matching backtest_final_board.py (ref=65g, 0.85x waiver)."""
    qual = sorted(
        (p.total_points / p.total_games for p in players.values() if p.total_games >= 20),
        reverse=True,
    )
    if not qual:
        qual = sorted((p.total_points / p.total_games for p in players.values() if p.total_games > 0), reverse=True)
    repl_fppg = qual[min(149, len(qual) - 1)] if qual else 20.0
    waiver_fppg = WAIVER_FACTOR * repl_fppg

    adj_by_id = {}
    raw_by_id = {}
    for aid, p in players.items():
        g = p.total_games
        if g <= 0:
            adj_by_id[aid] = waiver_fppg
            raw_by_id[aid] = 0.0
        else:
            raw = p.total_points / g
            raw_by_id[aid] = raw
            eff_g = min(float(g), REF_GAMES)
            adj_by_id[aid] = (eff_g * raw + (REF_GAMES - eff_g) * waiver_fppg) / REF_GAMES

    leaderboard = sorted(adj_by_id.values(), reverse=True)
    rank_by_score = {}
    for place, score in enumerate(leaderboard, start=1):
        rank_by_score.setdefault(score, place)
    return waiver_fppg, adj_by_id, raw_by_id, leaderboard, rank_by_score


def pos_group(pos_raw):
    p = (pos_raw or "").strip().upper()
    if p in ("PG", "SG", "G"):
        return "G"
    if p in ("SF", "PF", "F"):
        return "F"
    if p == "C":
        return "C"
    return None


def read_adp_metadata(path):
    """Read ADP entries plus cross-platform spreads and first-listed position from adp.csv."""
    entries = []
    seen = set()
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
            adp_pos = None
            m = re.search(r"\(([^()\s-]+)\s+-\s+([^()]+)\)", raw_text)
            if m:
                adp_team_abbr = m.group(1).strip().upper()
                adp_pos = m.group(2).split(",")[0].strip().upper()

            entries.append({
                "key": key,
                "player": cleaned,
                "adp": adp,
                "adp_platform_std": std,
                "espn_vs_avg": espn_vs_avg,
                "adp_team_abbr": adp_team_abbr,
                "adp_pos": adp_pos,
            })
    entries.sort(key=lambda r: (r["adp"], r["player"].casefold()))
    return entries


def parse_draft_info_from_profiles(profiles):
    """Extract (draft_year, overall_pick, birth_date, height, pos, team_id) from ESPN player_core profiles."""
    if not profiles:
        return {}, set()
    sample_cols = set(next(iter(profiles.values())).keys())

    # Detect whether draft_selection is overall (1..60) or pick-within-round (1..30)
    rd2_selections = []
    for p in profiles.values():
        rd = _safe_int(p.get("draft_round"))
        sel = _safe_int(p.get("draft_selection"))
        if rd == 2 and sel is not None and sel > 0:
            rd2_selections.append(sel)
    selection_is_overall = any(s > 30 for s in rd2_selections)

    parsed = {}
    for aid, p in profiles.items():
        dy = _safe_int(p.get("draft_year"))
        rd = _safe_int(p.get("draft_round"))
        sel = _safe_int(p.get("draft_selection"))

        overall = None
        for direct_col in ("draft_overall", "overall_pick", "draft_pick"):
            v = _safe_int(p.get(direct_col))
            if v is not None and 1 <= v <= 100:
                overall = v
                break
        if overall is None and sel is not None and sel > 0:
            if selection_is_overall or rd is None or rd <= 1:
                overall = sel
            else:
                overall = 30 * (rd - 1) + sel
        if overall is None:
            txt = str(p.get("draft_display_text") or p.get("draft_text") or "")
            m_rd_pk = re.search(r"(?:rd|round)\s*(\d+)[^0-9]+(?:pk|pick)\s*(\d+)", txt, flags=re.I)
            if m_rd_pk:
                r_num, p_num = int(m_rd_pk.group(1)), int(m_rd_pk.group(2))
                overall = p_num if (p_num > 30 or r_num <= 1) else 30 * (r_num - 1) + p_num

        dob = None
        raw_dob = p.get("date_of_birth") or ""
        if len(raw_dob) >= 10:
            try:
                dob = date.fromisoformat(raw_dob[:10])
            except ValueError:
                dob = None

        height = _safe_float(p.get("height"))
        if height is not None and height <= 0:
            height = None

        exp = _safe_int(p.get("experience_years") or p.get("experience"))
        tid = str(p.get("team_id") or p.get("draft_team_id") or "").strip() or None
        pos = (p.get("position_abbreviation") or p.get("position") or "").strip().upper() or None

        names = set()
        for col in ("display_name", "full_name", "athlete_display_name", "name"):
            val = (p.get(col) or "").strip()
            if val:
                names.add(val)

        parsed[aid] = {
            "draft_year": dy,
            "draft_round": rd,
            "overall_pick": overall,
            "dob": dob,
            "height": height,
            "experience": exp,
            "team_id": tid,
            "position": pos,
            "names": names,
        }
    return parsed, sample_cols


def compute_prior_team_context(prior_folder, all_star_date, prior_births, prior_adj_by_id, prior_rank_by_score):
    """Compute strictly prior-season team win %, established rotation age, vet share, and positional minutes."""
    schedule = list(read_csv(prior_folder / "games.csv"))
    reg_games = [g for g in schedule if regular_season_game(g)]
    game_dates = {g["game_id"]: g["game_date"] for g in reg_games}

    wins = Counter()
    games_played = Counter()
    abbr_to_tid = {}
    for g in reg_games:
        hid, aid = str(g["home_id"]), str(g["away_id"])
        games_played[hid] += 1
        games_played[aid] += 1
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

    # Accumulate player minutes per team + position group in prior regular season
    team_player_min = defaultdict(lambda: defaultdict(float))
    player_pos_counts = defaultdict(Counter)
    for gid, aid, _, stats, pos, _, tid in player_box_scores(prior_folder, game_dates):
        mins = parse_minutes(stats.get("minutes"))
        if mins is None or mins <= 0 or not tid:
            continue
        team_player_min[str(tid)][aid] += mins
        pg = pos_group(pos)
        if pg:
            player_pos_counts[aid][pg] += 1

    team_ctx = {}
    for tid, p_mins in team_player_min.items():
        tot_min = sum(p_mins.values())
        if tot_min <= 0:
            continue

        # Minutes-weighted age of ALL players and of the TOP-8 established rotation players
        weighted_age_sum = 0.0
        age_min_sum = 0.0
        vet30_min = 0.0
        pos_mins = Counter()
        top150_cnt = 0

        ranked_rot = sorted(p_mins.items(), key=lambda kv: -kv[1])
        top8_ids = {aid for aid, _ in ranked_rot[:8]}
        top8_age_sum = 0.0
        top8_min_sum = 0.0

        for aid, mins in ranked_rot:
            pg = player_pos_counts[aid].most_common(1)[0][0] if player_pos_counts[aid] else None
            if pg:
                pos_mins[pg] += mins
            score = prior_adj_by_id.get(aid)
            if score is not None and prior_rank_by_score.get(score, 999) <= 150 and mins >= 500:
                top150_cnt += 1

            dob = prior_births.get(aid)
            if dob is not None and all_star_date is not None:
                age = (all_star_date - dob).days / 365.2425
                weighted_age_sum += age * mins
                age_min_sum += mins
                if age >= 30.0:
                    vet30_min += mins
                if aid in top8_ids:
                    top8_age_sum += age * mins
                    top8_min_sum += mins

        gp = games_played[tid]
        team_ctx[tid] = {
            "win_pct": wins[tid] / gp if gp > 0 else None,
            "avg_age": weighted_age_sum / age_min_sum if age_min_sum > 0 else None,
            "top8_avg_age": top8_age_sum / top8_min_sum if top8_min_sum > 0 else None,
            "vet30_min_share": vet30_min / age_min_sum if age_min_sum > 0 else None,
            "top150_count": top150_cnt,
            "pos_min_share": {k: pos_mins[k] / tot_min for k in ("G", "F", "C")},
        }
    return team_ctx, abbr_to_tid


def first_appearance_teams(season_folder):
    """Map athlete_id -> (first regular-season team_id, primary box-score position)."""
    schedule = list(read_csv(season_folder / "games.csv"))
    game_dates = {g["game_id"]: g["game_date"] for g in schedule if regular_season_game(g)}
    first_seen = {}
    pos_counts = defaultdict(Counter)
    for gid, aid, _, _, pos, _, tid in player_box_scores(season_folder, game_dates):
        gdate = game_dates[gid]
        if tid and (aid not in first_seen or gdate < first_seen[aid][0]):
            first_seen[aid] = (gdate, str(tid))
        if pos:
            pos_counts[aid][pos.strip().upper()] += 1
    return (
        {aid: tid for aid, (_, tid) in first_seen.items()},
        {aid: ctr.most_common(1)[0][0] for aid, ctr in pos_counts.items()},
    )


def build_rookie_dataset(data):
    """Collect all drafted incoming NBA rookies (and veteran benchmark summary) across consecutive seasons."""
    seasons = {int(p.name[:4]): p for p in data.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}", p.name)}
    season_cache = {}

    def get_season(year):
        if year not in season_cache:
            folder = seasons.get(year)
            season_cache[year] = (
                read_season(folder, split_at_all_star=True, include_box_scores=False) if folder else None
            )
        return season_cache[year]

    played_before = set()
    archive_through = FIRST_ARCHIVE_YEAR - 1
    history_available = True
    reported_cols = False

    rookie_rows = []
    vet_benchmark_rows = []

    for year in sorted(seasons):
        following = seasons.get(year + 1)
        if following is None or not (following / "adp.csv").is_file():
            continue
        prior_res, next_res = get_season(year), get_season(year + 1)
        if prior_res is None or next_res is None:
            continue

        # Update historical NBA appearance set strictly through the end of `prior` season (ending year = year + 1)
        if history_available and year >= FIRST_ARCHIVE_YEAR:
            try:
                for ending_year in range(archive_through + 1, year + 2):
                    played_before.update(players_with_appearances(data, ending_year))
                archive_through = year + 1
            except Exception as err:
                print(f"Note: ESPN career appearance history unavailable ({err}); falling back to prior season + draft year")
                history_available = False

        all_star_date, prior_players = prior_res
        _, next_players = next_res
        played_before.update(prior_players.keys())

        prior_waiver, prior_adj_by_id, _, prior_adj_lb, prior_rank_by_score = compute_season_adj_leaderboard(prior_players)
        next_waiver, next_adj_by_id, next_raw_by_id, next_adj_lb, next_rank_by_score = compute_season_adj_leaderboard(next_players)
        next_raw_lb = sorted((v for v in next_raw_by_id.values() if v > 0), reverse=True)

        # Load prior births (for established team ages) and following profiles (for rookie draft picks & DOB)
        try:
            prior_births = load_birth_dates(data, year + 1)
        except Exception:
            prior_births = {}

        try:
            next_profiles = load_player_profiles(data, year + 2)
        except Exception as err:
            print(f"{following.name}: player_core profiles unavailable ({err})")
            next_profiles = {}
        parsed_profiles, profile_cols = parse_draft_info_from_profiles(next_profiles)
        if profile_cols and not reported_cols:
            draft_cols = sorted(c for c in profile_cols if "draft" in c.lower() or "birth" in c.lower() or "height" in c.lower())
            print(f"Detected player_core bio/draft columns: {', '.join(draft_cols)}")
            reported_cols = True

        team_ctx, abbr_to_tid = compute_prior_team_context(
            seasons[year], all_star_date, prior_births, prior_adj_by_id, prior_rank_by_score,
        )
        debut_teams, debut_pos = first_appearance_teams(following)

        # Build name -> athlete_id lookup from following_players + following player_core profiles
        ids_next_played = defaultdict(set)
        for aid, p in next_players.items():
            for n in p.names:
                ids_next_played[name_key(n)].add(aid)
        ids_next_profile = defaultdict(set)
        for aid, prof in parsed_profiles.items():
            for n in prof["names"]:
                ids_next_profile[name_key(n)].add(aid)

        # Also track prior-season returning veterans for the apples-to-apples rookie vs vet benchmark
        ids_prior = defaultdict(set)
        for aid, p in prior_players.items():
            for n in p.names:
                ids_prior[name_key(n)].add(aid)

        adp_entries = read_adp_metadata(following / "adp.csv")
        season_rookies = 0
        draft_date_ref = date(year + 1, 10, 15)  # Pre-season Oct 15 of the rookie's debut season
        matched_ids = set()

        for entry in adp_entries:
            key = entry["key"]
            adp = entry["adp"]

            # Check if returning veteran first (for benchmark comparison)
            prior_ids = ids_prior[key]
            if len(prior_ids) == 1:
                (vid,) = prior_ids
                v_next = next_players.get(vid)
                v_games = v_next.total_games if v_next else 0
                v_adj = next_adj_by_id.get(vid, next_waiver)
                v_rank = next_rank_by_score[v_adj] if (v_games > 0 and v_adj in next_rank_by_score) else (
                    sum(1 for v in next_adj_lb if v > next_waiver) + 1
                )
                v_slot = adp_slot_points(next_adj_lb, adp)
                vet_benchmark_rows.append({
                    "draft_season": following.name,
                    "adp": adp,
                    "next_games": v_games,
                    "beat_adp": v_rank < adp,
                    "fppg_above_adp": v_adj - v_slot,
                    "adp_slots_gained": adp - v_rank,
                })
                continue

            # Resolve athlete_id for non-prior player: prefer played in following season, else player_core profile
            cand_ids = ids_next_played.get(key, set())
            if len(cand_ids) == 1:
                (aid,) = cand_ids
            else:
                prof_cands = [
                    cid for cid in ids_next_profile.get(key, set())
                    if cid not in played_before
                ]
                if len(prof_cands) > 1:
                    recent = [
                        cid for cid in prof_cands
                        if parsed_profiles[cid]["draft_year"] in (year + 1, year)
                    ]
                    if len(recent) == 1:
                        prof_cands = recent
                if len(prof_cands) != 1:
                    continue
                aid = prof_cands[0]
                dy_check = parsed_profiles[aid].get("draft_year")
                if dy_check is not None and dy_check < year - 1:
                    continue

            if aid in matched_ids or aid in played_before:
                continue

            prof = parsed_profiles.get(aid, {})
            dy = prof.get("draft_year")
            # If career history archive was unavailable, guard against returning injured vets via draft_year / experience
            if not history_available:
                if dy is not None and dy < year - 1:
                    continue
                if prof.get("experience") is not None and prof["experience"] > 1:
                    continue

            matched_ids.add(aid)
            season_rookies += 1

            # Determine season-t NBA team (zero historical ADP team leakage: prefer first regular-season game team_id)
            tid = debut_teams.get(aid) or prof.get("team_id")
            if tid is None and entry["adp_team_abbr"]:
                tid = abbr_to_tid.get(entry["adp_team_abbr"])
            tctx = team_ctx.get(str(tid)) if tid is not None else None

            raw_pos = debut_pos.get(aid) or prof.get("position") or entry["adp_pos"]
            pg = pos_group(raw_pos)

            overall_pick = prof.get("overall_pick")
            is_undrafted = 1.0 if (overall_pick is None and prof) else 0.0
            eff_pick = float(overall_pick) if overall_pick is not None else (65.0 if prof else None)
            is_redshirt = 1.0 if (dy is not None and dy < year + 1) else (0.0 if dy is not None else None)

            dob = prof.get("dob")
            rookie_age = (draft_date_ref - dob).days / 365.2425 if dob is not None else None

            mkt_exp_adj = adp_slot_points(prior_adj_lb, adp)
            after = next_players.get(aid)
            next_games = after.total_games if after else 0
            next_raw_fppg = next_raw_by_id.get(aid, 0.0)
            next_adj_fppg = next_adj_by_id.get(aid, next_waiver)
            if next_games > 0 and next_adj_fppg in next_rank_by_score:
                next_adj_rank = next_rank_by_score[next_adj_fppg]
            else:
                next_adj_rank = sum(1 for v in next_adj_lb if v > next_waiver) + 1
            slot_adj = adp_slot_points(next_adj_lb, adp)
            slot_raw = adp_slot_points(next_raw_lb, adp) if next_raw_lb else 0.0

            rookie_rows.append({
                "prior_season": seasons[year].name,
                "draft_season": following.name,
                "player": entry["player"],
                "athlete_id": aid,
                "team_id": tid,
                "position": raw_pos,
                "pos_group": pg,
                "adp": adp,
                "log_adp": math.log(max(adp, 1.0)),
                "mkt_expected_adj_fppg": mkt_exp_adj,
                "draft_year": dy,
                "draft_pick": overall_pick,
                "log_draft_pick": math.log(eff_pick) if eff_pick is not None else None,
                "is_undrafted": is_undrafted if prof else None,
                "is_redshirt": is_redshirt,
                "rookie_age": round(rookie_age, 2) if rookie_age is not None else None,
                "height_inches": prof.get("height"),
                "team_prior_win_pct": round(tctx["win_pct"], 4) if (tctx and tctx["win_pct"] is not None) else None,
                "team_est_avg_age": round(tctx["avg_age"], 2) if (tctx and tctx["avg_age"] is not None) else None,
                "team_top8_avg_age": round(tctx["top8_avg_age"], 2) if (tctx and tctx["top8_avg_age"] is not None) else None,
                "team_vet30_min_share": round(tctx["vet30_min_share"], 4) if (tctx and tctx["vet30_min_share"] is not None) else None,
                "team_prior_top150_count": tctx["top150_count"] if tctx else None,
                "team_same_pos_min_share": round(tctx["pos_min_share"][pg], 4) if (tctx and pg in tctx["pos_min_share"]) else None,
                "adp_platform_std": entry["adp_platform_std"],
                "espn_vs_avg": entry["espn_vs_avg"],
                # Outcomes
                "next_games": next_games,
                "next_raw_fppg": round(next_raw_fppg, 3),
                "raw_fppg_above_adp": round(next_raw_fppg - slot_raw, 3) if next_games > 0 else None,
                "next_adj_fppg": round(next_adj_fppg, 3),
                "adp_slot_adj_fppg": round(slot_adj, 3),
                "fppg_above_adp": round(next_adj_fppg - slot_adj, 3),
                "next_rank": next_adj_rank,
                "adp_slots_gained": round(adp - next_adj_rank, 3),
                "beat_adp": next_adj_rank < adp,
            })

        print(f"{seasons[year].name} -> {following.name}: found {season_rookies} drafted incoming rookies (out of {len(adp_entries)} ADP entries)")

    return rookie_rows, vet_benchmark_rows


def cohort_stats(rows, label):
    if not rows:
        return None
    n = len(rows)
    hits = sum(1 for r in rows if r["beat_adp"])
    raw_diffs = [r["raw_fppg_above_adp"] for r in rows if r.get("raw_fppg_above_adp") is not None]
    return {
        "group": label,
        "n": n,
        "avg_adp": round(mean(r["adp"] for r in rows), 1),
        "avg_games": round(mean(r["next_games"] for r in rows), 1),
        "zero_g": sum(1 for r in rows if r["next_games"] == 0),
        "beat_adp_pct": round(100.0 * hits / n, 1),
        "adj_fppg_vs_adp": round(mean(r["fppg_above_adp"] for r in rows), 2),
        "raw_fppg_vs_adp": round(mean(raw_diffs), 2) if raw_diffs else None,
        "slots_gained": round(mean(r["adp_slots_gained"] for r in rows), 1),
    }


def print_breakdown_table(title, groups):
    rows = [s for label, grp in groups if (s := cohort_stats(grp, label)) is not None]
    if not rows:
        return []
    print(f"\n=== {title} ===")
    print(f"  {'Group':28s} | {'N':>4s} {'ADP':>5s} {'Gm':>4s} {'0g':>2s} | {'BeatADP':>7s} {'AdjVsADP':>8s} {'RawVsADP':>8s} {'Slots':>6s}")
    print("  " + "-" * 86)
    for r in rows:
        raw_s = f"{r['raw_fppg_vs_adp']:+8.2f}" if r["raw_fppg_vs_adp"] is not None else f"{'n/a':>8s}"
        print(
            f"  {r['group']:28s} | {r['n']:4d} {r['avg_adp']:5.1f} {r['avg_games']:4.1f} {r['zero_g']:2d} | "
            f"{r['beat_adp_pct']:6.1f}% {r['adj_fppg_vs_adp']:+8.2f} {raw_s} {r['slots_gained']:+6.1f}"
        )
    return [{"breakdown": title, **r} for r in rows]


def pearson_and_spearman(x, y):
    if len(x) < 6:
        return None, None
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.std() == 0 or ya.std() == 0:
        return None, None
    r = float(np.corrcoef(xa, ya)[0, 1])
    rx = np.argsort(np.argsort(xa)).astype(float)
    ry = np.argsort(np.argsort(ya)).astype(float)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    return r, rho


def to_matrix(rows, features):
    return np.array(
        [[np.nan if row.get(f) is None else float(row[f]) for f in features] for row in rows],
        dtype=float,
    )


def active_features(training, features):
    return [f for f in features if len({row[f] for row in training if row.get(f) is not None}) > 1]


def attach_pick_expected_feature(training, evaluated):
    """Fit expected AdjFPPG from log_draft_pick strictly on `training`, then compute Rookie Hype Tax."""
    tr_valid = [r for r in training if r.get("log_draft_pick") is not None]
    if len(tr_valid) >= 8:
        X_tr = np.array([[float(r["log_draft_pick"])] for r in tr_valid], dtype=float)
        y_tr = np.array([float(r["next_adj_fppg"]) for r in tr_valid], dtype=float)
        mod = HuberRegressor(alpha=5.0, max_iter=500).fit(X_tr, y_tr)
        fallback = float(np.median(y_tr))
        for grp in (training, evaluated):
            for r in grp:
                if r.get("log_draft_pick") is not None:
                    pe = float(mod.predict([[float(r["log_draft_pick"])]])[0])
                else:
                    pe = fallback
                r["pick_expected_adj_fppg"] = pe
                r["adp_minus_pick_expected"] = float(r["mkt_expected_adj_fppg"]) - pe
    else:
        for grp in (training, evaluated):
            for r in grp:
                r["pick_expected_adj_fppg"] = None
                r["adp_minus_pick_expected"] = None


def fit_and_score_rookies(training, evaluated):
    """Fit lean rookie models strictly on `training` and score `evaluated`."""
    attach_pick_expected_feature(training, evaluated)
    y_res = np.array([float(r["fppg_above_adp"]) for r in training], dtype=float)
    scores = {}

    # 1. Flat rookie bias (constant adjustment equal to historical mean rookie residual)
    flat_val = float(np.median(y_res))
    scores["rookie_flat_bias"] = np.full(len(evaluated), flat_val, dtype=float)

    # 2. Pick + ADP only (Huber)
    p_feats = active_features(training, PICK_FEATURES)
    if p_feats:
        pick_mod = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            HuberRegressor(alpha=25.0, epsilon=1.35, max_iter=1000),
        )
        pick_mod.fit(to_matrix(training, p_feats), y_res)
        scores["pick_and_adp"] = pick_mod.predict(to_matrix(evaluated, p_feats))
    else:
        scores["pick_and_adp"] = scores["rookie_flat_bias"].copy()

    # 3 & 4. Full lean rookie model (Huber & RidgeCV)
    f_feats = active_features(training, FULL_ROOKIE_FEATURES)
    huber_coefs = {}
    if f_feats:
        h_mod = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            HuberRegressor(alpha=30.0, epsilon=1.35, max_iter=1000),
        )
        h_mod.fit(to_matrix(training, f_feats), y_res)
        scores["rookie_full_huber"] = h_mod.predict(to_matrix(evaluated, f_feats))
        huber_coefs = dict(zip(f_feats, map(float, h_mod.named_steps["huberregressor"].coef_)))

        r_mod = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            RidgeCV(alphas=RIDGE_ALPHAS),
        )
        r_mod.fit(to_matrix(training, f_feats), y_res)
        scores["rookie_full_ridge"] = r_mod.predict(to_matrix(evaluated, f_feats))
    else:
        scores["rookie_full_huber"] = scores["rookie_flat_bias"].copy()
        scores["rookie_full_ridge"] = scores["rookie_flat_bias"].copy()

    return scores, huber_coefs


def evaluate_rookie_models(rows, split_label, top_pct=33.3):
    """Compare Top-33% Rookie Targets vs Bottom-33% Rookie Fades + AdjFPPG MAE vs Market ADP."""
    if not rows:
        return []
    seasons = sorted({r["draft_season"] for r in rows})
    mkt_mae = mean(abs(r["fppg_above_adp"]) for r in rows)
    pool_avg = mean(r["fppg_above_adp"] for r in rows)
    pool_hits = 100.0 * sum(1 for r in rows if r["beat_adp"]) / len(rows)

    out = [{
        "split": split_label,
        "model": "all_rookies_market",
        "seasons": len(seasons),
        "n_rookies": len(rows),
        "proj_mae_adj_fppg": round(mkt_mae, 3),
        "top_n": len(rows),
        "top_beat_adp_pct": round(pool_hits, 1),
        "top_avg_vs_adp": round(pool_avg, 2),
        "top_lift_vs_pool": 0.0,
        "bot_avg_vs_adp": round(pool_avg, 2),
        "top_minus_bot_fppg": 0.0,
        "top_slots_gained": round(mean(r["adp_slots_gained"] for r in rows), 1),
    }]

    for model in MODEL_NAMES:
        col = f"{model}_score"
        mae = mean(abs(r[col] - r["fppg_above_adp"]) for r in rows)
        has_ranking = any(len({r[col] for r in rows if r["draft_season"] == s}) > 1 for s in seasons)
        if not has_ranking:
            out.append({
                "split": split_label,
                "model": model,
                "seasons": len(seasons),
                "n_rookies": len(rows),
                "proj_mae_adj_fppg": round(mae, 3),
                "top_n": None,
                "top_beat_adp_pct": None,
                "top_avg_vs_adp": None,
                "top_lift_vs_pool": None,
                "bot_avg_vs_adp": None,
                "top_minus_bot_fppg": None,
                "top_slots_gained": None,
            })
            continue
        top_picks, bot_picks = [], []
        for s in seasons:
            grp = [r for r in rows if r["draft_season"] == s]
            k = max(1, math.ceil(len(grp) * top_pct / 100.0))
            grp.sort(key=lambda r: (-r[col], r["adp"], r["athlete_id"]))
            top_picks.extend(grp[:k])
            bot_picks.extend(grp[-k:])
        t_avg = mean(r["fppg_above_adp"] for r in top_picks)
        b_avg = mean(r["fppg_above_adp"] for r in bot_picks)
        out.append({
            "split": split_label,
            "model": model,
            "seasons": len(seasons),
            "n_rookies": len(rows),
            "proj_mae_adj_fppg": round(mae, 3),
            "top_n": len(top_picks),
            "top_beat_adp_pct": round(100.0 * sum(1 for r in top_picks if r["beat_adp"]) / len(top_picks), 1),
            "top_avg_vs_adp": round(t_avg, 2),
            "top_lift_vs_pool": round(t_avg - pool_avg, 2),
            "bot_avg_vs_adp": round(b_avg, 2),
            "top_minus_bot_fppg": round(t_avg - b_avg, 2),
            "top_slots_gained": round(mean(r["adp_slots_gained"] for r in top_picks), 1),
        })
    return out


def run_backtest(data, holdout_seasons=2, cv_folds=6, max_adp=200.0):
    rookies_all, vets_all = build_rookie_dataset(data)
    rookies = [r for r in rookies_all if r["adp"] <= max_adp]
    vets = [r for r in vets_all if r["adp"] <= max_adp]
    if not rookies:
        raise ValueError("no drafted incoming rookies found across downloaded seasons")

    seasons = sorted({r["draft_season"] for r in rookies})
    if len(seasons) <= holdout_seasons:
        raise ValueError(f"need more completed draft seasons than --holdout-seasons ({len(seasons)} found)")

    development, holdout = seasons[:-holdout_seasons], seasons[-holdout_seasons:]
    dev_rookies = [r for r in rookies if r["draft_season"] in development]
    ho_rookies = [r for r in rookies if r["draft_season"] in holdout]

    # 1. Descriptive Cohort Breakdowns (strictly on Development + Overall shown clearly)
    breakdowns = []
    breakdowns.extend(print_breakdown_table(
        f"Rookie Reality Check vs. Returning Veterans (All {len(seasons)} Seasons, ADP <= {max_adp:g})",
        [
            ("All Drafted Rookies", rookies),
            ("Returning Veterans (same ADP)", vets),
            ("Rookies (Dev seasons)", dev_rookies),
            ("Rookies (Holdout seasons)", ho_rookies),
        ],
    ))

    breakdowns.extend(print_breakdown_table(
        "Rookies by Fantasy ADP Tier (All Seasons)",
        [
            ("ADP 1-50", [r for r in rookies if r["adp"] <= 50]),
            ("ADP 51-100", [r for r in rookies if 50 < r["adp"] <= 100]),
            ("ADP 101-150", [r for r in rookies if 100 < r["adp"] <= 150]),
            ("ADP 151+", [r for r in rookies if r["adp"] > 150]),
        ],
    ))

    breakdowns.extend(print_breakdown_table(
        "Rookies by NBA Draft Pick Tier (All Seasons)",
        [
            ("Top 3 Pick", [r for r in rookies if r["draft_pick"] is not None and r["draft_pick"] <= 3]),
            ("Picks 4-7", [r for r in rookies if r["draft_pick"] is not None and 4 <= r["draft_pick"] <= 7]),
            ("Picks 8-14 (Late Lottery)", [r for r in rookies if r["draft_pick"] is not None and 8 <= r["draft_pick"] <= 14]),
            ("Picks 15-30 (Late 1st Rd)", [r for r in rookies if r["draft_pick"] is not None and 15 <= r["draft_pick"] <= 30]),
            ("2nd Rd / Undrafted", [r for r in rookies if r["draft_pick"] is None or r["draft_pick"] > 30]),
            ("Redshirt Rookies (prev draft)", [r for r in rookies if r["is_redshirt"] == 1.0]),
        ],
    ))

    breakdowns.extend(print_breakdown_table(
        "Rookies by Age on Oct 15 of Debut Season (All Seasons)",
        [
            ("Age <= 19.5 (One-and-done)", [r for r in rookies if r["rookie_age"] is not None and r["rookie_age"] <= 19.5]),
            ("Age 19.5 - 20.5", [r for r in rookies if r["rookie_age"] is not None and 19.5 < r["rookie_age"] <= 20.5]),
            ("Age 20.5 - 21.5", [r for r in rookies if r["rookie_age"] is not None and 20.5 < r["rookie_age"] <= 21.5]),
            ("Age > 21.5 (Older rookie)", [r for r in rookies if r["rookie_age"] is not None and r["rookie_age"] > 21.5]),
        ],
    ))

    # Tertiles of Established Team Rotation Age (User's Core Hypothesis!)
    aged = [r for r in rookies if r["team_top8_avg_age"] is not None]
    if len(aged) >= 9:
        q33, q67 = np.quantile([r["team_top8_avg_age"] for r in aged], [1 / 3, 2 / 3])
        breakdowns.extend(print_breakdown_table(
            "Rookies by Established Team Top-8 Rotation Age (Prior Season)",
            [
                (f"Youngest Teams (<= {q33:.1f} yrs)", [r for r in aged if r["team_top8_avg_age"] <= q33]),
                (f"Middle Teams ({q33:.1f}-{q67:.1f})", [r for r in aged if q33 < r["team_top8_avg_age"] <= q67]),
                (f"Oldest Teams (> {q67:.1f} yrs)", [r for r in aged if r["team_top8_avg_age"] > q67]),
            ],
        ))

    breakdowns.extend(print_breakdown_table(
        "Rookies by NBA Team Prior-Season Win %",
        [
            ("Rebuilding / Tanking (< .400)", [r for r in rookies if r["team_prior_win_pct"] is not None and r["team_prior_win_pct"] < 0.40]),
            ("Play-In / Mid (.400 - .550)", [r for r in rookies if r["team_prior_win_pct"] is not None and 0.40 <= r["team_prior_win_pct"] < 0.55]),
            ("Contenders (>= .550)", [r for r in rookies if r["team_prior_win_pct"] is not None and r["team_prior_win_pct"] >= 0.55]),
        ],
    ))

    # 2. Univariate Signal Audit strictly on Development seasons (zero holdout peeking)
    corr_features = (
        "log_draft_pick", "adp", "rookie_age", "is_redshirt", "height_inches",
        "team_top8_avg_age", "team_est_avg_age", "team_vet30_min_share",
        "team_prior_win_pct", "team_prior_top150_count", "team_same_pos_min_share",
        "espn_vs_avg", "adp_platform_std",
    )
    corr_rows = []
    print(f"\n=== Univariate Correlations with Rookie AdjFPPG vs. ADP (Dev Seasons: {len(dev_rookies)} rookies) ===")
    print(f"  {'Feature':25s} | {'N':>4s} | {'Pearson r':>9s} | {'Spearman':>8s}")
    print("  " + "-" * 56)
    for feat in corr_features:
        pairs = [(r[feat], r["fppg_above_adp"]) for r in dev_rookies if r.get(feat) is not None]
        if len(pairs) < 6:
            continue
        r_val, rho_val = pearson_and_spearman([a for a, _ in pairs], [b for _, b in pairs])
        if r_val is None:
            continue
        corr_rows.append({"feature": feat, "n": len(pairs), "pearson_r": round(r_val, 3), "spearman_rho": round(rho_val, 3)})
    corr_rows.sort(key=lambda d: -abs(d["spearman_rho"]))
    for c in corr_rows:
        print(f"  {c['feature']:25s} | {c['n']:4d} | {c['pearson_r']:+9.3f} | {c['spearman_rho']:+8.3f}")

    # 3. Expanding-Window Chronological Model Backtest (CV + Final Holdout)
    cv_seasons = development[2:][-cv_folds:] if len(development) >= 3 else development[1:]
    cv_scored = []
    for season in cv_seasons:
        earlier = [r.copy() for r in dev_rookies if r["draft_season"] < season]
        cands = [r.copy() for r in dev_rookies if r["draft_season"] == season]
        if len(earlier) < 15 or not cands:
            continue
        preds, _ = fit_and_score_rookies(earlier, cands)
        for i, row in enumerate(cands):
            row["split"] = "cv"
            row["training_through"] = max(r["draft_season"] for r in earlier)
            for name, vals in preds.items():
                row[f"{name}_score"] = round(float(vals[i]), 4)
        cv_scored.extend(cands)

    ho_scored = [r.copy() for r in ho_rookies]
    dev_copy = [r.copy() for r in dev_rookies]
    ho_preds, final_coefs = fit_and_score_rookies(dev_copy, ho_scored)
    for i, row in enumerate(ho_scored):
        row["split"] = "holdout"
        row["training_through"] = development[-1]
        for name, vals in ho_preds.items():
            row[f"{name}_score"] = round(float(vals[i]), 4)

    model_summary = []
    for split_label, split_rows in (("cv", cv_scored), ("holdout", ho_scored)):
        rows_eval = evaluate_rookie_models(split_rows, split_label)
        model_summary.extend(rows_eval)
        if not rows_eval:
            continue
        print(f"\n=== Rookie Model Backtest — {split_label.upper()} ({rows_eval[0]['seasons']} seasons, {rows_eval[0]['n_rookies']} rookies) ===")
        print(f"  {'Model':20s} | {'MAE':>5s} | {'Top33% BeatADP':>14s} {'Top33% vsADP':>12s} {'Lift':>6s} | {'Bot33% vsADP':>12s} {'Top-Bot':>8s}")
        print("  " + "-" * 92)
        for r in rows_eval:
            if r["top_beat_adp_pct"] is None:
                print(
                    f"  {r['model']:20s} | {r['proj_mae_adj_fppg']:5.2f} | "
                    f"{'(constant bias)':>34s} | {'n/a':>12s} {'n/a':>8s}"
                )
            else:
                print(
                    f"  {r['model']:20s} | {r['proj_mae_adj_fppg']:5.2f} | "
                    f"{r['top_beat_adp_pct']:13.1f}% {r['top_avg_vs_adp']:+12.2f} {r['top_lift_vs_pool']:+6.2f} | "
                    f"{r['bot_avg_vs_adp']:+12.2f} {r['top_minus_bot_fppg']:+8.2f}"
                )

    if final_coefs:
        top_w = sorted(final_coefs.items(), key=lambda kv: -abs(kv[1]))
        print("\nFinal Huber standardized weights (rookie_full_huber): " + ", ".join(f"{k}={v:+.2f}" for k, v in top_w[:8]))

    write_csv(data / "backtest_rookie_debut_dataset.csv", rookies)
    write_csv(data / "backtest_rookie_debut_breakdowns.csv", breakdowns)
    write_csv(data / "backtest_rookie_debut_correlations.csv", corr_rows)
    write_csv(data / "backtest_rookie_debut_summary.csv", model_summary)
    write_csv(data / "backtest_rookie_debut_predictions.csv", cv_scored + ho_scored)
    print(f"\nSaved rookie debut outputs under {data}/backtest_rookie_debut_*.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"), help="download folder (default: data)")
    parser.add_argument("--holdout-seasons", type=int, default=2, help="latest draft seasons held out (default: 2)")
    parser.add_argument("--cv-folds", type=int, default=6, help="max chronological CV folds (default: 6)")
    parser.add_argument("--max-adp", type=float, default=200.0, help="max rookie ADP included (default: 200)")
    args = parser.parse_args()
    if args.holdout_seasons < 1 or args.cv_folds < 1 or args.max_adp <= 0:
        parser.error("use valid positive parameters")
    try:
        run_backtest(**vars(args))
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(1, f"Rookie debut backtest failed: {error}\n")


if __name__ == "__main__":
    main()
