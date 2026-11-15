"""Behavioral checks for dated schemas, identity, scoring, and valid market joins."""
from datetime import date

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.config import SCORING
from src.normalize import (
    ADP_SCHEMA, GAME_SCHEMA, PLAYER_SCHEMA, PlayerResolver,
    build_players, calculate_fantasy_points, filter_game_logs, nba_id,
    normalize_adp, normalize_game_logs, normalize_name, write_parquet,
)


@pytest.fixture
def players():
    return pd.DataFrame([
        {"canonical_player_id": "nba:1", "full_name": "José O'Neal Jr.",
         "normalized_name": "jose oneal", "primary_position": "G", "birth_date": date(1990, 1, 1)},
        {"canonical_player_id": "nba:2", "full_name": "Anthony Davis",
         "normalized_name": "anthony davis", "primary_position": "F-C", "birth_date": date(1993, 3, 11)},
    ])


@pytest.fixture
def box():
    return {"season": 2025, "game_date": "2024-10-23", "player_id": 1,
            "player_name": "José O'Neal Jr.", "team_abbreviation": "BOS", "matchup": "BOS vs. NYK",
            "min": "30:30", "fgm": 8, "fga": 18, "ftm": 5, "fta": 6,
            "fg3m": 3, "reb": 10, "ast": 7, "stl": 2, "blk": 1, "tov": 4, "pts": 24,
            "season_type": "regular-season"}


def market(**overrides):
    row = {"season": 2025, "canonical_player_id": "nba:1", "adp": 2.5,
           "adp_rank": 1, "position_rank": 1, "available_at": "2024-10-01",
           "source": "https://example.test/archive/2024-10-01.csv"}
    return {**row, **overrides}


def test_scoring_exactly_matches_supplied_league_image(box):
    assert SCORING == {"fgm": 2, "fga": -1, "ftm": 1, "fta": -1,
                       "fg3m": 1, "reb": 1, "ast": 2, "stl": 4,
                       "blk": 4, "tov": -2, "pts": 1}
    # 16-18+5-6+3+10+14+8+4-8+24 = 52.
    assert calculate_fantasy_points(pd.DataFrame([box])).iloc[0] == 52
    assert calculate_fantasy_points(pd.DataFrame([box]), {"pts": 2, "tov": -3}).iloc[0] == 36


def test_scoring_missing_stat_fails_instead_of_zero_filling(box):
    del box["fga"]
    with pytest.raises(ValueError, match="Missing required columns"):
        calculate_fantasy_points(pd.DataFrame([box]))


@pytest.mark.parametrize("bad", [None, float("inf"), float("nan")])
def test_scoring_nonfinite_or_unknown_stats_fail(box, bad):
    box["pts"] = bad
    with pytest.raises(ValueError):
        calculate_fantasy_points(pd.DataFrame([box]))


def test_no_duplicate_keys_after_benign_raw_repetition(players, box):
    result = normalize_game_logs(pd.DataFrame([box, box]), players)
    assert len(result) == 1
    assert not result.duplicated(["season", "game_date", "canonical_player_id"]).any()
    assert result.minutes_or_snaps.iloc[0] == 30.5
    assert result.opponent.iloc[0] == "NYK"


def test_conflicting_duplicate_keys_raise(players, box):
    conflict = {**box, "pts": 25}
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        normalize_game_logs(pd.DataFrame([box, conflict]), players)


def test_traded_players_keep_one_canonical_identity(players, box):
    traded = {**box, "game_date": "2025-02-06", "team_abbreviation": "LAL", "matchup": "LAL @ BOS"}
    result = normalize_game_logs(pd.DataFrame([box, traded]), players)
    assert result.canonical_player_id.tolist() == ["nba:1", "nba:1"]
    assert set(result.team) == {"BOS", "LAL"}
    assert not result.duplicated(["season", "game_date", "canonical_player_id"]).any()


def test_playoffs_excluded_but_rounded_zero_minute_appearances_retained(players, box):
    playoff = {**box, "game_date": "2025-05-01", "season_type": "playoffs"}
    subminute = {**box, "game_date": "2024-10-24", "min": 0}
    result = normalize_game_logs(pd.DataFrame([box, playoff, subminute]), players)
    assert len(result) == 2
    assert result.game_date.tolist() == [date(2024, 10, 23), date(2024, 10, 24)]
    assert result.minutes_or_snaps.tolist() == [30.5, 0.0]


@pytest.mark.parametrize("bad_date", [None, "not-a-date", "2024-10-01", "2026-01-01"])
def test_game_date_is_exact_required_and_belongs_to_season(players, box, bad_date):
    box["game_date"] = bad_date
    with pytest.raises((ValueError, TypeError)):
        normalize_game_logs(pd.DataFrame([box]), players)


def test_unknown_player_cannot_enter_box_score_table(players, box):
    box["player_id"] = 999
    with pytest.raises(ValueError, match="unresolved"):
        normalize_game_logs(pd.DataFrame([box]), players)


@pytest.mark.parametrize("name,expected", [
    ("  José O'Neal Jr. ", "jose oneal"), ("Gary Payton II", "gary payton"),
    ("Robert Williams III", "robert williams"), ("Nicolas Claxton Sr.", "nicolas claxton"),
])
def test_name_normalization_removes_accents_punctuation_and_suffixes(name, expected):
    assert normalize_name(name) == expected


def test_ids_and_unambiguous_names_resolve_across_sources(players):
    aliases = pd.DataFrame([{"source": "espn", "source_player_id": "101", "canonical_player_id": "nba:2"}])
    resolver = PlayerResolver(players, aliases)
    assert resolver.resolve(full_name="JOSE ONEAL") == "nba:1"
    assert resolver.resolve(nba_player_id="2.0") == "nba:2"
    assert resolver.resolve(source="espn", source_player_id="101") == "nba:2"
    with pytest.raises(ValueError, match="Unmapped"):
        resolver.resolve(source="espn", source_player_id="102", full_name="Anthony Davis")
    with pytest.raises(ValueError, match="Unknown"):
        resolver.resolve(canonical_player_id="nba:404")


def test_name_collision_requires_birthday_or_explicit_id(players):
    another = players.iloc[[0]].copy()
    another["canonical_player_id"] = "nba:3"
    another["birth_date"] = date(1960, 1, 1)
    resolver = PlayerResolver(pd.concat([players, another], ignore_index=True))
    with pytest.raises(ValueError, match="ambiguous"):
        resolver.resolve(full_name="Jose O'Neal III")
    assert resolver.resolve(full_name="Jose O'Neal", birth_date="1960-01-01") == "nba:3"
    assert resolver.resolve(canonical_player_id="nba:1") == "nba:1"


def test_master_consolidates_teams_and_seasons_by_id():
    rosters = pd.DataFrame([
        {"player_id": 1, "player": "José O'Neal Jr.", "position": "G", "birth_date": "JAN 1, 1990", "season": 2024, "team": "BOS"},
        {"player_id": 1, "player": "José O'Neal Jr.", "position": "F-G", "birth_date": "1990-01-01", "season": 2025, "team": "LAL"},
    ])
    result = build_players(rosters)
    assert len(result) == 1
    assert result.canonical_player_id.iloc[0] == "nba:1"
    assert result.birth_date.iloc[0] == date(1990, 1, 1)
    assert result.primary_position.iloc[0] == "F-G"


def test_master_conflicting_birthday_is_not_silently_overwritten():
    rows = [{"player_id": 1, "player": "A", "position": "G", "birth_date": b, "season": s}
            for b, s in [("1990-01-01", 2024), ("1991-01-01", 2025)]]
    with pytest.raises(ValueError, match="Conflicting birth dates"):
        build_players(pd.DataFrame(rows))


def test_point_in_time_slice_uses_exclusive_asof_and_rolling_window():
    logs = pd.DataFrame({"game_date": pd.date_range("2025-01-01", "2025-02-01").date, "fantasy_points": range(32)})
    result = filter_game_logs(logs, as_of="2025-01-20", start="2025-01-06")
    assert len(result) == 14
    assert result.game_date.min() == date(2025, 1, 6)
    assert result.game_date.max() == date(2025, 1, 19)
    assert (result.game_date < date(2025, 1, 20)).all()
    # Even an end beyond as_of cannot reveal future observations.
    pd.testing.assert_frame_equal(result, filter_game_logs(logs, as_of="2025-01-20", start="2025-01-06", end="2025-02-01"))
    assert filter_game_logs(logs, as_of="2025-01-20", end="2025-01-10").game_date.max() == date(2025, 1, 9)


@pytest.mark.parametrize("as_of", ["2025-01-20 12:00", "2025-01-20T00:00:00Z", None])
def test_day_granularity_query_rejects_ambiguous_cutoff(as_of):
    logs = pd.DataFrame({"game_date": [date(2025, 1, 1)]})
    with pytest.raises(ValueError):
        filter_game_logs(logs, as_of=as_of)


def test_adp_validated_foreign_key_join(players):
    raw = pd.DataFrame([market(), market(canonical_player_id="nba:2", adp=8.2, adp_rank=2, position_rank=1)])
    result = normalize_adp(raw, players)
    joined = result.merge(players, on="canonical_player_id", how="left", validate="many_to_one", indicator=True)
    assert joined._merge.eq("both").all()
    assert not result.duplicated(["season", "canonical_player_id"]).any()
    assert result.adp_rank.dtype == "int64"
    assert result.available_at.tolist() == [date(2024, 10, 1)] * 2


@pytest.mark.parametrize("override,match", [
    ({"canonical_player_id": "nba:404"}, "Unknown"),
    ({"available_at": "2024-10-22"}, "precede"),
    ({"available_at": "2025-01-01"}, "precede"),
    ({"available_at": "2020-10-01"}, "draft cycle"),
    ({"available_at": None}, "cannot be missing"),
    ({"source": None}, "provenance"),
    ({"source": "  "}, "provenance"),
    ({"adp": 0}, "positive"),
    ({"adp": float("inf")}, "positive"),
    ({"adp_rank": 1.2}, "integers"),
    ({"position_rank": 0}, "positive"),
])
def test_adp_invalid_prices_provenance_and_joins_rejected(players, override, match):
    with pytest.raises(ValueError, match=match):
        normalize_adp(pd.DataFrame([market(**override)]), players)


def test_adp_missing_provenance_cannot_be_inferred_from_file_mtime(players):
    row = market()
    del row["available_at"]
    with pytest.raises(ValueError, match="available_at"):
        normalize_adp(pd.DataFrame([row]), players)


def test_adp_snapshot_selection_is_whole_snapshot_not_playerwise(players):
    raw = pd.DataFrame([
        market(canonical_player_id="nba:1", available_at="2024-09-01", adp=2),
        market(canonical_player_id="nba:2", available_at="2024-09-01", adp=5, adp_rank=2),
        market(canonical_player_id="nba:1", available_at="2024-10-01", adp=3),
    ])
    result = normalize_adp(raw, players)
    assert len(result) == 1  # Do not backfill absent players using older market knowledge.
    assert result.adp.iloc[0] == 3
    assert result.canonical_player_id.iloc[0] == "nba:1"


def test_adp_duplicate_market_player_and_rank_rejected(players):
    with pytest.raises(ValueError, match="Duplicate ADP"):
        normalize_adp(pd.DataFrame([market(), market(adp=3)]), players)
    with pytest.raises(ValueError, match="ranks must be unique"):
        normalize_adp(pd.DataFrame([market(), market(canonical_player_id="nba:2")]), players)


def test_adp_different_markets_not_blended(players):
    with pytest.raises(ValueError, match="different ADP markets"):
        normalize_adp(pd.DataFrame([market(), market(canonical_player_id="nba:2", source="other")]), players)


def test_adp_ranks_can_be_derived_from_actual_price_order(players):
    raw = pd.DataFrame([market(adp=10), market(canonical_player_id="nba:2", adp=5)]).drop(columns="adp_rank")
    result = normalize_adp(raw, players)
    assert result.canonical_player_id.tolist() == ["nba:2", "nba:1"]
    assert result.adp_rank.tolist() == [1, 2]


def test_parquet_tables_enforce_requested_schema_and_exact_dates(tmp_path, players, box):
    games = normalize_game_logs(pd.DataFrame([box]), players)
    adp = normalize_adp(pd.DataFrame([market()]), players)
    for name, frame, schema in [("players", players, PLAYER_SCHEMA), ("game_logs", games, GAME_SCHEMA), ("adp", adp, ADP_SCHEMA)]:
        path = tmp_path / f"{name}.parquet"
        write_parquet(frame, path, schema)
        disk = pq.read_table(path)
        assert disk.schema.remove_metadata().equals(schema)
        assert disk.num_rows == len(frame)
    assert pq.read_schema(tmp_path / "game_logs.parquet").field("game_date").type == pa.date32()
    assert pd.read_parquet(tmp_path / "game_logs.parquet").game_date.iloc[0] == date(2024, 10, 23)
    assert not list(tmp_path.glob(".*.parquet.*"))


def test_empty_adp_still_writes_explicit_typed_schema(tmp_path, players):
    empty = normalize_adp(pd.DataFrame(), players)
    path = tmp_path / "adp.parquet"
    write_parquet(empty, path, ADP_SCHEMA)
    assert pq.read_table(path).num_rows == 0
    assert pq.read_schema(path).remove_metadata().equals(ADP_SCHEMA)


@pytest.mark.parametrize("value", ["", -1, 0, "bad", float("nan")])
def test_invalid_nba_ids_are_not_coerced_to_unrelated_entities(value):
    with pytest.raises(ValueError):
        nba_id(value)


def mini_schedule():
    return pd.DataFrame([
        {"game_id": game_id, "team_abbreviation": team, "player_id": player,
         "season_type": "regular-season"}
        for game_id in ["0022400001", "0022400002"]
        for team, ids in [("BOS", range(1, 6)), ("NYK", range(6, 11))]
        for player in ids
    ])


def test_season_coverage_requires_schedule_count_not_just_nonempty_table():
    from src.normalize import validate_season_coverage
    with pytest.raises(ValueError, match="Incomplete season 2025: 2 games, expected 1230"):
        validate_season_coverage(mini_schedule(), 2025)


def test_season_coverage_accepts_complete_schedule_with_two_full_teams(monkeypatch):
    import src.normalize as normalize
    monkeypatch.setitem(normalize.EXPECTED_GAMES, 2025, 2)
    assert normalize.validate_season_coverage(mini_schedule(), 2025) == {
        "source_game_count": 2, "source_player_rows": 20,
    }


def test_season_coverage_rejects_missing_team_even_with_matching_game_count(monkeypatch):
    import src.normalize as normalize
    monkeypatch.setitem(normalize.EXPECTED_GAMES, 2025, 2)
    frame = mini_schedule()
    with pytest.raises(ValueError, match="exactly two teams"):
        normalize.validate_season_coverage(frame[frame.team_abbreviation.eq("BOS")], 2025)


def test_season_coverage_rejects_team_aggregate_or_truncated_player_table(monkeypatch):
    import src.normalize as normalize
    monkeypatch.setitem(normalize.EXPECTED_GAMES, 2025, 2)
    frame = mini_schedule()
    with pytest.raises(ValueError, match="at least five player appearances"):
        normalize.validate_season_coverage(frame[~frame.player_id.isin([5, 10])], 2025)


def test_season_coverage_counts_only_regular_season_and_handles_shortened_seasons(monkeypatch):
    import src.normalize as normalize
    assert normalize.EXPECTED_GAMES[2012] == 990
    assert normalize.EXPECTED_GAMES[2013] == 1229
    assert normalize.EXPECTED_GAMES[2020] == 1059
    assert normalize.EXPECTED_GAMES[2021] == 1080
    monkeypatch.setitem(normalize.EXPECTED_GAMES, 2025, 2)
    frame = mini_schedule()
    playoff = frame.copy()
    playoff["game_id"] = playoff.game_id.str.replace("002", "004", n=1)
    playoff["season_type"] = "playoffs"
    result = normalize.validate_season_coverage(pd.concat([frame, playoff], ignore_index=True), 2025)
    assert result == {"source_game_count": 2, "source_player_rows": 20}


def test_adp_rank_must_follow_actual_market_price(players):
    # Both keys and ranks are unique, but reversing price order is still invalid.
    reversed_prices = pd.DataFrame([
        market(canonical_player_id="nba:1", adp=1, adp_rank=200),
        market(canonical_player_id="nba:2", adp=200, adp_rank=1),
    ])
    with pytest.raises(ValueError, match="nondecreasing in adp_rank"):
        normalize_adp(reversed_prices, players)


def test_distinct_games_same_player_date_cannot_collapse_even_with_identical_stats(players, box):
    first_game = {**box, "game_id": "0022400001"}
    second_game = {**box, "game_id": "0022400002"}
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        normalize_game_logs(pd.DataFrame([first_game, second_game]), players)
