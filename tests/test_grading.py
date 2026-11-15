"""Synthetic behavioral checks for point-in-time benchmark and cohort grading."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.benchmark import (
    aggregate_outcomes,
    fit_chronological_curve,
    fit_expected_curve,
)
from src.grader import evaluate_cohort


def market(*seasons):
    return pd.DataFrame([
        {"season": season, "canonical_player_id": player, "adp": float(rank),
         "adp_rank": rank, "position_rank": rank}
        for season in seasons
        for player, rank in [("nba:1", 1), ("nba:2", 2)]
    ])


def logs(*seasons):
    return pd.DataFrame([
        {"season": season, "game_date": pd.Timestamp(f"{season}-01-01").date(),
         "canonical_player_id": player, "fantasy_points": points,
         "minutes_or_snaps": 30.0}
        for season in seasons
        for player, points in [("nba:1", 20.0), ("nba:2", 10.0)]
    ])


def cohort(*seasons):
    return market(*seasons)[["season", "canonical_player_id"]]


def test_curve_has_every_draft_slot_and_monotone_predictions():
    outcomes = pd.DataFrame({
        "season": [2007, 2007, 2008, 2008],
        "canonical_player_id": ["nba:1", "nba:2"] * 2,
        "adp_rank": [1, 200, 1, 200],
        "actual_total_points": [90.0, 120.0, 110.0, 80.0],
        "actual_ppg": [30.0, 40.0, 35.0, 25.0],
    })
    curve = fit_expected_curve(outcomes)
    assert curve["adp_rank"].tolist() == list(range(1, 201))
    assert np.isfinite(curve[["expected_total_points", "expected_ppg"]]).all().all()
    assert (curve["expected_total_points"].diff().dropna() <= 0).all()
    assert (curve["expected_ppg"].diff().dropna() <= 0).all()


def test_current_and_future_outcomes_do_not_change_historical_curve():
    game_logs, adp = logs(2007, 2008, 2009), market(2007, 2008, 2009)
    original = fit_chronological_curve(game_logs, adp, 2008)
    changed = game_logs.copy()
    changed.loc[changed["season"] >= 2008, "fantasy_points"] = 999999.0
    assert_frame_equal(original, fit_chronological_curve(changed, adp, 2008))


def test_holdout_curve_is_frozen_before_first_holdout_season():
    game_logs = logs(2021, 2022, 2023, 2024, 2025, 2026)
    adp = market(2021, 2022, 2023, 2024, 2025, 2026)
    game_logs.loc[game_logs["season"] >= 2023, "fantasy_points"] = 100000.0
    first = fit_chronological_curve(game_logs, adp, 2023)
    latest = fit_chronological_curve(game_logs, adp, 2026)
    assert_frame_equal(first, latest)
    assert first.loc[first["adp_rank"] == 1, "expected_total_points"].item() == 20.0


def test_absent_player_is_zero_total_but_absent_season_is_unknown():
    game_logs = logs(2007)
    adp = pd.concat([market(2007, 2008), pd.DataFrame([{
        "season": 2007, "canonical_player_id": "nba:injured", "adp": 3.0,
        "adp_rank": 3, "position_rank": 3,
    }])], ignore_index=True)
    outcomes = aggregate_outcomes(game_logs, adp)
    absent = outcomes.loc[outcomes["canonical_player_id"] == "nba:injured"].iloc[0]
    assert absent["actual_total_points"] == 0
    assert absent["games_played"] == 0
    assert pd.isna(absent["actual_ppg"])
    unknown = outcomes.loc[outcomes["season"] == 2008]
    assert unknown["actual_total_points"].isna().all()
    assert unknown["actual_ppg"].isna().all()


def test_grader_computes_alpha_and_explicit_ppg_denominator():
    game_logs, adp = logs(2007, 2008), market(2007, 2008)
    game_logs.loc[(game_logs["season"] == 2008) &
                  (game_logs["canonical_player_id"] == "nba:1"), "fantasy_points"] = 30.0
    game_logs = game_logs.loc[~((game_logs["season"] == 2008) &
                              (game_logs["canonical_player_id"] == "nba:2"))]
    report = evaluate_cohort(cohort(2008), game_logs=game_logs, adp=adp)
    summary = report["summary"]
    assert summary["n_cohort"] == 2
    assert summary["n_graded"] == 2
    assert summary["n_ppg"] == 1
    assert summary["mean_alpha_total"] == 0.0  # +10 and -10
    assert summary["median_alpha_total"] == 0.0
    assert summary["mean_alpha_ppg"] == 10.0
    assert summary["beat_adp_rate"] == 50.0


def test_grader_exposes_missing_adp_and_no_training():
    rows = pd.concat([cohort(2007, 2008), pd.DataFrame([
        {"season": 2008, "canonical_player_id": "nba:unknown"}
    ])], ignore_index=True)
    report = evaluate_cohort(rows, game_logs=logs(2007, 2008), adp=market(2007, 2008))
    results = report["player_results"]
    assert len(results) == 5
    assert set(results.loc[results["season"] == 2007, "status"]) == {"no_training"}
    assert results.loc[results["canonical_player_id"] == "nba:unknown", "status"].item() == "missing_adp"
    assert report["summary"]["n_graded"] == 2
    assert len(report["exclusions"]) == 3


def test_grader_distinguishes_missing_season_from_zero_games():
    report = evaluate_cohort(cohort(2008), game_logs=logs(2007), adp=market(2007, 2008))
    assert report["summary"]["n_graded"] == 0
    assert set(report["player_results"]["status"]) == {"missing_season"}


def test_grader_separates_holdout_and_five_year_eras():
    seasons = (2007, 2010, 2011, 2022, 2023, 2026)
    report = evaluate_cohort(cohort(2010, 2011, 2023, 2026), game_logs=logs(*seasons), adp=market(*seasons))
    assert set(report["split_breakdown"]["split"]) == {"development", "holdout"}
    assert report["split_breakdown"].set_index("split").loc["holdout", "n_graded"] == 4
    assert len(report["era_breakdown"]) == 4


def test_duplicate_cohort_is_rejected_instead_of_weighting_twice():
    rows = pd.concat([cohort(2008), cohort(2008)], ignore_index=True)
    with pytest.raises(ValueError):
        evaluate_cohort(rows, game_logs=logs(2007, 2008), adp=market(2007, 2008))


def test_duplicate_adp_is_rejected_instead_of_multiplying_join():
    adp = pd.concat([market(2007), market(2007)], ignore_index=True)
    with pytest.raises(ValueError):
        aggregate_outcomes(logs(2007), adp)


@pytest.mark.parametrize("bad_rank", [0, -1, 1.5, np.nan])
def test_invalid_rank_is_rejected(bad_rank):
    adp = market(2007)
    adp["adp_rank"] = adp["adp_rank"].astype(float)
    adp.loc[0, "adp_rank"] = bad_rank
    with pytest.raises(ValueError):
        aggregate_outcomes(logs(2007), adp)


@pytest.mark.parametrize("bad_season", [2006, 2027, 2007.5])
def test_cohort_season_outside_configured_window_is_rejected(bad_season):
    rows = pd.DataFrame({"season": [bad_season], "canonical_player_id": ["nba:1"]})
    with pytest.raises(ValueError):
        evaluate_cohort(rows, game_logs=logs(2007), adp=market(2007))


def test_null_cohort_player_id_is_rejected():
    rows = pd.DataFrame({"season": [2008], "canonical_player_id": [None]})
    with pytest.raises(ValueError):
        evaluate_cohort(rows, game_logs=logs(2007, 2008), adp=market(2007, 2008))


def test_disk_grading_requires_complete_coverage_manifest(tmp_path):
    logs(2007, 2008).to_parquet(tmp_path / 'game_logs.parquet')
    market(2007, 2008).to_parquet(tmp_path / 'adp.parquet')
    with pytest.raises(ValueError, match='coverage.json'):
        evaluate_cohort(cohort(2008), processed_dir=tmp_path)


def test_disk_manifest_checks_completed_seasons_scoring_and_rowcount(tmp_path):
    import json
    from src.config import SCORING
    games = logs(2007, 2008)
    games.to_parquet(tmp_path / 'game_logs.parquet')
    market(2007, 2008).to_parquet(tmp_path / 'adp.parquet')
    metadata = {'completed_seasons': [2007], 'scoring': SCORING, 'game_logs': len(games)}
    path = tmp_path / 'coverage.json'
    path.write_text(json.dumps(metadata))
    incomplete = evaluate_cohort(cohort(2008), processed_dir=tmp_path)
    assert incomplete['summary']['n_graded'] == 0
    assert set(incomplete['player_results']['status']) == {'incomplete_season'}
    metadata['completed_seasons'] = [2007, 2008]
    metadata['scoring'] = {'pts': 100}
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='scoring'):
        evaluate_cohort(cohort(2008), processed_dir=tmp_path)
    metadata['scoring'] = SCORING
    metadata['game_logs'] = 0
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='count'):
        evaluate_cohort(cohort(2008), processed_dir=tmp_path)
    metadata['game_logs'] = len(games)
    path.write_text(json.dumps(metadata))
    report = evaluate_cohort(cohort(2008), processed_dir=tmp_path)
    assert report['summary']['n_graded'] == 2


def test_zero_minute_appearance_counts_in_ppg_denominator():
    games = logs(2007)
    games.loc[games.canonical_player_id.eq('nba:1'), 'minutes_or_snaps'] = 0.0
    result = aggregate_outcomes(games, market(2007))
    appearance = result.loc[result.canonical_player_id.eq('nba:1')].iloc[0]
    assert appearance.games_played == 1
    assert appearance.actual_ppg == 20.0


def test_disk_grader_rejects_stale_parquet_hash(tmp_path):
    import hashlib
    import json
    from src.config import SCORING
    games = logs(2007, 2008)
    games.to_parquet(tmp_path / 'game_logs.parquet')
    market(2007, 2008).to_parquet(tmp_path / 'adp.parquet')
    metadata = {'completed_seasons': [2007, 2008], 'scoring': SCORING,
                'game_logs': len(games), 'processed_sha256': {
                    stem: hashlib.sha256((tmp_path / f'{stem}.parquet').read_bytes()).hexdigest()
                    for stem in ['game_logs', 'adp']}}
    (tmp_path / 'coverage.json').write_text(json.dumps(metadata))
    assert evaluate_cohort(cohort(2008), processed_dir=tmp_path)['summary']['n_graded'] == 2
    changed = market(2007, 2008)
    changed.loc[changed.season.eq(2008), 'adp_rank'] += 5
    changed.to_parquet(tmp_path / 'adp.parquet')
    with pytest.raises(ValueError, match='SHA256'):
        evaluate_cohort(cohort(2008), processed_dir=tmp_path)


def test_example_cohort_never_uses_upcoming_season_outcomes():
    from experiments.template import demo_data, make_cohort
    games, adp = demo_data()
    expected = make_cohort(games, adp, 2023)
    games.loc[games.season.eq(2023), 'fantasy_points'] = -1e9
    assert_frame_equal(make_cohort(games, adp, 2023), expected)


def test_benchmark_explicit_completion_excludes_quarantined_season():
    games, prices = logs(2007, 2008, 2009), market(2007, 2008, 2009)
    games.loc[games.season.eq(2008), 'fantasy_points'] = 1e9
    outcomes = aggregate_outcomes(games, prices, completed_seasons=[2007, 2009])
    quarantined = outcomes.loc[outcomes.season.eq(2008)]
    assert quarantined.actual_total_points.isna().all()
    assert quarantined.actual_ppg.isna().all()
    assert not quarantined.season_available.any()
    curve = fit_chronological_curve(games, prices, 2009, completed_seasons=[2007, 2009])
    expected = fit_chronological_curve(games.loc[games.season.ne(2008)], prices, 2009)
    assert_frame_equal(curve, expected)
    assert curve.attrs['training_seasons'] == [2007]


def test_benchmark_rejects_rank_order_inconsistent_with_market_prices():
    prices = market(2007)
    prices['adp_rank'] = [200, 1]
    with pytest.raises(ValueError, match='inconsistent'):
        aggregate_outcomes(logs(2007), prices)


def test_benchmark_allows_tied_prices_with_distinct_ranks():
    prices = market(2007)
    prices['adp'] = [2.0, 2.0]
    assert len(aggregate_outcomes(logs(2007), prices)) == 2
