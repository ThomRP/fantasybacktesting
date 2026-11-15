"""Chronological All-Star surge signal and following-season ADP rank outcomes."""
from datetime import date
import hashlib
import json

import pandas as pd
import pytest

from src.backtest import run_backtest, load_data
from src.config import SCORING


def logs_for(player, season, pre, post):
    dates = [f'{season}-02-01', f'{season}-02-02'][:len(pre)] + [f'{season}-03-01', f'{season}-03-02'][:len(post)]
    return [{'season':season,'game_date':date.fromisoformat(day),'canonical_player_id':player,
             'full_name':player,'minutes':0.0 if points == 1 else 30.0,'fantasy_points':float(points)}
            for day,points in zip(dates,[*pre,*post])]


@pytest.fixture
def data():
    games = pd.DataFrame([
        *logs_for('nba:A',2024,[10,10],[15,15]),
        *logs_for('nba:B',2024,[10,10],[10,10]),
        *logs_for('nba:C',2024,[10,10],[15,15]),
        *logs_for('nba:D',2024,[10],[15,15]),
        *logs_for('nba:A',2025,[1,1],[]),
        *logs_for('nba:B',2025,[20,20],[]),
        *logs_for('nba:undrafted',2025,[30,30],[]),
    ])
    adp = pd.DataFrame([{'season':2025,'canonical_player_id':f'nba:{name}','full_name':f'nba:{name}',
                         'adp':float(i),'adp_rank':i,'available_at':date(2024,10,1),'timing_status':'preseason'}
                        for i,name in enumerate('ABCD',1)])
    return games,adp


def player(report, name, season=2025):
    matches = report['players'].loc[lambda df:df.canonical_player_id.eq(f'nba:{name}') & df.season.eq(season)]
    assert len(matches) == 1
    return matches.iloc[0]


def test_surge_is_prior_season_and_rank_underperformance_uses_all_nba_players(data):
    games,adp = data
    report = run_backtest(games,adp,min_games=2)
    a,b,c,d = [player(report,p) for p in 'ABCD']
    assert a.signal_season == 2024 and a.season == 2025
    assert a.pre_games == a.post_games == 2
    assert a.pre_ppg == 10 and a.post_ppg == 15
    assert a.surge_pct == pytest.approx(.5)
    assert bool(a.is_surge) and not bool(b.is_surge)
    assert a.actual_total == 2 and a.actual_ppg == 1
    assert a.total_rank == 3 and a.total_rank_delta == 2
    assert bool(a.underperformed_total)
    assert b.total_rank == 2 and b.total_rank_delta == 0
    assert not bool(b.underperformed_total)
    assert d.status == 'insufficient_prior_games'
    assert report['summary']['n_graded'] == 3
    assert report['summary']['n_surge'] == 2
    assert report['summary']['n_reference'] == 1
    assert report['summary']['underperform_total_rate_surge'] == 100
    assert report['summary']['underperform_total_rate_reference'] == 0


def test_zero_game_drafted_player_is_zero_total_and_unknown_ppg(data):
    games,adp = data
    c = player(run_backtest(games,adp,min_games=2),'C')
    assert c.status == 'graded'
    assert c.actual_total == 0
    assert pd.isna(c.actual_ppg) and pd.isna(c.ppg_rank) and pd.isna(c.ppg_rank_delta)
    assert c.total_rank == 4 and c.total_rank_delta == 1
    assert bool(c.underperformed_total)
    assert pd.isna(c.underperformed_ppg)


def test_rounded_zero_minute_appearances_still_count_toward_ppg(data):
    games,adp = data
    a = player(run_backtest(games,adp,min_games=2),'A')
    assert a.actual_ppg == 1
    assert a.ppg_rank == 3


def test_future_outcomes_cannot_change_the_surge_signal(data):
    games,adp = data
    first = run_backtest(games,adp,min_games=2)
    altered = games.copy()
    altered.loc[altered.season.eq(2025),'fantasy_points'] *= 100
    second = run_backtest(altered,adp,min_games=2)
    columns = ['season','signal_season','canonical_player_id','pre_games','post_games','pre_ppg','post_ppg','surge_pct','is_surge']
    pd.testing.assert_frame_equal(first['players'].loc[lambda df:df.season.eq(2025),columns],
                                  second['players'].loc[lambda df:df.season.eq(2025),columns])
    assert player(second,'A').actual_total == 200


def test_exact_threshold_counts_as_surge(data):
    games,adp = data
    games.loc[games.season.eq(2024) & games.canonical_player_id.eq('nba:A') & games.game_date.gt(date(2024,2,18)),'fantasy_points'] = 12
    a = player(run_backtest(games,adp,min_games=2,surge_threshold=.2),'A')
    assert a.surge_pct == pytest.approx(.2)
    assert bool(a.is_surge)


def test_all_star_boundary_is_in_pre_window_and_following_day_in_post_window(data):
    games,adp = data
    mask = games.season.eq(2024) & games.canonical_player_id.eq('nba:A')
    games.loc[mask,'game_date'] = [date(2024,2,17),date(2024,2,18),date(2024,2,19),date(2024,2,20)]
    a = player(run_backtest(games,adp,min_games=2),'A')
    assert a.pre_games == 2 and a.post_games == 2
    assert a.pre_ppg == 10 and a.post_ppg == 15


@pytest.mark.parametrize('status,available',[
    ('opening_day',date(2024,10,22)),('after_opener',date(2024,10,23)),('unknown',None)])
def test_unsafe_adp_snapshots_excluded_by_default(data,status,available):
    games,adp = data
    adp.loc[adp.canonical_player_id.eq('nba:A'),['timing_status','available_at']] = [status,available]
    a = player(run_backtest(games,adp,min_games=2),'A')
    assert a.status == 'unsafe_adp'
    assert pd.isna(a.total_rank_delta) and pd.isna(a.ppg_rank_delta)


def test_opening_day_sensitivity_is_explicit_and_does_not_admit_after_opener(data):
    games,adp = data
    adp.loc[adp.canonical_player_id.eq('nba:A'),['timing_status','available_at']] = ['opening_day',date(2024,10,22)]
    adp.loc[adp.canonical_player_id.eq('nba:B'),['timing_status','available_at']] = ['after_opener',date(2024,10,23)]
    report = run_backtest(games,adp,min_games=2,include_opening_day=True)
    assert player(report,'A').status == 'graded'
    assert player(report,'B').status == 'unsafe_adp'


def test_missing_adp_and_missing_entire_outcome_season_are_not_zero_returns(data):
    games,adp = data
    missing_price = adp.loc[~adp.canonical_player_id.eq('nba:A')]
    assert player(run_backtest(games,missing_price,min_games=2),'A').status == 'missing_adp'
    prior_only = games.loc[games.season.eq(2024)]
    a = player(run_backtest(prior_only,adp,min_games=2),'A')
    assert a.status == 'missing_outcomes'
    assert pd.isna(a.actual_total)


def test_nonpositive_pre_ppg_does_not_fabricate_a_percentage_surge(data):
    games,adp = data
    games.loc[games.season.eq(2024) & games.canonical_player_id.eq('nba:A') & games.game_date.le(date(2024,2,18)),'fantasy_points'] = 0
    a = player(run_backtest(games,adp,min_games=2),'A')
    assert a.status == 'nonpositive_pre_ppg'
    assert pd.isna(a.surge_pct)


def test_first_dataset_season_cannot_use_unobserved_prior_year(data):
    games,adp = data
    first_market = adp.iloc[[0]].copy()
    first_market['season'] = 2015
    first_market['available_at'] = date(2014,10,1)
    report = run_backtest(games,pd.concat([adp,first_market],ignore_index=True),min_games=2)
    assert player(report,'A',season=2015).status == 'missing_prior_season'


@pytest.mark.parametrize('duplicate_target',['games','adp'])
def test_duplicate_keys_raise_instead_of_double_counting(data,duplicate_target):
    games,adp = data
    if duplicate_target == 'games': games = pd.concat([games,games.iloc[[0]]],ignore_index=True)
    else: adp = pd.concat([adp,adp.iloc[[0]]],ignore_index=True)
    with pytest.raises(ValueError,match='duplicate|Duplicate'):
        run_backtest(games,adp,min_games=2)


@pytest.mark.parametrize('kwargs',[{'min_games':0},{'min_games':1.5},{'surge_threshold':-.1},{'surge_threshold':float('nan')}])
def test_invalid_parameters_raise(data,kwargs):
    with pytest.raises(ValueError):
        run_backtest(*data,**kwargs)


def write_processed(tmp_path,data):
    games,adp = data
    files = {}
    for name,frame in [('game_logs',games),('adp',adp)]:
        path = tmp_path / f'{name}.parquet'
        frame.to_parquet(path,index=False)
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {'seasons':[2024,2025],'scoring':SCORING,'files':files}
    (tmp_path/'metadata.json').write_text(json.dumps(metadata))
    return metadata


def test_disk_loading_rejects_missing_manifest_wrong_scoring_and_corrupt_files(tmp_path,data):
    write_processed(tmp_path,data)
    games,adp = load_data(tmp_path)
    assert len(games) == len(data[0]) and len(adp) == len(data[1])
    manifest = tmp_path/'metadata.json'
    metadata = json.loads(manifest.read_text())
    metadata['scoring'] = {'pts':999}
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='scoring|Scoring'):
        load_data(tmp_path)
    write_processed(tmp_path,data)
    path = tmp_path/'adp.parquet'
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='checksum|SHA|hash'):
        load_data(tmp_path)
    manifest.unlink()
    with pytest.raises((ValueError,FileNotFoundError)):
        load_data(tmp_path)


def test_unresolved_adp_player_is_reported_not_treated_as_nba_no_show(data):
    games,adp = data
    unknown = pd.DataFrame([{'season':2025,'canonical_player_id':None,'full_name':'Unknown Prospect',
                             'adp':5.,'adp_rank':5,'available_at':date(2024,10,1),'timing_status':'preseason',
                             'match_status':'unresolved'}])
    market = adp.assign(match_status='matched')
    report = run_backtest(games,pd.concat([market,unknown],ignore_index=True),min_games=2)
    assert report['exclusions'].status.eq('unresolved_adp').sum() == 1
    assert report['summary']['n_graded'] == 3


def test_claimed_preseason_marker_cannot_hide_late_snapshot_date(data):
    games,adp = data
    adp.loc[adp.canonical_player_id.eq('nba:A'),'available_at'] = date(2025,1,1)
    assert player(run_backtest(games,adp,min_games=2),'A').status == 'unsafe_adp'


def test_cli_writes_only_flagged_surge_players_and_strict_json(tmp_path,data):
    from src.backtest import main
    write_processed(tmp_path,data)
    output = tmp_path / 'reports'
    assert main(['--processed-dir',str(tmp_path),'--output-dir',str(output),'--min-games','2']) == 0
    players = pd.read_csv(output/'surge_players.csv')
    assert set(players.canonical_player_id) == {'nba:A','nba:C'}
    assert players.is_surge.all()
    summary = json.loads((output/'summary.json').read_text(), parse_constant=lambda x: pytest.fail(f'Invalid JSON constant {x}'))
    assert summary['n_surge'] == 2 and summary['n_reference'] == 1
    assert 'reference' in set(pd.read_csv(output/'season_summary.csv').cohort)
