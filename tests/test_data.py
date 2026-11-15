"""The two source tables must preserve dates, AVG prices and honest availability."""
from datetime import date, datetime, timezone
import json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src import data


def games_raw():
    rows=[]
    for team,opponent,offset in [('BOS','NYK',0),('NYK','BOS',5)]:
        for i in range(1,6):
            stats=dict.fromkeys((*data.BOX_STATS,*data.SHOOTING_RATES),0.)
            stats.update(fgm=8,fga=18,ftm=5,fta=6,fg3m=3,reb=10,ast=7,stl=2,blk=1,tov=4,pts=24)
            rows.append({'season':2018,'season_type':'regular-season','game_id':'0021700001','game_date':'2017-10-18',
                         'player_id':i+offset,'player_name':f'Player {i+offset}','team_abbreviation':team,
                         'matchup':f'{team} vs. {opponent}','wl':'W' if team=='BOS' else 'L','min':0 if i==1 else '30:30',**stats})
    return pd.DataFrame(rows)


def normalized(monkeypatch):
    monkeypatch.setitem(data.EXPECTED_GAMES,2018,1)
    return data.normalize_games(games_raw(),2018)


def page(*,season=2018,providers=('Yahoo','ESPN'),updates=None,quotes=None,names=('Player 1','Player 2'),averages=None):
    updates=updates or {p:'2017-10-10' for p in providers}
    quotes=quotes or [{p:float(i) for p in providers} for i in [1,2]]
    averages=averages or [sum(v for v in row.values() if v is not None)/sum(v is not None for v in row.values()) for row in quotes]
    headers=''.join(f'<th>{p}</th>' for p in providers)
    rows=''
    for i,(name,q,avg) in enumerate(zip(names,quotes,averages),1):
        cells=''.join(f'<td>{q.get(p) if q.get(p) is not None else "-"}</td>' for p in providers)
        rows+=f'<tr><td>{i}</td><td><a fp-player-name="{name}">{name}</a><small>(CURRENT - C) injured</small></td>{cells}<td>{avg}</td>'
    experts=''
    for provider,day in updates.items():
        stamp=int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp()) if day else 0
        experts+=f'<tr><td>ADP</td><td><a>{provider}</a></td><td><span>{stamp}</span>10/09</td></tr>'
    return (f'<title>Average Draft Position (ADP) - Fantasy Basketball {data.season_label(season)} | FantasyPros</title>'
            f'<table id="data"><tr><th>Rank</th><th>Player</th>{headers}<th>AVG</th></tr>{rows}</table>'
            f'<table id="experts">{experts}</table>').encode()


def test_exact_scoring_dates_and_real_zero_minute_appearances(monkeypatch,tmp_path):
    frame=normalized(monkeypatch)
    assert len(frame)==10 and frame.fantasy_points.eq(52.).all()
    assert frame.minutes.eq(0).sum()==2
    assert not frame.duplicated(['season','game_date','canonical_player_id']).any()
    path=tmp_path/'game_logs.parquet'
    data._write_table(frame,path,data.GAME_SCHEMA)
    assert str(pq.read_schema(path).field('game_date').type)=='date32[day]'
    assert pd.read_parquet(path).game_date.iloc[0]==date(2017,10,18)


def test_playoffs_excluded_benign_repeats_deduped_and_trades_same_id(monkeypatch):
    monkeypatch.setitem(data.EXPECTED_GAMES,2018,2)
    first=games_raw()
    second=first.copy()
    second['game_id']='0021700002';second['game_date']='2017-10-20'
    second.loc[second.player_id.eq(1),'team_abbreviation']='NYK'
    second.loc[second.player_id.eq(1),'matchup']='NYK vs. BOS'
    # A regular two-team source still has >=five distinct appearances per team.
    second.loc[second.player_id.eq(6),'team_abbreviation']='BOS'
    second.loc[second.player_id.eq(6),'matchup']='BOS vs. NYK'
    playoff=first.assign(season_type='playoffs',game_date='2018-05-01',game_id='0041700001')
    frame=data.normalize_games(pd.concat([first,first,second,playoff]),2018)
    assert len(frame)==20
    assert frame[frame.canonical_player_id.eq('nba:1')].team.tolist()==['BOS','NYK']


@pytest.mark.parametrize('problem',['partial','missing_date','missing_stat','conflict','wrong_season','missing_game_id'])
def test_broken_source_never_becomes_a_complete_season(monkeypatch,problem):
    monkeypatch.setitem(data.EXPECTED_GAMES,2018,1)
    frame=games_raw()
    if problem=='partial': frame=frame.iloc[:5]
    if problem=='missing_date': frame.loc[0,'game_date']=None
    if problem=='missing_stat': frame.loc[0,'stl']=None
    if problem=='conflict': frame=pd.concat([frame,frame.iloc[[0]].assign(pts=100)])
    if problem=='wrong_season': frame['season']=2019
    if problem=='missing_game_id': frame.loc[0,'game_id']=None
    with pytest.raises(ValueError): data.normalize_games(frame,2018)


def test_avg_not_espn_or_site_rank_and_current_positions_ignored():
    payload=page(quotes=[{'Yahoo':20,'ESPN':10},{'Yahoo':2,'ESPN':4}],averages=[15,3])
    frame,evidence=data.parse_adp(payload,2018)
    assert frame.adp.tolist()==[3,15]
    assert frame.full_name.tolist()==['Player 2','Player 1']
    assert frame.source_rank.tolist()==[2,1] and frame.adp_rank.tolist()==[1,2]
    assert 'position' not in frame and frame.timing_status.eq('preseason').all()
    assert evidence['provider_dates_displayed']['Yahoo']=='10/09'


def test_wrong_year_fallback_rejected():
    with pytest.raises(ValueError,match='Wrong'):
        data.parse_adp(page(season=2026),2018)


def test_non_scoring_box_stats_retained_including_negative_plus_minus(monkeypatch):
    monkeypatch.setitem(data.EXPECTED_GAMES,2018,1)
    raw=games_raw()
    raw['plus_minus']=-5
    raw['oreb']=3
    raw['fg3a']=7
    raw['fg_pct']=float('nan')
    result=data.normalize_games(raw,2018)
    assert result.plus_minus.eq(-5).all() and result.oreb.eq(3).all() and result.fg3a.eq(7).all()
    assert result.fg_pct.isna().all() and set(result.result)=={'W','L'}


def test_consensus_rank_availability_uses_entire_market_not_only_own_quote():
    payload=page(season=2025,updates={'Yahoo':'2025-07-22','ESPN':'2024-10-21'},
                 quotes=[{'Yahoo':None,'ESPN':1},{'Yahoo':3,'ESPN':3}])
    frame,_=data.parse_adp(payload,2025)
    assert frame.iloc[0].contributors=='ESPN'
    assert frame.timing_status.eq('after_opener').all()
    assert frame.available_at.eq(date(2025,7,22)).all()


@pytest.mark.parametrize('updated,status',[('2017-10-17','opening_day'),('2017-10-18','after_opener')])
def test_opening_day_and_late_prices_retained_but_labeled(updated,status):
    frame,_=data.parse_adp(page(updates={'Yahoo':updated,'ESPN':updated}),2018)
    assert len(frame)==2 and frame.timing_status.eq(status).all()


def test_undated_provider_does_not_become_false_preseason():
    payload=page(updates={'Yahoo':'2017-10-10','ESPN':None})
    frame,_=data.parse_adp(payload,2018)
    # Equal prices cannot establish that an undated provider is inactive.
    assert frame.available_at.isna().all() and frame.timing_status.eq('unknown').all()


def test_stale_cbs_is_ignored_only_when_published_avg_verifies_exclusion():
    payload=page(providers=('Yahoo','ESPN','CBS'),updates={'Yahoo':'2017-10-10','ESPN':'2017-10-10','CBS':None},
                 quotes=[{'Yahoo':1,'ESPN':3,'CBS':99},{'Yahoo':3,'ESPN':5,'CBS':88}],averages=[2,4])
    frame,evidence=data.parse_adp(payload,2018)
    assert evidence['inactive_columns']==['CBS']
    assert frame.contributors.eq('ESPN,Yahoo').all() and frame.timing_status.eq('preseason').all()


def test_nfbc_header_and_source_name_aliases_share_date():
    frame,_=data.parse_adp(page(providers=('Yahoo','NFBCK'),updates={'Yahoo':'2017-10-10','NFC':'2017-10-11'}),2018)
    assert frame.contributors.eq('NFBC,Yahoo').all()
    assert frame.available_at.eq(date(2017,10,11)).all()


def test_unresolved_prices_retained_and_suffixes_distinguish_players(monkeypatch):
    frame=normalized(monkeypatch)
    frame.loc[0,['full_name','canonical_player_id']]=['Gary Payton','nba:56']
    frame.loc[1,['full_name','canonical_player_id']]=['Gary Payton II','nba:1627780']
    raw,_=data.parse_adp(page(names=('Gary Payton II','Never Played')),2018)
    prices=data.resolve_adp(raw,frame)
    assert len(prices)==2
    assert prices.iloc[0].canonical_player_id=='nba:1627780'
    assert prices.iloc[1].match_status=='unresolved' and pd.isna(prices.iloc[1].canonical_player_id)
    assert prices.adp_rank.tolist()==[1,2]


def test_manual_aliases_must_use_a_known_nba_id(monkeypatch):
    frame=normalized(monkeypatch)
    raw,_=data.parse_adp(page(names=('Known Alias','Player 2')),2018)
    aliases=pd.DataFrame([{'adp_name':'Known Alias','canonical_player_id':'nba:1'}])
    assert data.resolve_adp(raw,frame,aliases).iloc[0].canonical_player_id=='nba:1'
    with pytest.raises(ValueError,match='unknown'):
        data.resolve_adp(raw,frame,aliases.assign(canonical_player_id='nba:999'))


def test_typed_adp_roundtrip_preserves_missing_identity_and_date(monkeypatch,tmp_path):
    raw,_=data.parse_adp(page(names=('Player 1','Unknown'),updates={'Yahoo':None,'ESPN':None}),2018)
    frame=data.resolve_adp(raw,normalized(monkeypatch))
    path=tmp_path/'adp.parquet'
    data._write_table(frame,path,data.ADP_SCHEMA)
    result=pd.read_parquet(path)
    assert len(result)==2 and result.available_at.isna().all()
    assert pd.isna(result.iloc[1].canonical_player_id)
