"""Prepare two NBA tables: dated game logs and FantasyPros consensus ADP."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import logging
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import pyarrow as pa

from .cache import CachedDownloader
from .config import ROOT, RAW_DIR, PROCESSED_DIR, SEASONS, SCORING, BOX_STATS, SHOOTING_RATES, SEASON_OPENERS, EXPECTED_GAMES, season_label

LOG = logging.getLogger(__name__)
MIRROR = 'https://github.com/sportsdataverse/sportsdataverse-data/releases/download/nba_stats_player_game_logs'
GAME_SCHEMA = pa.schema([
    ('season', pa.int64()), ('game_date', pa.date32()), ('game_id', pa.string()),
    ('canonical_player_id', pa.string()), ('full_name', pa.string()),
    ('team', pa.string()), ('opponent', pa.string()), ('result', pa.string()), ('minutes', pa.float64()),
    *[(stat, pa.float64()) for stat in (*BOX_STATS,*SHOOTING_RATES)], ('fantasy_points', pa.float64()),
])
ADP_SCHEMA = pa.schema([
    ('season', pa.int64()), ('canonical_player_id', pa.string()), ('full_name', pa.string()),
    ('adp', pa.float64()), ('adp_rank', pa.int64()), ('source_rank', pa.int64()),
    ('available_at', pa.date32()), ('timing_status', pa.string()),
    ('match_status', pa.string()), ('contributors', pa.string()), ('source', pa.string()),
])


def name_key(value, *, suffixes=False):
    text = unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode().lower()
    text = ' '.join(re.sub(r'[^a-z0-9\s]', '', text).split())
    return text if suffixes else re.sub(r'(?:\s+(?:jr|sr|ii|iii|iv|v))+$', '', text)


def score_games(frame):
    stats = frame[list(SCORING)].apply(pd.to_numeric, errors='raise')
    if stats.isna().any().any() or not np.isfinite(stats.to_numpy()).all() or stats.lt(0).any().any():
        raise ValueError('Box-score stats must be finite, nonnegative and complete')
    return stats.mul(pd.Series(SCORING)).sum(axis=1).astype(float)


def normalize_games(raw, season):
    frame = raw.rename(columns=str.lower).copy()
    required = {'season_type', 'game_id', 'game_date', 'player_id', 'player_name', 'team_abbreviation', 'matchup', 'min', 'wl', *BOX_STATS, *SHOOTING_RATES}
    if not required <= set(frame):
        raise ValueError(f'Season {season}: missing player-log columns {required-set(frame)}')
    if not frame.season_type.isin(['regular-season', 'playoffs']).all():
        raise ValueError('Unexpected NBA season type')
    frame = frame.loc[frame.season_type.eq('regular-season')].drop_duplicates().copy()
    if frame[['game_id','team_abbreviation']].isna().any().any():
        raise ValueError('Missing game/team keys')
    if 'season' in frame and not pd.to_numeric(frame.season,errors='raise').eq(season).all():
        raise ValueError('Wrong NBA season archive')
    if frame.game_id.nunique() != EXPECTED_GAMES[season]:
        raise ValueError(f'Incomplete season {season}: expected {EXPECTED_GAMES[season]} regular-season games')
    if not frame.groupby('game_id').team_abbreviation.nunique().eq(2).all():
        raise ValueError('Each game must contain two teams')
    if frame.groupby(['game_id','team_abbreviation']).player_id.nunique().lt(5).any():
        raise ValueError('Each team must contain at least five player appearances')
    frame['season'] = season
    frame['game_date'] = pd.to_datetime(frame.game_date, format='mixed', errors='raise').dt.date
    if frame.game_date.isna().any() or not frame.game_date.between(SEASON_OPENERS[season], datetime(season,10,15).date()).all():
        raise ValueError('Missing or out-of-season game date')
    ids = pd.to_numeric(frame.player_id, errors='raise')
    if ids.isna().any() or not np.isfinite(ids).all() or ids.le(0).any() or ids.mod(1).ne(0).any():
        raise ValueError('Invalid NBA player IDs')
    frame['canonical_player_id'] = 'nba:' + ids.astype('int64').astype(str)
    frame['full_name'] = frame.player_name.astype('string')
    if frame.full_name.isna().any() or frame.full_name.str.strip().eq('').any():
        raise ValueError('Missing player name')
    match = frame.matchup.str.extract(r'^([A-Z]{2,3})\s+(?:vs\.?|@)\s+([A-Z]{2,3})$')
    if match.isna().any().any() or not match[0].eq(frame.team_abbreviation).all():
        raise ValueError('Malformed matchup')
    frame['team'], frame['opponent'] = match[0], match[1]
    frame['result']=frame.wl
    if not frame.result.isin(['W','L']).all(): raise ValueError('Invalid game result')
    def minutes(value):
        if isinstance(value,str) and ':' in value:
            mins,secs=value.split(':')
            return float(mins)+float(secs)/60
        return float(value)
    frame['minutes'] = frame['min'].map(minutes)
    if not np.isfinite(frame.minutes).all() or frame.minutes.lt(0).any():
        raise ValueError('Invalid minutes')
    # NBA logs contain appearances; rounded zero-minute rows still count for PPG.
    frame['fantasy_points'] = score_games(frame)
    for stat in BOX_STATS:
        frame[stat] = pd.to_numeric(frame[stat],errors='raise').astype(float)
        if not np.isfinite(frame[stat]).all() or (stat!='plus_minus' and frame[stat].lt(0).any()):
            raise ValueError(f'Invalid box-score stat {stat}')
    for rate in SHOOTING_RATES:
        frame[rate]=pd.to_numeric(frame[rate],errors='raise').astype(float)
        if np.isinf(frame[rate]).any() or not frame[rate].dropna().between(0,1).all():
            raise ValueError(f'Invalid shooting rate {rate}')
    frame['game_id'] = frame.game_id.astype(str).str.zfill(10)
    if frame.duplicated(['season','game_date','canonical_player_id']).any():
        raise ValueError('Duplicate player/date keys; inspect source game IDs')
    return frame[GAME_SCHEMA.names].sort_values(['game_date','game_id','canonical_player_id']).reset_index(drop=True)

class _Tables(HTMLParser):
    """Parse only required tables, including the site's omitted closing tr tags."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables, self.table, self.row, self.cell = {}, None, None, None
        self.title, self.in_title = [], False

    def _close_cell(self):
        if self.cell is not None and self.row is not None:
            self.row.append(self.cell)
        self.cell = None

    def _close_row(self):
        self._close_cell()
        if self.row is not None and self.table:
            self.tables[self.table].append(self.row)
        self.row = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "title":
            self.in_title = True
        if tag == "table" and attrs.get("id") in {"data", "experts"}:
            self.table = attrs["id"]
            self.tables[self.table] = []
        if self.table:
            if tag == "tr":
                self._close_row()
                self.row = []
            elif tag in {"td", "th"}:
                self._close_cell()
                self.cell = {"text": "", "name": None}
            elif self.cell is not None and "fp-player-name" in attrs:
                self.cell["name"] = attrs["fp-player-name"]

    def handle_data(self, data):
        if self.in_title:
            self.title.append(data)
        if self.cell is not None:
            self.cell["text"] += data

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        elif tag in {"td", "th"}:
            self._close_cell()
        elif tag == "tr":
            self._close_row()
        elif tag == "table" and self.table:
            self._close_row()
            self.table = None



def _provider(value):
    value = re.sub(r'[^a-z]', '', value.lower())
    if value.startswith('yahoo'): return 'Yahoo'
    if value.startswith('espn'): return 'ESPN'
    if value.startswith('cbs'): return 'CBS'
    if value.startswith(('nfc','nfbc')): return 'NFBC'
    return value


def parse_adp(payload, season):
    """Read AVG as published, retaining late/unknown prices with timing labels.

    UTC sort-key dates are conservative day-level availability estimates. The
    displayed local MM/DD is retained too; neither is an independent capture.
    Historical team/position/injury labels are never used.
    """
    page = _Tables()
    page.feed(payload.decode('utf-8-sig'))
    title = ''.join(page.title)
    if season_label(season) not in title or 'Average Draft Position (ADP)' not in title:
        raise ValueError(f'Wrong FantasyPros archive for {season_label(season)}')
    dates, displayed = {}, {}
    for row in page.tables.get('experts', []):
        if len(row)<3 or not any(cell['text'].strip()=='ADP' for cell in row): continue
        provider = _provider(row[-2]['text'].strip())
        text = row[-1]['text'].strip()
        stamp = re.match(r'(\d{10})', text)
        dates[provider] = datetime.fromtimestamp(int(stamp.group(1)),timezone.utc).date() if stamp else None
        local = re.search(r'\d{1,2}/\d{1,2}', text)
        displayed[provider] = local.group() if local else None
    rows = page.tables.get('data', [])
    if not rows: raise ValueError('No ADP table')
    header = [c['text'].strip() for c in rows[0]]
    if not {'Rank','Player','AVG'} <= set(header): raise ValueError('No consensus AVG column')
    providers = [_provider(h) for h in header if h not in {'Rank','Player','AVG'}]
    parsed = []
    def number(text):
        text = text.strip()
        if text in {'','-','—','–','N/A'}: return None
        value = float(text.replace(',',''))
        if not np.isfinite(value) or value<=0: raise ValueError('ADP must be finite and positive')
        return value
    for row in rows[1:]:
        if len(row)!=len(header): raise ValueError('Malformed consensus ADP row')
        cells = dict(zip(header,row))
        price=number(cells['AVG']['text'])
        if price is None: continue
        name=cells['Player']['name']
        if not name: raise ValueError('Missing player identity; current team labels cannot resolve it')
        rank=number(cells['Rank']['text'])
        if rank is None or rank%1: raise ValueError('Invalid ADP rank')
        quotes={_provider(h):number(cells[h]['text']) for h in header if h not in {'Rank','Player','AVG'}}
        parsed.append({'full_name':name.strip(),'adp':price,'source_rank':int(rank),'quotes':quotes})
    if not parsed: raise ValueError('Empty ADP archive')
    # Some pages retain an inactive provider column (CBS in 2019–20). Only
    # disregard undated columns if the published AVG globally verifies exclusion.
    known={p for p in providers if dates.get(p) is not None}
    def agrees(row, allowed):
        values=[v for p,v in row['quotes'].items() if p in allowed and v is not None]
        return bool(values) and abs(sum(values)/len(values)-row['adp'])<=0.051
    inactive=[]
    if known!=set(providers) and all(agrees(r,known) for r in parsed) and any(not agrees(r,set(providers)) for r in parsed):
        inactive=sorted(set(providers)-known)
    records=[]
    for row in parsed:
        used=[p for p,v in row.pop('quotes').items() if v is not None and p not in inactive]
        available=max((dates[p] for p in used if dates.get(p) is not None),default=None)
        if not used or any(dates.get(p) is None for p in used):
            available=None
        status=('unknown' if available is None else 'preseason' if available<SEASON_OPENERS[season]
                else 'opening_day' if available==SEASON_OPENERS[season] else 'after_opener')
        records.append({'season':season,**row,'available_at':available,'timing_status':status,
                        'contributors':','.join(sorted(used)),
                        'source':f'https://www.fantasypros.com/nba/adp/overall.php?year={season-1}'})
    frame=pd.DataFrame(records)
    if frame.source_rank.duplicated().any() or frame.full_name.map(lambda n:name_key(n,suffixes=True)).duplicated().any():
        raise ValueError('Duplicate ADP player or rank')
    # Published Rank can disagree with AVG when providers are missing. This
    # project uses the user's selected AVG; retain the site's Rank for audit.
    frame=frame.sort_values(['adp','source_rank']).reset_index(drop=True)
    frame['adp_rank']=np.arange(1,len(frame)+1)
    # A player's rank depends on the entire market, not just their own quote.
    # A late Yahoo price can shift even an ESPN-only player's derived slot.
    available=None if frame.available_at.isna().any() else frame.available_at.max()
    status=('unknown' if available is None else 'preseason' if available<SEASON_OPENERS[season]
            else 'opening_day' if available==SEASON_OPENERS[season] else 'after_opener')
    frame['available_at'],frame['timing_status']=available,status
    return frame, {'provider_dates_utc':{p:d.isoformat() if d else None for p,d in dates.items()},
                   'provider_dates_displayed':displayed,'inactive_columns':inactive}


def resolve_adp(adp, games, aliases=None):
    exact, folded = {}, {}
    for row in games[['canonical_player_id','full_name']].drop_duplicates().itertuples(index=False):
        exact.setdefault(name_key(row.full_name,suffixes=True),set()).add(row.canonical_player_id)
        folded.setdefault(name_key(row.full_name),set()).add(row.canonical_player_id)
    overrides={}
    if aliases is not None:
        required={'adp_name','canonical_player_id'}
        if not required<=set(aliases) or aliases.adp_name.duplicated().any(): raise ValueError('Invalid player_aliases.csv')
        if not set(aliases.canonical_player_id)<=set(games.canonical_player_id): raise ValueError('Alias references unknown NBA ID')
        overrides=dict(zip(aliases.adp_name,aliases.canonical_player_id))
    result=adp.copy()
    matches=[]
    for name in result.full_name:
        candidates=({overrides[name]} if name in overrides else exact.get(name_key(name,suffixes=True),set()))
        if not candidates: candidates=folded.get(name_key(name),set())
        matches.append((next(iter(candidates)),'matched') if len(candidates)==1 else (None,'ambiguous' if candidates else 'unresolved'))
    result['canonical_player_id']=[m[0] for m in matches]
    result['match_status']=[m[1] for m in matches]
    duplicate=result.canonical_player_id.notna() & result.duplicated(['season','canonical_player_id'],keep=False)
    result.loc[duplicate,'canonical_player_id']=None
    result.loc[duplicate,'match_status']='ambiguous'
    return result[ADP_SCHEMA.names]


def _write_table(frame,path,schema):
    # Build complete tables before replacing existing outputs. The manifest is
    # written last so interrupted writes cannot pass hash validation.
    table=pa.Table.from_pandas(frame[schema.names],schema=schema,preserve_index=False,safe=True)
    import io
    import pyarrow.parquet as pq
    buffer=io.BytesIO()
    pq.write_table(table,buffer,compression='zstd')
    CachedDownloader._atomic_bytes(path,buffer.getvalue())


def prepare_data(raw_dir=RAW_DIR,processed_dir=PROCESSED_DIR,reports_dir=ROOT/'reports',downloader=None):
    raw_dir,processed_dir,reports_dir=map(Path,(raw_dir,processed_dir,reports_dir))
    client=downloader or CachedDownloader()
    logs,prices,evidence=[],[],{}
    for season in SEASONS:
        path=client.get(f'{MIRROR}/player_game_logs_{season}.parquet',raw_dir/'sportsdataverse'/f'game_logs_{season}.parquet')
        logs.append(normalize_games(pd.read_parquet(path),season))
        url=f'https://www.fantasypros.com/nba/adp/overall.php?year={season-1}'
        page=client.get(url,raw_dir/'archive_sources'/f'fantasypros_{season-1}.html')
        frame,details=parse_adp(page.read_bytes(),season)
        prices.append(frame)
        evidence[str(season)]=details
        LOG.info('%s: %s game appearances, %s ADP prices',season_label(season),len(logs[-1]),len(frame))
    games=pd.concat(logs,ignore_index=True)
    aliases=pd.read_csv(raw_dir.parent/'player_aliases.csv') if (raw_dir.parent/'player_aliases.csv').exists() else None
    adp=resolve_adp(pd.concat(prices,ignore_index=True),games,aliases)
    for frame,name,schema in [(games,'game_logs',GAME_SCHEMA),(adp,'adp',ADP_SCHEMA)]:
        _write_table(frame,processed_dir/f'{name}.parquet',schema)
    metadata={'format_version':2,'seasons':list(SEASONS),'scoring':SCORING,
              'adp_market':'FantasyPros consensus AVG','date_policy':'UTC calendar day of latest contributing source sort key; source-reported, not independent capture',
              'files':{name:hashlib.sha256((processed_dir/f'{name}.parquet').read_bytes()).hexdigest() for name in ['game_logs','adp']},
              'source_dates':evidence}
    CachedDownloader._atomic_bytes(processed_dir/'metadata.json',(json.dumps(metadata,indent=2)+'\n').encode())
    reports_dir.mkdir(parents=True,exist_ok=True)
    coverage=adp.groupby('season').agg(adp_players=('adp','size'),matched_players=('canonical_player_id','count'))
    counts=pd.crosstab(adp.season,adp.timing_status).reindex(columns=['preseason','opening_day','after_opener','unknown'],fill_value=0)
    coverage=coverage.join(counts).join(games.groupby('season').size().rename('game_rows'))
    coverage.to_csv(reports_dir/'data_coverage.csv')
    adp.loc[adp.match_status.ne('matched')].to_csv(reports_dir/'unmatched_adp.csv',index=False)
    print(coverage.to_string())
    print(f'Prepared {len(games):,} games and {len(adp):,} ADP rows. Files: {processed_dir}')
    return games,adp


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    prepare_data(args.data_dir/'raw',args.data_dir/'processed')


if __name__=='__main__':
    main()
