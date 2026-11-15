"""Offline regressions for archive provenance, prices, identities and safe reruns."""
from datetime import datetime, timezone, date
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from src.fantasypros_adp import ArchiveUnavailable, parse_archive, fetch_archives, archive_url
from src.ingest import CachedDownloader, CacheIntegrityError
from src.normalize import normalize_adp, write_parquet, ADP_SCHEMA


def html(*, label="2017-18", updated="2017-10-11", prices=("20.5", "1.5", "-"), names=("A Player", "B Player", "Unpriced")):
    stamp = int(datetime.fromisoformat(updated).replace(tzinfo=timezone.utc).timestamp())
    # Consensus orders A before B; historical ESPN order is B before A.
    rows = ''.join(f'<tr><td>{i}</td><td><a fp-player-name="{name}">{name}</a><small>(CURRENT - C)</small></td><td>999</td><td>{price}</td><td>7</td>'
                   for i, (name, price) in enumerate(zip(names, prices), 1))
    return (f'<title>Average Draft Position (ADP) - Fantasy Basketball {label} | FantasyPros</title>'
            '<table id="data"><thead><tr><th>Rank</th><th>Player</th><th>Yahoo</th><th>ESPN</th><th>AVG</th></tr></thead>'
            f'<tbody>{rows}</tbody></table><table id="experts"><tr><th>Source</th><th>Site</th><th>Last Update</th></tr>'
            '<tr><td>ADP</td><td>Yahoo! Sports</td><td>today</td></tr>'
            f'<tr><td>ADP</td><td><a>ESPN</a></td><td><span style="display:none;">{stamp}</span>10/11</td></tr></table>').encode()


def identity_data(tmp_path, rows=None):
    raw = tmp_path / "raw"
    base = raw / "sportsdataverse"
    base.mkdir(parents=True)
    pd.DataFrame(rows or [{"player": "A Player", "player_id": 1}, {"player": "B Player", "player_id": 2}]).to_parquet(base / "rosters_2018.parquet", index=False)
    path = base / "rosters_2018.parquet"
    path.with_suffix('.parquet.metadata.json').write_text(json.dumps({'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
    return raw


def downloader(payload):
    class Session:
        def __init__(self): self.calls = 0
        def get(self, *args, **kwargs):
            self.calls += 1
            return SimpleNamespace(status_code=200, headers={}, content=payload, raise_for_status=lambda: None)
    session = Session()
    return CachedDownloader(session=session, min_interval=0), session


def test_prices_are_espn_not_consensus_or_yahoo_and_positions_are_unknown():
    frame, meta = parse_archive(html(), 2018)
    assert frame.full_name.tolist() == ["B Player", "A Player"]
    assert frame.adp.tolist() == [1.5, 20.5]
    assert frame.adp_rank.tolist() == [1, 2]
    assert frame.position_rank.isna().all()
    assert frame.position_rank_status.eq("unavailable_historical_positions").all()
    assert meta["available_at"] == "2017-10-11"
    assert meta["source"] == archive_url(2018)
    assert "not_independent" in meta["date_evidence_kind"]
    assert "not_specified" in meta["market_format"]


@pytest.mark.parametrize("label", ["2026-27", "2018-19", "2006-07"])
def test_wrong_season_or_current_page_fallback_is_rejected(label):
    with pytest.raises(ArchiveUnavailable, match="requested"):
        parse_archive(html(label=label), 2018)


@pytest.mark.parametrize("updated", ["2017-10-17", "2017-10-18", "2026-01-01", "2016-12-31"])
def test_opener_day_future_and_wrong_cycle_updates_rejected(updated):
    with pytest.raises(ArchiveUnavailable, match="strictly before"):
        parse_archive(html(updated=updated), 2018)


def test_undated_page_and_missing_espn_column_rejected():
    with pytest.raises(ArchiveUnavailable, match="last-update"):
        parse_archive(html().replace(b' style="display:none;">1507680000', b' style="display:none;">unknown'), 2018)
    with pytest.raises(ArchiveUnavailable, match="price column"):
        parse_archive(html().replace(b'<th>ESPN</th>', b'<th>Expert Rank</th>'), 2018)


@pytest.mark.parametrize("price", ["junk", "0", "-2", "inf", "nan"])
def test_bad_prices_never_filled_from_other_columns(price):
    with pytest.raises(ArchiveUnavailable):
        parse_archive(html(prices=(price, "1.5", "-")), 2018)


def test_duplicate_names_rejected_but_suffixes_preserved_for_identity():
    with pytest.raises(ArchiveUnavailable, match="Duplicate"):
        parse_archive(html(names=("A Player", "A Player", "Unpriced")), 2018)
    frame, _ = parse_archive(html(names=("Gary Payton", "Gary Payton II", "Unpriced")), 2018)
    assert len(frame) == 2


def test_caches_reused_and_original_bytes_preserved(tmp_path):
    raw = identity_data(tmp_path)
    client, session = downloader(html())
    first = fetch_archives(raw_dir=raw, processed_dir=tmp_path / 'processed', seasons=[2018], downloader=client)
    staged = raw / 'adp' / 'fantasypros_espn_2018.csv'
    initial_bytes = staged.read_bytes()
    assert first['results'][0]['rows'] == 2
    fetch_archives(raw_dir=raw, processed_dir=tmp_path / 'processed', seasons=[2018], downloader=client)
    assert session.calls == 1
    assert staged.read_bytes() == initial_bytes
    assert (raw / 'archive_sources' / 'fantasypros_2017.html').read_bytes() == html()


def test_unmapped_player_reported_without_shifting_other_ranks(tmp_path):
    raw = identity_data(tmp_path)
    client, _ = downloader(html(prices=("20", "1", "10"), names=("A Player", "B Player", "Unknown")))
    report = fetch_archives(raw_dir=raw, processed_dir=tmp_path / 'processed', seasons=[2018], downloader=client)
    rows = pd.read_csv(raw / 'adp' / 'fantasypros_espn_2018.csv')
    assert rows.adp_rank.tolist() == [1, 3]
    assert report['results'][0]['unresolved_players'] == [{'season':2018, 'full_name':'Unknown', 'adp_rank':2, 'reason':'unmapped_name'}]


def test_exact_suffix_distinguishes_father_from_son(tmp_path):
    raw = identity_data(tmp_path, [{"player":"Gary Payton", "player_id":56}, {"player":"Gary Payton II", "player_id":1627780}])
    client, _ = downloader(html(names=("Gary Payton", "Gary Payton II", "Unpriced")))
    result = fetch_archives(raw_dir=raw, processed_dir=tmp_path / 'processed', seasons=[2018], downloader=client)
    assert result['results'][0]['unresolved_players'] == []
    frame = pd.read_csv(raw / 'adp' / 'fantasypros_espn_2018.csv')
    assert frame.canonical_player_id.tolist() == ['nba:1627780', 'nba:56']


def test_user_edited_output_is_never_overwritten(tmp_path):
    raw = identity_data(tmp_path)
    client, _ = downloader(html())
    fetch_archives(raw_dir=raw, processed_dir=tmp_path / 'processed', seasons=[2018], downloader=client)
    target = raw / 'adp' / 'fantasypros_espn_2018.csv'
    edited = target.read_bytes().replace(b'20.5', b'25.5')
    target.write_bytes(edited)
    with pytest.raises(CacheIntegrityError, match="Refusing"):
        fetch_archives(raw_dir=raw, processed_dir=tmp_path / 'processed', seasons=[2018], downloader=client)
    assert target.read_bytes() == edited


def test_existing_manual_season_prevents_second_source_or_network_fetch(tmp_path):
    raw = identity_data(tmp_path)
    (raw / 'adp').mkdir()
    target = raw / 'adp' / 'manual.csv'
    target.write_text('season,adp\n2018,1\n')
    client, session = downloader(html())
    report = fetch_archives(raw_dir=raw, processed_dir=tmp_path / 'processed', seasons=[2018], downloader=client)
    assert report['results'][0]['status'] == 'existing_input'
    assert session.calls == 0
    assert target.read_text() == 'season,adp\n2018,1\n'


def test_missing_archives_reported_and_no_unsafe_csv_staged(tmp_path):
    raw = identity_data(tmp_path)
    client, _ = downloader(html(updated='2017-10-17'))
    report = fetch_archives(raw_dir=raw, processed_dir=tmp_path / 'processed', seasons=[2018], downloader=client)
    assert report['results'][0]['status'] == 'manual_snapshot_needed'
    assert not list((raw / 'adp').glob('*.csv'))
    assert json.loads((raw / 'adp' / 'coverage_report.json').read_text()) == report


def test_network_denial_is_not_described_as_missing_archive(tmp_path):
    raw = identity_data(tmp_path)
    class Denied:
        def get(self, *a, **k): raise requests.ConnectionError('403 Forbidden')
    report = fetch_archives(raw_dir=raw, processed_dir=tmp_path / 'processed', seasons=[2018], downloader=Denied())
    assert report['results'][0]['status'] == 'fetch_failed'


def test_corrupt_identity_cache_is_not_used_for_joins(tmp_path):
    raw = identity_data(tmp_path)
    path = raw / 'sportsdataverse' / 'rosters_2018.parquet'
    path.write_bytes(path.read_bytes() + b'changed')
    client, session = downloader(html())
    with pytest.raises(CacheIntegrityError, match='checksum'):
        fetch_archives(raw_dir=raw, processed_dir=tmp_path / 'processed', seasons=[2018], downloader=client)
    assert session.calls == 0


def test_fresh_install_resolves_players_present_only_in_box_scores(tmp_path):
    raw = identity_data(tmp_path, [{"player":"A Player", "player_id":1}])
    path = raw / 'sportsdataverse' / 'game_logs_2018.parquet'
    pd.DataFrame([{'player_name':'B Player','player_id':2}]).to_parquet(path,index=False)
    path.with_suffix('.parquet.metadata.json').write_text(json.dumps({'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
    client, _ = downloader(html())
    report = fetch_archives(raw_dir=raw, processed_dir=tmp_path / 'processed', seasons=[2018], downloader=client)
    assert report['results'][0]['rows'] == 2
    assert report['results'][0]['unresolved_players'] == []


def test_nullable_historical_position_ranks_roundtrip_as_arrow_int64(tmp_path):
    frame, _ = parse_archive(html(), 2018)
    players = pd.DataFrame([{'canonical_player_id':f'nba:{i}', 'full_name':f'{name} Player', 'normalized_name':f'{name.lower()} player',
                             'primary_position':'CURRENT', 'birth_date':date(1990,1,1)} for i,name in [(1,'A'),(2,'B')]])
    output = normalize_adp(frame, players)
    assert output.position_rank.isna().all()
    assert output.position_rank.dtype == 'Int64'
    path = tmp_path / 'adp.parquet'
    write_parquet(output, path, ADP_SCHEMA)
    loaded = pd.read_parquet(path)
    assert loaded.position_rank.isna().all()
    assert loaded.position_rank_status.eq('unavailable_historical_positions').all()
    bad = frame.drop(columns='position_rank_status')
    with pytest.raises(ValueError, match='explicit'):
        normalize_adp(bad, players)
