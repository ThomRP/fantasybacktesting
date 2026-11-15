from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from src.ingest import CachedDownloader, CacheIntegrityError, _validate_player_logs


def response(body=b'{"ok":true}', status=200, headers=None):
    result = Mock(status_code=status, content=body, headers=headers or {})
    if status >= 400:
        result.raise_for_status.side_effect = requests.HTTPError(str(status))
    return result


def test_cache_preserves_bytes_avoids_network_and_detects_corruption(tmp_path):
    session = Mock()
    session.get.return_value = response(b'{ "original" : [1,2] }\n')
    path = tmp_path / "download.json"
    client = CachedDownloader(session=session, min_interval=0)
    client.get("https://example.com/data", path)
    assert path.read_bytes() == b'{ "original" : [1,2] }\n'
    client.get("https://example.com/data", path)
    assert session.get.call_count == 1
    path.write_bytes(b"corrupt")
    with pytest.raises(CacheIntegrityError, match="checksum"):
        client.get("https://example.com/data", path)
    assert session.get.call_count == 1


def test_transient_retry_then_success_and_no_retry_on_access_denied(tmp_path):
    session = Mock()
    sleeps = []
    session.get.side_effect = [response(status=429, headers={"Retry-After": "3"}), response()]
    client = CachedDownloader(session=session, min_interval=0, sleep=sleeps.append)
    client.get("https://example.com/data", tmp_path / "ok.json")
    assert session.get.call_count == 2
    assert 3 in sleeps
    session.reset_mock()
    session.get.side_effect = [response(status=403)]
    with pytest.raises(requests.HTTPError):
        client.get("https://example.com/data", tmp_path / "denied.json")
    assert session.get.call_count == 1
    assert not (tmp_path / "denied.json").exists()


def test_partial_failure_does_not_cache_error_and_resumes(tmp_path):
    session = Mock()
    session.get.side_effect = requests.Timeout("timeout")
    client = CachedDownloader(session=session, retries=0, min_interval=0)
    path = tmp_path / "download.json"
    with pytest.raises(requests.Timeout):
        client.get("https://example.com/data", path)
    assert list(tmp_path.iterdir()) == []
    session.get.side_effect = None
    session.get.return_value = response()
    client.get("https://example.com/data", path)
    assert path.exists()


def test_pending_metadata_recovery_after_interrupted_commit(tmp_path):
    session = Mock()
    session.get.return_value = response()
    client = CachedDownloader(session=session, min_interval=0)
    path = tmp_path / "download.json"
    client.get("https://example.com/data", path)
    metadata = path.with_suffix(".json.metadata.json")
    metadata.rename(metadata.with_suffix(".json.pending"))
    client.get("https://example.com/data", path)
    assert metadata.exists()
    assert session.get.call_count == 1


def test_validator_failure_does_not_cache_and_team_rows_rejected(tmp_path):
    session = Mock()
    session.get.return_value = response(b'{"resultSets":[{"headers":["TEAM_ID","GAME_DATE"],"rowSet":[[1,"2025-01-01"]]}]}')
    client = CachedDownloader(session=session, min_interval=0)
    with pytest.raises(ValueError, match="PLAYER"):
        client.get("https://example.com/data", tmp_path / "bad.json",
                   validator=lambda b: _validate_player_logs(b, parquet=False))
    assert list(tmp_path.iterdir()) == []


def test_request_identity_cannot_reuse_a_different_source(tmp_path):
    session = Mock()
    session.get.return_value = response()
    client = CachedDownloader(session=session, min_interval=0)
    path = tmp_path / "download.json"
    client.get("https://example.com/data", path, params={"season": 2025})
    with pytest.raises(CacheIntegrityError, match="identity"):
        client.get("https://example.com/data", path, params={"season": 2026})
    assert session.get.call_count == 1


def test_local_adp_adapter_respects_requested_season_subset(tmp_path):
    from src.ingest import read_adp_csvs
    import pandas as pd
    (tmp_path / "adp").mkdir()
    pd.DataFrame({"season": [2024, 2025], "adp": [1.0, 2.0]}).to_csv(tmp_path / "adp" / "market.csv", index=False)
    result = read_adp_csvs(raw_dir=tmp_path, seasons=[2025])
    assert result.season.tolist() == [2025]
