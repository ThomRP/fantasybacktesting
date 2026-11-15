import pandas as pd
import pytest

from src.adp_archive import adapt_espn_2025, ARCHIVE_URL


def test_archive_uses_actual_espn_prices_and_preseason_positions():
    payload = (
        "Name,Pos,ESPN,AVG\nB Player,G/F,20.5,2\nA Player,C,1.5,3\n"
        "C Player,G,20.5,1\nUnpriced,F,-,4\n"
    ).encode()
    result = adapt_espn_2025(payload)
    assert result.full_name.tolist() == ["A Player", "B Player", "C Player"]
    assert result.adp.tolist() == [1.5, 20.5, 20.5]
    assert result.adp_rank.tolist() == [1, 2, 3]
    assert result.position_rank.tolist() == [1, 1, 2]
    assert result.market_position.tolist() == ["C", "G", "G"]
    assert result.season.eq(2025).all()
    assert result.available_at.eq("2024-10-19").all()
    assert result.source.eq(ARCHIVE_URL).all()


@pytest.mark.parametrize("value", ["junk", "-1", "0", "inf", "NaN"])
def test_archive_rejects_invalid_price_instead_of_imputing(value):
    with pytest.raises(ValueError):
        adapt_espn_2025(f"Name,Pos,ESPN\nPlayer,C,{value}\n".encode())


def test_archive_rejects_duplicate_priced_names():
    with pytest.raises(ValueError, match="Duplicate"):
        adapt_espn_2025(b"Name,Pos,ESPN\nPlayer,C,1\nPlayer,C,2\n")
