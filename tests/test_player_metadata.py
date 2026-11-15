from datetime import date

import pandas as pd

from src.player_metadata import load_supplemental_bios


def _write(tmp_path, crosswalk, core, season=2025):
    base = tmp_path / "player_metadata"
    base.mkdir(exist_ok=True)
    pd.DataFrame(crosswalk).assign(season=season).to_parquet(base / f"crosswalk_{season}.parquet")
    pd.DataFrame(core).assign(season=season).to_parquet(base / f"core_{season}.parquet")


def test_supplement_requires_exact_names_bijective_ids_and_exact_bridge(tmp_path):
    _write(tmp_path, {
        "nba_player_id": ["1", "2", "3", "4", "4", "5"],
        "nba_player_name": ["José Player Jr.", "Wrong Name", "Fuzzy Player", "Conflict", "Conflict", "Exact Player"],
        "espn_athlete_id": ["11", "22", "33", "44", "55", "66"],
        "match_method": ["exact_name", "exact_name", "fuzzy", "exact_name", "exact_name", "exact_name"],
        "match_confidence": [1., 1., .95, 1., 1., 1.],
    }, {
        "athlete_id": ["11", "22", "33", "44", "55", "66"],
        "full_name": ["Jose Player", "Other Name", "Fuzzy Player", "Conflict", "Conflict", "Exact Player"],
        "date_of_birth": ["1990-01-02T08:00Z"] * 6,
        "position_abbreviation": ["G"] * 6,
    })
    result = load_supplemental_bios(tmp_path, [2025])
    assert set(result.player_id) == {"1", "5"}
    assert result.birth_date.eq(date(1990, 1, 2)).all()
    assert result.position.eq("G").all()


def test_conflicting_birthdays_across_years_are_not_guessed(tmp_path):
    crosswalk = {"nba_player_id": ["1"], "nba_player_name": ["Player"],
                 "espn_athlete_id": ["11"], "match_method": ["exact_name"], "match_confidence": [1.]}
    core = {"athlete_id": ["11"], "full_name": ["Player"],
            "date_of_birth": ["1990-01-02T08:00Z"], "position_abbreviation": ["G"]}
    _write(tmp_path, crosswalk, core, 2024)
    core["date_of_birth"] = ["1990-01-03T08:00Z"]
    _write(tmp_path, crosswalk, core, 2025)
    assert load_supplemental_bios(tmp_path, [2024, 2025]).empty
