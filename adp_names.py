"""Separate player names from FantasyPros display metadata."""

import re


def clean_adp_player_name(name):
    # A team/position tag can be followed by a status, such as `(LAL - PG,SG) DTD`.
    return re.sub(r"\s+\([^()]+\s+-\s+[^()]+\).*$", "", name).strip()
