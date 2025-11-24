"""Helpers for describing per-row events in round timelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

ROLE_COLUMNS: Dict[str, str] = {
    "t1_duelists_alive": "t1 duelist",
    "t1_controllers_alive": "t1 controller",
    "t1_initiators_alive": "t1 initiator",
    "t1_sentinels_alive": "t1 sentinel",
    "t2_duelists_alive": "t2 duelist",
    "t2_controllers_alive": "t2 controller",
    "t2_initiators_alive": "t2 initiator",
    "t2_sentinels_alive": "t2 sentinel",
}


def describe_round_events(round_df: pd.DataFrame) -> pd.Series:
    """Return a human-readable description for each snapshot."""

    required = {"match_id", "round_number", *ROLE_COLUMNS.keys(), "spike_planted"}
    missing = required - set(round_df.columns)
    if missing:
        raise ValueError(f"Dataframe missing required columns: {', '.join(sorted(missing))}")

    descriptions: List[str] = []
    prev_key = None
    prev_row = None

    for _, row in round_df.iterrows():
        key = (row["match_id"], row["round_number"])
        if key != prev_key or prev_row is None:
            descriptions.append("round start snapshot")
        else:
            descriptions.append(_describe_state_change(prev_row, row))
        prev_key = key
        prev_row = row
    return pd.Series(descriptions, index=round_df.index, name="event_description")


def _pluralize(label: str, count: int) -> str:
    return f"{label}{'' if count == 1 else 's'}"


def _describe_state_change(prev_row: pd.Series, curr_row: pd.Series) -> str:
    changes: List[str] = []

    prev_spike = int(prev_row["spike_planted"])
    curr_spike = int(curr_row["spike_planted"])
    if prev_spike == 0 and curr_spike == 1:
        changes.append("spike planted")
    elif prev_spike == 1 and curr_spike == 0:
        changes.append("spike reset")

    for column, label in ROLE_COLUMNS.items():
        prev = int(prev_row[column])
        curr = int(curr_row[column])
        diff = curr - prev
        if diff == 0:
            continue
        count = abs(diff)
        team, role = label.split(" ", 1)
        role_text = _pluralize(role, count)
        if diff < 0:
            changes.append(f"{team} lost {count} {role_text}")
        else:
            changes.append(f"{team} gained {count} {role_text}")

    if not changes:
        return "status update"
    return "; ".join(changes)

