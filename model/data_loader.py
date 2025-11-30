"""Data loading and preprocessing helpers for Valorant round prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_FEATURE_COLUMNS: List[str] = [
    "time_remaining_s",
    "spike_planted",
    "atk_loadout_value",
    "def_loadout_value",
    "atk_duelists_alive",
    "atk_controllers_alive",
    "atk_initiators_alive",
    "atk_sentinels_alive",
    "def_duelists_alive",
    "def_controllers_alive",
    "def_initiators_alive",
    "def_sentinels_alive",
]

DEFAULT_TARGET_COLUMN = "round_winner"
DEFAULT_LABEL_ORDER = ["ATK", "DEF"]


@dataclass
class RoundDataset:
    """Container for a fully loaded dataset."""

    features: pd.DataFrame
    labels: pd.Series
    label_mapping: Dict[int, str]


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def encode_labels(
    labels: pd.Series, order: Optional[Sequence[str]] = None
) -> Tuple[pd.Series, Dict[int, str]]:
    """Convert textual round winners into integer classes."""

    if order is None:
        # Preserve encounter order to avoid accidental reordering by sort
        order = []
        for value in labels:
            if value not in order:
                order.append(value)
    label_to_index = {label: idx for idx, label in enumerate(order)}
    encoded = labels.map(label_to_index)
    if encoded.isnull().any():
        unknown = labels[encoded.isnull()].unique()
        raise ValueError(f"Found unknown label(s): {unknown}")
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    return encoded.astype(int), index_to_label


def load_round_dataset(
    dataset_path: Path | str,
    feature_columns: Optional[Sequence[str]] = None,
    target_column: str = DEFAULT_TARGET_COLUMN,
    drop_missing: bool = True,
    label_order: Optional[Sequence[str]] = DEFAULT_LABEL_ORDER,
) -> RoundDataset:
    """Load the Valorant round dataset from CSV."""

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    features = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    _ensure_columns(df, [*features, target_column])

    working = df[features + [target_column]].copy()
    if drop_missing:
        working = working.dropna().reset_index(drop=True)

    encoded_labels, label_mapping = encode_labels(
        working[target_column], order=label_order
    )
    feature_frame = working[features].astype(float)
    return RoundDataset(
        features=feature_frame.reset_index(drop=True),
        labels=encoded_labels.reset_index(drop=True),
        label_mapping=label_mapping,
    )


def train_validation_split(
    features: pd.DataFrame,
    labels: pd.Series,
    val_ratio: float = 0.2,
    shuffle: bool = True,
    stratify: bool = True,
    random_state: Optional[int] = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and labels into train/validation partitions."""

    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if len(features) != len(labels):
        raise ValueError("Features and labels must have the same length.")

    if stratify and labels.nunique() > 1:
        val_indices: List[int] = []
        rng = np.random.default_rng(random_state)

        for _, idx_group in labels.groupby(labels).groups.items():
            idx_array = np.array(idx_group, dtype=int)
            if shuffle:
                rng.shuffle(idx_array)
            val_count = max(1, int(round(len(idx_array) * val_ratio)))
            if val_count >= len(idx_array) and len(idx_array) > 1:
                val_count = len(idx_array) - 1
            val_indices.extend(idx_array[:val_count])

        val_set = set(val_indices)
        train_indices = [idx for idx in labels.index if idx not in val_set]
    else:
        indices = list(labels.index)
        if shuffle:
            rng = np.random.default_rng(random_state)
            rng.shuffle(indices)
        val_size = max(1, int(round(len(indices) * val_ratio)))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

    if not train_indices:
        raise ValueError("Not enough samples to create a training split.")

    train_X = features.loc[train_indices].reset_index(drop=True)
    train_y = labels.loc[train_indices].reset_index(drop=True)
    val_X = features.loc[val_indices].reset_index(drop=True)
    val_y = labels.loc[val_indices].reset_index(drop=True)
    return train_X, val_X, train_y, val_y

