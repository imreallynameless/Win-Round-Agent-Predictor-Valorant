"""Tree-based baselines for Valorant round prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError as exc:  # pragma: no cover - guard for optional dependency
    RandomForestClassifier = None  # type: ignore
    _RF_ERROR = exc
else:
    _RF_ERROR = None


@dataclass
class TreeTrainingResult:
    model: Any
    feature_columns: list[str]
    label_mapping: Dict[int, str]


def _require_random_forest() -> None:
    if RandomForestClassifier is None:
        raise ImportError(
            "scikit-learn is required for the tree baseline. "
            "Install it via `pip install scikit-learn`."
        ) from _RF_ERROR


def train_random_forest(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    random_state: int = 42,
) -> TreeTrainingResult:
    """Train a RandomForest classifier as a stronger baseline."""

    _require_random_forest()
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(features, labels)
    label_mapping = {int(idx): str(idx) for idx in sorted(labels.unique())}
    return TreeTrainingResult(
        model=model,
        feature_columns=list(features.columns),
        label_mapping=label_mapping,
    )

