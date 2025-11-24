"""Gaussian Naive Bayes utilities tailored for Valorant round features."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import data_loader

SMALL_STD = 1e-6


@dataclass
class ClassStatistics:
    prior: float
    mean: List[float]
    std: List[float]


@dataclass
class GaussianNaiveBayes:
    input_columns: List[str]
    label_mapping: Dict[int, str]
    class_stats: Dict[int, ClassStatistics]
    preprocessing: Optional[List[Dict[str, Any]]] = None

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return class probabilities for each row in `features`."""

        matrix = features[self.input_columns].to_numpy(dtype=float)
        matrix = _apply_preprocessing_matrix(matrix, self.preprocessing)
        class_indices = sorted(self.class_stats.keys())

        log_probs = []
        for class_idx in class_indices:
            stats = self.class_stats[class_idx]
            mean = np.asarray(stats.mean)
            std = np.maximum(np.asarray(stats.std), SMALL_STD)
            log_prior = math.log(stats.prior)

            # Gaussian log PDF per feature
            log_pdf = (
                -0.5 * np.log(2 * math.pi * (std**2))
                - ((matrix - mean) ** 2) / (2 * (std**2))
            )
            class_log_prob = log_prior + np.sum(log_pdf, axis=1)
            log_probs.append(class_log_prob)

        stacked = np.vstack(log_probs).T  # shape: (n_samples, n_classes)
        normalized = _softmax_log_probs(stacked)

        columns = [self.label_mapping[idx] for idx in class_indices]
        return pd.DataFrame(normalized, columns=columns)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return the most likely class label for each row."""

        probabilities = self.predict_proba(features)
        labels = probabilities.idxmax(axis=1)
        return labels

    def to_dict(self) -> Dict:
        return {
            "input_columns": self.input_columns,
            "label_mapping": self.label_mapping,
            "class_stats": {
                str(idx): asdict(stats) for idx, stats in self.class_stats.items()
            },
            "preprocessing": self.preprocessing,
        }

    def save(self, output_path: Path | str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def from_dict(cls, payload: Dict) -> "GaussianNaiveBayes":
        class_stats = {
            int(idx): ClassStatistics(**stats)
            for idx, stats in payload["class_stats"].items()
        }
        return cls(
            input_columns=list(payload["input_columns"]),
            label_mapping={int(k): v for k, v in payload["label_mapping"].items()},
            class_stats=class_stats,
            preprocessing=payload.get("preprocessing"),
        )

    @classmethod
    def load(cls, model_path: Path | str) -> "GaussianNaiveBayes":
        with Path(model_path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)


def _softmax_log_probs(log_probs: np.ndarray) -> np.ndarray:
    """Convert log-probabilities to normalized probabilities."""

    max_vals = np.max(log_probs, axis=1, keepdims=True)
    stabilized = log_probs - max_vals
    exp_vals = np.exp(stabilized)
    sums = np.sum(exp_vals, axis=1, keepdims=True)
    return exp_vals / sums


def _apply_preprocessing_matrix(
    matrix: np.ndarray, preprocessing: Optional[List[Dict[str, Any]]]
) -> np.ndarray:
    if not preprocessing:
        return matrix

    transformed = matrix
    for step in preprocessing:
        name = step.get("name")
        params = step.get("params", {})
        if name == "standardize":
            transformed = _apply_standardize(transformed, params)
        elif name == "pca":
            transformed = _apply_pca(transformed, params)
        else:
            raise ValueError(f"Unknown preprocessing step: {name}")
    return transformed


def _apply_standardize(matrix: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    mean = np.asarray(params.get("mean"), dtype=float)
    scale = np.asarray(params.get("scale"), dtype=float)
    scale = np.where(scale == 0, 1.0, scale)
    return (matrix - mean) / scale


def _apply_pca(matrix: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    components = np.asarray(params.get("components"), dtype=float)
    mean = np.asarray(params.get("mean"), dtype=float)
    centered = matrix - mean
    return centered @ components.T


def compute_statistics(matrix: np.ndarray, labels: np.ndarray) -> Dict[int, ClassStatistics]:
    """Compute per-class mean/std/prior for Gaussian Naive Bayes."""

    label_array = np.asarray(labels, dtype=int)
    unique_labels, counts = np.unique(label_array, return_counts=True)
    total = len(label_array)

    stats: Dict[int, ClassStatistics] = {}
    for label_value, count in zip(unique_labels, counts):
        mask = label_array == label_value
        class_matrix = matrix[mask]
        mean = np.mean(class_matrix, axis=0)
        if len(class_matrix) > 1:
            std = np.std(class_matrix, axis=0, ddof=1)
        else:
            std = np.full_like(mean, SMALL_STD)
        std = np.where(np.isnan(std) | (std <= 0), SMALL_STD, std)
        stats[int(label_value)] = ClassStatistics(
            prior=count / total, mean=mean.tolist(), std=std.tolist()
        )

    return stats


def train_gaussian_nb(
    features: pd.DataFrame,
    labels: pd.Series,
    feature_columns: Optional[Sequence[str]] = None,
    label_mapping: Optional[Dict[int, str]] = None,
    preprocessing: Optional[List[Dict[str, Any]]] = None,
) -> GaussianNaiveBayes:
    """Train a Gaussian Naive Bayes model and return a serializable wrapper."""

    input_columns = list(feature_columns) if feature_columns is not None else list(features.columns)
    matrix = features[input_columns].to_numpy(dtype=float)
    processed = _apply_preprocessing_matrix(matrix, preprocessing)
    stats = compute_statistics(processed, labels.to_numpy())
    if label_mapping is None:
        unique_labels = sorted({int(idx) for idx in labels.unique()})
        label_mapping = {idx: str(idx) for idx in unique_labels}
    return GaussianNaiveBayes(
        input_columns=input_columns,
        label_mapping=label_mapping,
        class_stats=stats,
        preprocessing=preprocessing,
    )


def predict_round_winner(
    dataset_path: Path | str,
    round_features: Sequence[float] | Dict[str, float],
    *,
    feature_columns: Optional[Sequence[str]] = None,
) -> tuple[str, float]:
    """Train on `dataset_path` and predict a single round outcome.

    Args:
        dataset_path: CSV containing historical rounds with the expected columns.
        round_features: either an ordered sequence that matches `feature_columns`
            or a mapping of column -> value describing the live round.
        feature_columns: optional custom column ordering. Defaults to the loader
            constant: time_remaining_s, spike_planted, loadouts, agent counts.

    Returns:
        (round_winner_label, win_probability) where the label is "ATK" or "DEF"
        and the probability is the model's confidence for that winner.
    """

    dataset = data_loader.load_round_dataset(
        dataset_path,
        feature_columns=feature_columns or data_loader.DEFAULT_FEATURE_COLUMNS,
    )
    model = train_gaussian_nb(
        dataset.features,
        dataset.labels,
        feature_columns=dataset.features.columns,
        label_mapping=dataset.label_mapping,
    )

    if isinstance(round_features, dict):
        row = []
        for col in dataset.features.columns:
            if col not in round_features:
                raise KeyError(f"Missing feature '{col}' in round_features.")
            row.append(float(round_features[col]))
        sample = pd.DataFrame([row], columns=dataset.features.columns)
    else:
        ordered = list(round_features)
        expected = len(dataset.features.columns)
        if len(ordered) != expected:
            raise ValueError(
                f"Expected {expected} feature values, received {len(ordered)}."
            )
        sample = pd.DataFrame([ordered], columns=dataset.features.columns)

    probabilities = model.predict_proba(sample).iloc[0]
    winner = probabilities.idxmax()
    win_probability = float(probabilities[winner])
    return winner, win_probability

