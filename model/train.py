"""Training entry point for Valorant round prediction models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from . import data_loader
from .naive_bayes import GaussianNaiveBayes, train_gaussian_nb
from .tree_baseline import TreeTrainingResult, train_random_forest

try:  # Optional dependencies for dimensionality reduction
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover - guard for optional dependency
    PCA = None  # type: ignore
    StandardScaler = None  # type: ignore
    _PREPROCESS_ERROR = exc
else:
    _PREPROCESS_ERROR = None

try:  # Optional persistence for tree models
    import joblib
except ImportError:  # pragma: no cover
    joblib = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "all_things_data" / "training_data" / "training_haven_round_data.csv"
DEFAULT_TEST_DATASET = REPO_ROOT / "all_things_data" / "test_data" / "test_haven_round_data.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Valorant round prediction models."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the training CSV.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Optional subset of feature columns to use.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable stratified splitting.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for splitting and PCA.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/naive_bayes_stats.json"),
        help="Where to persist the Naive Bayes model stats.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("models/training_metrics.json"),
        help="Where to persist evaluation metrics.",
    )
    parser.add_argument(
        "--with-pca",
        action="store_true",
        help="Apply StandardScaler+PCA before Naive Bayes.",
    )
    parser.add_argument(
        "--pca-variance",
        type=float,
        default=0.95,
        help="Retained variance for PCA when components not specified.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help="Explicit PCA component count (overrides variance).",
    )
    parser.add_argument(
        "--with-tree",
        action="store_true",
        help="Train a RandomForest baseline alongside Naive Bayes.",
    )
    parser.add_argument(
        "--tree-model-out",
        type=Path,
        default=Path("models/random_forest.pkl"),
        help="Optional path to persist the tree baseline (requires joblib).",
    )
    parser.add_argument(
        "--tree-estimators",
        type=int,
        default=500,
        help="Number of trees for the RandomForest baseline.",
    )
    parser.add_argument(
        "--tree-max-depth",
        type=int,
        default=None,
        help="Maximum depth for the RandomForest baseline.",
    )
    parser.add_argument(
        "--test-dataset",
        type=Path,
        default=DEFAULT_TEST_DATASET,
        help="Held-out test CSV used for final reporting.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Disable evaluation on the held-out test dataset.",
    )
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = parse_args()

    dataset = data_loader.load_round_dataset(
        args.dataset,
        feature_columns=args.features or data_loader.DEFAULT_FEATURE_COLUMNS,
    )
    train_X, val_X, train_y, val_y = data_loader.train_validation_split(
        dataset.features,
        dataset.labels,
        val_ratio=args.val_ratio,
        stratify=not args.no_stratify,
        random_state=args.random_state,
    )

    preprocessing_steps = build_preprocessing_metadata(
        train_X, args.with_pca, args.pca_variance, args.pca_components, args.random_state
    )

    nb_model = train_gaussian_nb(
        train_X,
        train_y,
        feature_columns=train_X.columns,
        label_mapping=dataset.label_mapping,
        preprocessing=preprocessing_steps,
    )
    nb_model.save(args.model_out)

    metrics: Dict[str, Any] = {
        "naive_bayes": {
            "train": evaluate_model(nb_model, train_X, train_y, dataset.label_mapping),
            "val": evaluate_model(nb_model, val_X, val_y, dataset.label_mapping),
            "preprocessing": {
                "enabled": bool(preprocessing_steps),
                "steps": preprocessing_steps or [],
            },
        }
    }

    tree_result: Optional[TreeTrainingResult] = None
    if args.with_tree:
        tree_result = train_random_forest(
            train_X,
            train_y,
            n_estimators=args.tree_estimators,
            max_depth=args.tree_max_depth,
            random_state=args.random_state,
        )
        metrics["random_forest"] = {
            "train": evaluate_tree(tree_result, train_X, train_y),
            "val": evaluate_tree(tree_result, val_X, val_y),
        }
        maybe_save_tree_model(tree_result, args.tree_model_out)

    if not args.skip_test and args.test_dataset:
        if args.test_dataset.exists():
            feature_order = list(train_X.columns)
            label_order = [dataset.label_mapping[idx] for idx in sorted(dataset.label_mapping)]
            test_data = data_loader.load_round_dataset(
                args.test_dataset,
                feature_columns=feature_order,
                label_order=label_order,
            )
            metrics["naive_bayes"]["test"] = evaluate_model(
                nb_model, test_data.features, test_data.labels, dataset.label_mapping
            )
            if tree_result:
                metrics["random_forest"]["test"] = evaluate_tree(
                    tree_result, test_data.features, test_data.labels
                )
        else:
            print(f"Warning: test dataset {args.test_dataset} not found; skipping test evaluation.")

    save_metrics(metrics, args.metrics_out)
    print("Training complete.")
    print(json.dumps(metrics, indent=2))


def build_preprocessing_metadata(
    features: pd.DataFrame,
    use_pca: bool,
    pca_variance: float,
    pca_components: Optional[int],
    random_state: int,
) -> Optional[List[Dict[str, Any]]]:
    if not use_pca:
        return None
    if StandardScaler is None or PCA is None:
        raise ImportError(
            "scikit-learn is required for PCA preprocessing. "
            "Install it via `pip install scikit-learn`."
        ) from _PREPROCESS_ERROR

    scaler = StandardScaler()
    scaler.fit(features)
    scaled = scaler.transform(features)

    if pca_components:
        pca = PCA(n_components=pca_components, random_state=random_state)
    else:
        pca = PCA(n_components=pca_variance, random_state=random_state, svd_solver="full")
    pca.fit(scaled)

    return [
        {"name": "standardize", "params": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}},
        {
            "name": "pca",
            "params": {
                "mean": pca.mean_.tolist(),
                "components": pca.components_.tolist(),
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            },
        },
    ]


def evaluate_model(
    model: GaussianNaiveBayes,
    features: pd.DataFrame,
    labels: pd.Series,
    label_mapping: Dict[int, str],
) -> Dict[str, Any]:
    actual = labels.map(label_mapping)
    predicted = model.predict(features)
    return _build_metrics(actual, predicted, label_mapping)


def evaluate_tree(
    result: TreeTrainingResult,
    features: pd.DataFrame,
    labels: pd.Series,
) -> Dict[str, Any]:
    predictions = result.model.predict(features[result.feature_columns])
    label_mapping = result.label_mapping
    actual = labels.map(label_mapping)
    predicted = pd.Series(predictions).map(label_mapping)
    return _build_metrics(actual, predicted, label_mapping)


def _build_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    label_mapping: Dict[int, str],
) -> Dict[str, Any]:
    label_order = [label_mapping[idx] for idx in sorted(label_mapping.keys())]
    accuracy = float((actual.values == predicted.values).mean())
    confusion = (
        pd.crosstab(actual, predicted, dropna=False)
        .reindex(index=label_order, columns=label_order, fill_value=0)
        .astype(int)
    )
    return {
        "accuracy": accuracy,
        "confusion_matrix": {
            row: confusion.loc[row].to_dict() for row in confusion.index
        },
        "support": actual.value_counts().to_dict(),
    }


def maybe_save_tree_model(result: TreeTrainingResult, output_path: Path) -> None:
    if joblib is None:
        print("joblib not available; skipping tree model serialization.")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, output_path)


def save_metrics(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()

