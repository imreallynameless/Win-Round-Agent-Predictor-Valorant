"""CLI to score each round snapshot and include event descriptions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .events import describe_round_events
from .naive_bayes import GaussianNaiveBayes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-snapshot predictions with event descriptions."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/naive_bayes_stats.json"),
        help="Path to the saved Naive Bayes model JSON.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="CSV containing match snapshots (e.g. test_haven_round_data.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/test_round_predictions.csv"),
        help="Destination CSV for predictions + descriptions.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    model = GaussianNaiveBayes.load(args.model)
    df = pd.read_csv(args.dataset)
    missing = [col for col in ["match_id", "round_number"] + model.input_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns required by the model: {missing}")

    features = df[model.input_columns].astype(float)
    probabilities = model.predict_proba(features)
    predictions = probabilities.idxmax(axis=1)
    confidence = probabilities.max(axis=1)

    results = df[["match_id", "round_number", "round_winner"]].copy()
    results["predicted_winner"] = predictions
    results["predicted_confidence"] = confidence
    results["event_description"] = describe_round_events(df)

    feature_snapshot = df[model.input_columns].reset_index(drop=True)
    results = pd.concat(
        [results, feature_snapshot, probabilities.reset_index(drop=True)], axis=1
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()

