#!/usr/bin/env python3
"""
Train a model for a given map.

Usage:
    python train_model.py ascent
    python train_model.py haven
    python train_model.py ascent --with-test
"""

import subprocess
import sys
import os
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(str(c) for c in cmd)}")
    print('='*60 + "\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        print(f"\n❌ Failed: {description}")
        return False
    
    return True


def print_metrics_summary(metrics_path: Path, map_name: str):
    """Print a formatted summary of the training metrics."""
    if not metrics_path.exists():
        print("⚠️  Metrics file not found")
        return
    
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    nb = metrics.get("naive_bayes", {})
    train_acc = nb.get("train", {}).get("accuracy", 0) * 100
    val_acc = nb.get("val", {}).get("accuracy", 0) * 100
    test_data = nb.get("test", {})
    test_acc = test_data.get("accuracy", 0) * 100 if test_data else None
    
    # Print accuracy table
    print("\n" + "="*60)
    print("📊 MODEL PERFORMANCE SUMMARY")
    print("="*60)
    print(f"\n{'Map':<12} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12}")
    print("-"*48)
    test_str = f"{test_acc:.1f}%" if test_acc else "(skipped)"
    print(f"{map_name.upper():<12} {train_acc:.1f}%{'':<7} {val_acc:.1f}%{'':<7} {test_str:<12}")
    
    # Print confusion matrices
    print("\n" + "-"*48)
    print("CONFUSION MATRICES")
    print("-"*48)
    
    for split_name in ["train", "val", "test"]:
        split_data = nb.get(split_name, {})
        if not split_data:
            continue
        
        cm = split_data.get("confusion_matrix", {})
        if not cm:
            continue
        
        print(f"\n{split_name.upper()} Confusion Matrix:")
        print(f"{'':>12} {'Pred ATK':>10} {'Pred DEF':>10}")
        
        for actual in ["ATK", "DEF"]:
            row = cm.get(actual, {})
            pred_atk = row.get("ATK", 0)
            pred_def = row.get("DEF", 0)
            print(f"{'Actual '+actual:>12} {pred_atk:>10} {pred_def:>10}")
    
    print("\n" + "="*60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_model.py <map_name> [--with-test]")
        print("Example: python train_model.py ascent")
        print("         python train_model.py haven --with-test")
        sys.exit(1)
    
    map_name = sys.argv[1].lower()
    with_test = "--with-test" in sys.argv
    
    # Check if training data exists
    training_csv = REPO_ROOT / "all_things_data" / f"training_data_{map_name}" / f"training_{map_name}_round_data.csv"
    if not training_csv.exists():
        print(f"❌ Error: Training data not found at {training_csv}")
        print(f"   Run: cd all_things_data && python data_grab.py training_data_{map_name}")
        sys.exit(1)
    
    # Check if test data exists
    test_csv = REPO_ROOT / "all_things_data" / f"test_data_{map_name}" / f"test_{map_name}_round_data.csv"
    has_test_data = test_csv.exists()
    skip_test = not with_test or not has_test_data
    
    if with_test and not has_test_data:
        print(f"⚠️  Warning: Test data not found at {test_csv}")
        print(f"   Skipping test evaluation.")
    
    # Model paths
    model_path = REPO_ROOT / "models" / map_name / "naive_bayes_stats.json"
    metrics_path = REPO_ROOT / "models" / map_name / "training_metrics.json"
    predictions_path = REPO_ROOT / "models" / map_name / "test_round_predictions.csv"
    
    print(f"\n🎮 Training model for map: {map_name.upper()}")
    print("="*60)
    print(f"Training data: {training_csv}")
    if has_test_data:
        print(f"Test data: {test_csv}")
    print(f"Model output: models/{map_name}/")
    print("="*60)
    
    # Run from repo root
    os.chdir(REPO_ROOT)
    
    # Step 1: Train the model
    train_cmd = [
        sys.executable, "-m", "model.train",
        "--map", map_name,
    ]
    if skip_test:
        train_cmd.append("--skip-test")
    
    if not run_command(train_cmd, "Train Naive Bayes model"):
        sys.exit(1)
    
    # Step 2: Generate predictions (if test data exists and --with-test)
    if with_test and has_test_data:
        predict_cmd = [
            sys.executable, "-m", "model.predict_rounds",
            "--model", str(model_path),
            "--dataset", str(test_csv),
            "--output", str(predictions_path),
        ]
        
        if not run_command(predict_cmd, "Generate test predictions"):
            print("⚠️  Warning: Failed to generate predictions, but model was trained.")
    
    # Print metrics summary
    print_metrics_summary(metrics_path, map_name)
    
    print(f"\n✅ Model trained for {map_name.upper()}!")
    print(f"   Model: models/{map_name}/naive_bayes_stats.json")
    print(f"   Metrics: models/{map_name}/training_metrics.json")
    if with_test and has_test_data:
        print(f"   Predictions: models/{map_name}/test_round_predictions.csv")
    print()


if __name__ == "__main__":
    main()
