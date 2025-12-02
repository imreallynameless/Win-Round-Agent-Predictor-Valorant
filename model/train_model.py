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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(str(c) for c in cmd)}")
    print('='*60 + "\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        return False
    
    return True


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
    
    print(f"\n{'='*60}")
    print(f"✅ Model trained for {map_name.upper()}!")
    print(f"   Model: models/{map_name}/naive_bayes_stats.json")
    print(f"   Metrics: models/{map_name}/training_metrics.json")
    if with_test and has_test_data:
        print(f"   Predictions: models/{map_name}/test_round_predictions.csv")
    print('='*60)


if __name__ == "__main__":
    main()
