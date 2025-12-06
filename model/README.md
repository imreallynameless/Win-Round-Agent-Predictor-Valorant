# model

Training, evaluation, and inference utilities for the Valorant round predictor. Uses a lightweight Gaussian Naive Bayes implementation with optional PCA preprocessing and a RandomForest baseline.

## Install
```bash
pip install -r model/requirements.txt
```

## Train
Map-aware defaults expect:
- Train CSV: `all_things_data/training_data_<map>/training_<map>_round_data.csv`
- Test CSV:  `all_things_data/test_data_<map>/test_<map>_round_data.csv`

Example (Haven, with PCA + tree baseline):
```bash
python -m model.train \
  --map haven \
  --with-pca \
  --with-tree
```
Artifacts go to `models/<map>/` (`naive_bayes_stats.json`, `training_metrics.json`, optional `random_forest.pkl`).

Key flags:
- `--features ...` use a feature subset.
- `--pca-variance 0.95` or `--pca-components N` to tune PCA.
- `--skip-test` if you lack a test CSV.

## Batch scoring with events
```bash
python -m model.predict_rounds \
  --model models/haven/naive_bayes_stats.json \
  --dataset all_things_data/test_data_haven/test_haven_round_data.csv \
  --output models/haven/test_round_predictions.csv
```
Adds `predicted_winner`, probabilities, and `event_description` per snapshot.

## Single-round helper
```python
from model.naive_bayes import predict_round_winner
winner, prob = predict_round_winner(
    "all_things_data/training_data_haven/training_haven_round_data.csv",
    [100, 0, 4100, 3400, 1, 1, 1, 0, 1, 1, 2, 1],
)
print(winner, prob)
```

## Tests
```bash
pytest
```

