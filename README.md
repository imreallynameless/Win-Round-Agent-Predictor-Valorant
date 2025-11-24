# Win Round Agent Predictor - Valorant

This project scrapes and processes professional Valorant match data (specifically for the map **Haven**) from [rib.gg](https://rib.gg). It provides two separate datasets: one for **training** and one for **testing**.

## Project Structure

All scripts and data are contained within the `all_things_data/` directory.

```
all_things_data/
├── process_matches.py       # Main processing script
├── gather_matches.py        # Scrapes match IDs
├── fetch_slugs.py           # Fetches URL slugs
├── rename_json.py           # Renames JSON files
├── requirements.txt         # Dependencies
├── training_data/           # Training dataset folder
│   ├── training_haven_round_data.csv
│   ├── training_matches.txt
│   └── training_data_json/
└── test_data/               # Test dataset folder
    ├── test_haven_round_data.csv
    ├── test_matches.txt
    └── test_data_json/
```

## Setup & Installation

1.  **Prerequisites:**
    - Python 3.8+
    - `pip` (Python package manager)

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r all_things_data/requirements.txt
    playwright install chromium
    ```

## Usage

Navigate to the `all_things_data` directory:
```bash
cd all_things_data
```

### Generating Training Data
To generate or update the training dataset:

```bash
python process_matches.py training
```
This reads `training_data/training_matches.txt` and outputs `training_data/training_haven_round_data.csv`.

### Generating Test Data
To generate or update the test dataset:

```bash
python process_matches.py test
```
This reads `test_data/test_matches.txt` and outputs `test_data/test_haven_round_data.csv`.

## Adding New Matches

To add new matches to a dataset (e.g., `test`):

1.  Add rib.gg search URLs to `test_data/test_searchlinks.txt`.
2.  Run the gather script:
    ```bash
    python gather_matches.py test
    ```
3.  (Optional) Fetch slugs and rename files:
    ```bash
    python fetch_slugs.py test
    python rename_json.py test
    ```
4.  Run the processing script:
    ```bash
    python process_matches.py test
    ```

## Model Training

The `model/` package contains a Gaussian Naive Bayes pipeline that learns from the `training_haven_round_data.csv` file.

1.  Install the additional modeling dependencies (within your virtual environment):
    ```bash
    pip install -r model/requirements.txt
    ```
2.  Run the trainer (add `--with-pca` to enable dimensionality reduction, or `--with-tree` to compare against a RandomForest baseline). By default the script trains on `training_haven_round_data.csv`, keeps a validation split, and then evaluates the saved model on `test_haven_round_data.csv`:
    ```bash
    python -m model.train \
      --dataset all_things_data/training_data/training_haven_round_data.csv \
      --test-dataset all_things_data/test_data/test_haven_round_data.csv \
      --with-pca \
      --with-tree
    ```
3.  Artifacts (`models/naive_bayes_stats.json`, optional `models/random_forest.pkl`) and metrics (`models/training_metrics.json`) will be written to the repository root. The saved Naive Bayes model includes the fitted statistics, PCA metadata (if enabled), validation results, and a test-set report.

### Single-Round Prediction

For quick experiments you can call the helper that mirrors the original assignment interface: it trains on the full training CSV, evaluates a single round from the test CSV, and returns the predicted winner plus its probability.

```python
from model.naive_bayes import predict_round_winner

training_csv = "all_things_data/training_data/training_haven_round_data.csv"
# Feature order: time_remaining_s, spike_planted, t1_loadout_value, t2_loadout_value,
#                t1_duelists_alive, t1_controllers_alive, t1_initiators_alive, t1_sentinels_alive,
#                t2_duelists_alive, t2_controllers_alive, t2_initiators_alive, t2_sentinels_alive
round_features = [100, 0, 4100, 3400, 1, 1, 1, 0, 1, 1, 2, 1]

winner, confidence = predict_round_winner(training_csv, round_features)
print(winner, confidence)  # -> "ATK", 0.78 (example)
```

### Per-Snapshot Predictions with Event Descriptions

After training, you can score every snapshot in the test CSV (or any other dataset with the same schema) and capture a human-readable description of what changed at each point in the round (e.g., “t1 lost 1 sentinel”, “spike planted”).

```bash
python -m model.predict_rounds \
  --model models/naive_bayes_stats.json \
  --dataset all_things_data/test_data/test_haven_round_data.csv \
  --output models/test_round_predictions.csv
```

The output CSV includes the original match/round identifiers, the ground-truth winner, the model’s prediction + confidence, the ATK/DEF probabilities, and a new `event_description` column summarizing the detected change for that snapshot.
