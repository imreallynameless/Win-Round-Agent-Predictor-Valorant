# Win Round Agent Predictor (Valorant)

An end-to-end pipeline to:
- Scrape pro Valorant match data from [rib.gg](https://rib.gg) for multiple maps.
- Build round-level datasets (training + test) with loadout, agent-role counts, spike state, and time remaining.
- Train and evaluate a Gaussian Naive Bayes model (with optional PCA and RandomForest baseline).
- Score individual rounds or full timelines, including human-readable event descriptions.
- (Experimental) Extract features directly from broadcast screenshots via OpenCV/EasyOCR to feed the model.

## Repository Layout

```
all_things_data/   # rib.gg scraping + dataset generation (has its own README with details)
model/             # training, evaluation, prediction utilities (see module docs/tests)
models/            # saved model artifacts & metrics (per map)
opencv/            # computer-vision feature extractor for screenshots (see opencv/README.md)
tests/             # unit tests for model utilities
```

Each major folder ships its own README or docstrings with deeper, task-specific instructions; start there when working within a subcomponent.

## Quickstart

1) Create & activate a virtualenv (Python 3.10+ recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2) Install dependencies:
```bash
pip install -r all_things_data/requirements.txt
pip install -r model/requirements.txt
playwright install chromium
```

3) Generate data for a map (example: Haven test set). Recommended: run the orchestrator so all steps execute in order (gather → process → slugs → rename):
```bash
cd all_things_data
python data_grab.py test_data_haven
```
This produces `all_things_data/test_data_haven/test_haven_round_data.csv` (and fetches/renames JSON as needed).  
If you already have JSON downloaded and just want to rebuild the CSV, you can instead run:
```bash
python process_matches.py test_data_haven
```

4) Train & evaluate (map defaults auto-set paths; override with flags as needed):
```bash
python -m model.train \
  --map haven \
  --with-pca \
  --with-tree
```
Map defaults expect:
- Training CSV: `all_things_data/training_data_<map>/training_<map>_round_data.csv`
- Test CSV: `all_things_data/test_data_<map>/test_<map>_round_data.csv`

Artifacts land under `models/<map>/`:
- `naive_bayes_stats.json` (saved statistics + preprocessing metadata)
- `training_metrics.json` (train/val/test accuracy + confusion)
- `random_forest.pkl` (optional baseline when `--with-tree`)

5) Predict a single live round:
```python
from model.naive_bayes import predict_round_winner

round_features = [100, 0, 4100, 3400, 1, 1, 1, 0, 1, 1, 2, 1]
winner, confidence = predict_round_winner(
    "all_things_data/training_data_haven/training_haven_round_data.csv",
    round_features,
)
print(winner, confidence)
```

6) Score whole timelines with event descriptions:
```bash
python -m model.predict_rounds \
  --model models/haven/naive_bayes_stats.json \
  --dataset all_things_data/test_data_haven/test_haven_round_data.csv \
  --output models/haven/test_round_predictions.csv
```
Adds `predicted_winner`, `predicted_confidence`, per-class probabilities, and an `event_description` column derived from snapshot deltas.

## Data Pipeline (rib.gg)

Scripts live in `all_things_data/` and expect per-map folders such as `training_data_haven`, `test_data_ascent`, etc.

- `gather_matches.py <folder>`: open search URLs from `<folder>/<mode>_searchlinks.txt` and collect match IDs for the chosen map.
- `process_matches.py <folder>`: fetch match JSON (if missing) and emit `<mode>_<map>_round_data.csv` with per-snapshot rows.
- `fetch_slugs.py <folder>` + `rename_json.py <folder>`: optional readability helpers for JSON filenames.
- `data_grab.py <folder>`: orchestration wrapper running the above steps sequentially.

Resulting CSV schema (order matches `model.data_loader.DEFAULT_FEATURE_COLUMNS`):
```
time_remaining_s, spike_planted, atk_loadout_value, def_loadout_value,
atk_duelists_alive, atk_controllers_alive, atk_initiators_alive, atk_sentinels_alive,
def_duelists_alive, def_controllers_alive, def_initiators_alive, def_sentinels_alive,
round_winner, match_id, round_number, ...
```

## Modeling

- `model.train`: CLI for Gaussian Naive Bayes with optional PCA preprocessing and RandomForest baseline.
- `model.naive_bayes`: lightweight, serializable NB implementation with reproducible preprocessing metadata.
- `model.events`: converts per-snapshot deltas into readable text (e.g., “DEF lost 1 sentinel; spike planted”).
- `model.predict_rounds`: batch scoring + event descriptions to CSV.

Key flags:
- `--map haven|ascent|...`: sets default data/model paths.
- `--features ...`: use a feature subset.
- `--with-pca --pca-variance 0.95`: standardize + PCA before NB.
- `--with-tree`: also train a RandomForest baseline.
- `--skip-test`: skip held-out test evaluation when a file is absent.

## Computer Vision Extractor (experimental)

`opencv/opencv.py` detects HUD elements (agents, weapons, shields, timer, spike state) from 1920x1080 VCT broadcast screenshots using OpenCV SIFT + EasyOCR, then feeds the trained NB model:
- Agent/weapon/shield templates live in `opencv/icons/`.
- Sample screenshots in `opencv/screenshots/`.
- Read the module doc `opencv/README.md` for coordinate calibration and limitations.

## Testing

Run unit tests (model utilities):
```bash
pytest
```

## Notes & Limitations

- Playwright steps open Chromium; ensure a display or use headless mode if you adapt the scripts.
- Data pipeline assumes rib.gg responses contain `events` and `economies`; missing fields will currently fail fast.
- CV extractor is resolution- and HUD-layout-specific; it loads a sample screenshot on import, so run it from within `opencv/` with assets present.

## Authors

Lei Wu, Kevin Wu, Ryan Guo

