# all_things_data

Rib.gg scraping and dataset builder for Valorant rounds. Each map lives in its own folder named `training_data_<map>` or `test_data_<map>`, containing:
- `<mode>_matches.txt` / `<mode>_searchlinks.txt` – inputs
- `<mode>_data_json/` – raw match JSON (downloaded if missing)
- `<mode>_<map>_round_data.csv` – per-snapshot dataset output

## Quickstart (recommended)
From this directory:
```bash
python data_grab.py test_data_haven
```
This runs gather → process → fetch_slugs → rename in order and produces `test_data_haven/test_haven_round_data.csv`. Replace `test_data_haven` with any map folder (e.g., `training_data_ascent`).

## Manual steps
```bash
python gather_matches.py test_data_haven    # uses <mode>_searchlinks.txt to collect match IDs
python process_matches.py test_data_haven   # fetch JSON if missing, build CSV
python fetch_slugs.py test_data_haven       # optional: map IDs to readable slugs
python rename_json.py test_data_haven       # optional: rename JSON with slugs
```

## Requirements
```bash
pip install -r requirements.txt
playwright install chromium
```

## Notes
- Folder naming drives defaults; keep the `training_data_<map>` / `test_data_<map>` pattern.
- CSV columns align with `model.data_loader.DEFAULT_FEATURE_COLUMNS` plus metadata (match_id, round_number, etc.).

