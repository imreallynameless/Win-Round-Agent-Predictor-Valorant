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
