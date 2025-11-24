from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import data_loader
from model.events import describe_round_events
from model.naive_bayes import GaussianNaiveBayes, predict_round_winner, train_gaussian_nb


def _make_dataset():
    features = pd.DataFrame(
        {
            "f1": [1.0, 1.2, 8.5, 9.0],
            "f2": [0.8, 1.1, 8.2, 8.7],
        }
    )
    labels = pd.Series([0, 0, 1, 1])
    label_mapping = {0: "ATK", 1: "DEF"}
    return features, labels, label_mapping


def test_naive_bayes_predicts_expected_labels():
    features, labels, mapping = _make_dataset()
    model = train_gaussian_nb(features, labels, feature_columns=features.columns, label_mapping=mapping)

    candidates = pd.DataFrame({"f1": [1.1, 8.8], "f2": [0.9, 8.4]})
    predictions = model.predict(candidates)
    assert list(predictions) == ["ATK", "DEF"]


def test_naive_bayes_with_preprocessing_pipeline(tmp_path):
    features, labels, mapping = _make_dataset()
    preprocessing = [
        {"name": "standardize", "params": {"mean": [4.925, 4.7], "scale": [3.5, 3.6]}},
        {
            "name": "pca",
            "params": {
                "mean": [0.0, 0.0],
                "components": [[1.0, 0.0], [0.0, 1.0]],
                "explained_variance_ratio": [0.5, 0.5],
            },
        },
    ]
    model = train_gaussian_nb(
        features,
        labels,
        feature_columns=features.columns,
        label_mapping=mapping,
        preprocessing=preprocessing,
    )

    out_path = tmp_path / "nb.json"
    model.save(out_path)
    loaded = GaussianNaiveBayes.load(out_path)

    candidates = pd.DataFrame({"f1": [1.1, 9.0], "f2": [1.0, 8.9]})
    predictions = loaded.predict(candidates)
    assert list(predictions) == ["ATK", "DEF"]


def test_predict_round_winner_from_dataset(tmp_path):
    feature_columns = data_loader.DEFAULT_FEATURE_COLUMNS
    rows = [
        {
            "round_winner": "ATK",
            "time_remaining_s": 90,
            "spike_planted": 0,
            "t1_loadout_value": 4000,
            "t2_loadout_value": 3200,
            "t1_duelists_alive": 2,
            "t1_controllers_alive": 1,
            "t1_initiators_alive": 1,
            "t1_sentinels_alive": 0,
            "t2_duelists_alive": 1,
            "t2_controllers_alive": 1,
            "t2_initiators_alive": 2,
            "t2_sentinels_alive": 1,
        },
        {
            "round_winner": "ATK",
            "time_remaining_s": 60,
            "spike_planted": 1,
            "t1_loadout_value": 5000,
            "t2_loadout_value": 3500,
            "t1_duelists_alive": 2,
            "t1_controllers_alive": 1,
            "t1_initiators_alive": 1,
            "t1_sentinels_alive": 1,
            "t2_duelists_alive": 1,
            "t2_controllers_alive": 0,
            "t2_initiators_alive": 1,
            "t2_sentinels_alive": 0,
        },
        {
            "round_winner": "DEF",
            "time_remaining_s": 40,
            "spike_planted": 0,
            "t1_loadout_value": 3000,
            "t2_loadout_value": 4500,
            "t1_duelists_alive": 1,
            "t1_controllers_alive": 0,
            "t1_initiators_alive": 1,
            "t1_sentinels_alive": 0,
            "t2_duelists_alive": 2,
            "t2_controllers_alive": 1,
            "t2_initiators_alive": 1,
            "t2_sentinels_alive": 1,
        },
        {
            "round_winner": "DEF",
            "time_remaining_s": 30,
            "spike_planted": 0,
            "t1_loadout_value": 2500,
            "t2_loadout_value": 4600,
            "t1_duelists_alive": 1,
            "t1_controllers_alive": 0,
            "t1_initiators_alive": 0,
            "t1_sentinels_alive": 0,
            "t2_duelists_alive": 2,
            "t2_controllers_alive": 1,
            "t2_initiators_alive": 1,
            "t2_sentinels_alive": 1,
        },
    ]

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "rounds.csv"
    df.to_csv(csv_path, index=False)

    atk_sample = [rows[0][col] for col in feature_columns]
    atk_label, atk_prob = predict_round_winner(csv_path, atk_sample)
    assert atk_label == "ATK"
    assert 0.0 <= atk_prob <= 1.0

    def_sample = {col: rows[-1][col] for col in feature_columns}
    def_label, def_prob = predict_round_winner(csv_path, def_sample)
    assert def_label == "DEF"
    assert 0.0 <= def_prob <= 1.0


def test_describe_round_events():
    rows = [
        {
            "match_id": 1,
            "round_number": 1,
            "round_winner": "ATK",
            "time_remaining_s": 90,
            "spike_planted": 0,
            "t1_loadout_value": 4000,
            "t2_loadout_value": 3200,
            "t1_duelists_alive": 2,
            "t1_controllers_alive": 1,
            "t1_initiators_alive": 1,
            "t1_sentinels_alive": 1,
            "t2_duelists_alive": 2,
            "t2_controllers_alive": 1,
            "t2_initiators_alive": 1,
            "t2_sentinels_alive": 1,
        },
        {
            "match_id": 1,
            "round_number": 1,
            "round_winner": "ATK",
            "time_remaining_s": 60,
            "spike_planted": 1,
            "t1_loadout_value": 4200,
            "t2_loadout_value": 2800,
            "t1_duelists_alive": 2,
            "t1_controllers_alive": 1,
            "t1_initiators_alive": 1,
            "t1_sentinels_alive": 1,
            "t2_duelists_alive": 1,
            "t2_controllers_alive": 1,
            "t2_initiators_alive": 1,
            "t2_sentinels_alive": 0,
        },
        {
            "match_id": 1,
            "round_number": 1,
            "round_winner": "ATK",
            "time_remaining_s": 30,
            "spike_planted": 1,
            "t1_loadout_value": 3900,
            "t2_loadout_value": 1500,
            "t1_duelists_alive": 1,
            "t1_controllers_alive": 1,
            "t1_initiators_alive": 1,
            "t1_sentinels_alive": 1,
            "t2_duelists_alive": 1,
            "t2_controllers_alive": 0,
            "t2_initiators_alive": 1,
            "t2_sentinels_alive": 0,
        },
    ]
    df = pd.DataFrame(rows)
    descriptions = describe_round_events(df)
    assert descriptions.iloc[0] == "round start snapshot"
    assert "spike planted" in descriptions.iloc[1]
    assert "t2 lost 1 sentinel" in descriptions.iloc[1]
    assert "t1 lost 1 duelist" in descriptions.iloc[2]

