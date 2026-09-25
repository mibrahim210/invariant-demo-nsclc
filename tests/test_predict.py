"""Inference CLI tests on a small temporary cohort."""
import json

import numpy as np
import pytest

from demo_repo.data_gen import DEFAULT_CONFIG, generate_dataset, load_config
from demo_repo.dataset import load_metadata
from demo_repo.predict import build_model, load_run, predict
from demo_repo.train import extract_features

CONFIG = load_config(DEFAULT_CONFIG)
MODEL = CONFIG["experiment"]["model"]


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    root = tmp_path_factory.mktemp("cohort")
    generate_dataset(CONFIG, root, patient_count=6)
    return root, load_metadata(root / "data" / "metadata.csv")


def write_manifest(tmp_path, run_id, train_ids, status="completed"):
    (tmp_path / run_id).mkdir()
    manifest = {"run_id": run_id, "status": status, "split_assignments": {"train_slice_ids": train_ids}}
    (tmp_path / run_id / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_predictions_are_probabilities(cohort, tmp_path):
    root, df = cohort
    write_manifest(tmp_path, "r1", list(df.index[:24]))
    clf = build_model(load_run("r1", tmp_path), df, MODEL, root=root)
    probs = predict(clf, [root / p for p in df["array_path"].iloc[24:]])
    assert len(probs) == len(df) - 24
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_inference_uses_the_evaluation_transform(cohort):
    root, df = cohort
    path = root / df["array_path"].iloc[0]
    arr = np.load(path, allow_pickle=False)
    expected = arr.reshape(2, 8, 8, 8, 8).mean(axis=(2, 4)).ravel(order="C")
    assert np.array_equal(extract_features(path), expected)


def test_incomplete_runs_cannot_serve(tmp_path):
    write_manifest(tmp_path, "r2", [], status="failed")
    with pytest.raises(ValueError):
        load_run("r2", tmp_path)
