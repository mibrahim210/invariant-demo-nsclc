"""Baseline dataset tests. They use small temporary cohorts, never data/slices/."""
import json

import numpy as np
import pytest

from demo_repo.data_gen import DEFAULT_CONFIG, generate_dataset, load_config
from demo_repo.dataset import load_array, load_metadata, verify_image_manifest

N_PATIENTS = 6


@pytest.fixture(scope="module")
def small_cohort(tmp_path_factory):
    root = tmp_path_factory.mktemp("cohort")
    config = load_config(DEFAULT_CONFIG)
    generate_dataset(config, root, patient_count=N_PATIENTS)
    return root, config


def test_metadata_contract(small_cohort):
    root, config = small_cohort
    df = load_metadata(root / "data" / "metadata.csv")
    assert df.index.name == "slice_id" and df.index.is_unique
    assert len(df) == N_PATIENTS * config["cohort"]["slices_per_patient"]
    assert df["patient_id"].nunique() == N_PATIENTS
    assert set(df["label"]) == {0, 1}
    assert (df.groupby("patient_id")["label"].nunique() == 1).all()  # one label per patient


def test_arrays_shape_dtype_range(small_cohort):
    root, config = small_cohort
    df = load_metadata(root / "data" / "metadata.csv")
    for path in df["array_path"]:
        arr = load_array(path, root=root)
        assert arr.shape == tuple(config["sample"]["shape"])
        assert arr.dtype == np.float32
        assert np.isfinite(arr).all()
        assert arr.min() >= 0.0 and arr.max() <= 1.0


def test_image_manifest_verifies(small_cohort):
    root, _ = small_cohort
    df = load_metadata(root / "data" / "metadata.csv")
    verify_image_manifest(root / "data" / "image_manifest.json", root=root, metadata=df)


def test_generation_is_deterministic(small_cohort, tmp_path):
    root, config = small_cohort
    generate_dataset(config, tmp_path, patient_count=N_PATIENTS)
    first = json.loads((root / "data" / "image_manifest.json").read_text())
    second = json.loads((tmp_path / "data" / "image_manifest.json").read_text())
    assert first == second
    assert (root / "data" / "metadata.csv").read_bytes() == (tmp_path / "data" / "metadata.csv").read_bytes()
