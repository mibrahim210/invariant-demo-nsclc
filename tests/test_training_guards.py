"""Guards in the training harness: partition validation and durable failure outcomes."""
import pandas as pd
import pytest

from demo_repo.data_gen import DEFAULT_CONFIG, generate_dataset, load_config
from demo_repo.train import evaluate_seed, validate_partition

MODEL = load_config(DEFAULT_CONFIG)["experiment"]["model"]


def frame():
    rows = [{"slice_id": f"P{p}_S{s}", "patient_id": f"P{p}", "label": p % 2} for p in range(6) for s in range(2)]
    return pd.DataFrame(rows).set_index("slice_id")


@pytest.mark.parametrize("train, test, expected", [
    ([], None, "train set is empty"),
    (None, [], "test set is empty"),
    (lambda ids: ids[:6] + ids[:1], lambda ids: ids[6:], "duplicate"),
    (lambda ids: ids[:6] + ["nope"], lambda ids: ids[6:], "unknown"),
    (lambda ids: ids[:7], lambda ids: ids[6:], "both sets"),
    (lambda ids: ids[:5], lambda ids: ids[6:], "neither set"),
])
def test_invalid_partitions_are_rejected(train, test, expected):
    df = frame()
    ids = list(df.index)
    tr = ids[:6] if train is None else (train(ids) if callable(train) else train)
    te = ids[6:] if test is None else (test(ids) if callable(test) else test)
    assert any(expected in e for e in validate_partition(df, tr, te))


def test_valid_partition_passes():
    df = frame()
    ids = list(df.index)
    assert validate_partition(df, ids[:6], ids[6:]) == []


def test_missing_patient_id_is_rejected():
    df = frame()
    df.iloc[0, df.columns.get_loc("patient_id")] = None
    ids = list(df.index)
    assert any("missing patient_id" in e for e in validate_partition(df, ids[:6], ids[6:]))


def features_for(df):
    lookup = {i: [float(n), 0.0] for n, i in enumerate(df.index)}
    return lookup.__getitem__


def test_one_class_training_set_is_unavailable_not_a_crash():
    df = frame()
    zeros = [i for i in df.index if df.loc[i, "label"] == 0]
    rest = [i for i in df.index if i not in zeros]
    result = evaluate_seed(df, features_for(df), lambda d, seed, test_size: (zeros, rest), 0, 0.2, MODEL)
    assert result["status"] == "unavailable"
    assert "only class" in result["status_reason"]
    assert result["metrics"]["overlap_count"] == 0


def test_invalid_partition_is_recorded():
    df = frame()
    ids = list(df.index)
    result = evaluate_seed(df, features_for(df), lambda d, seed, test_size: (ids[:7], ids[6:]), 0, 0.2, MODEL)
    assert result["status"] == "invalid_partition"


def test_split_exception_is_recorded():
    def broken(d, seed, test_size):
        raise RuntimeError("boom")
    result = evaluate_seed(frame(), lambda i: [0.0], broken, 3, 0.2, MODEL)
    assert result["status"] == "failed" and result["stage"] == "split"
    assert "RuntimeError: boom" in result["status_reason"]


def test_generator_refuses_existing_output(tmp_path):
    config = load_config(DEFAULT_CONFIG)
    generate_dataset(config, tmp_path, patient_count=2)
    with pytest.raises(FileExistsError):
        generate_dataset(config, tmp_path, patient_count=2)


def test_completed_run_records_counts_and_timing_boundary():
    df = frame()
    ids = list(df.index)  # P0..P2 train, P3..P5 test: both classes on each side
    result = evaluate_seed(df, features_for(df), lambda d, seed, test_size: (ids[:6], ids[6:]), 0, 0.2, MODEL)
    assert result["status"] == "completed"
    m = result["metrics"]
    assert m["train_row_count"] == 6 and m["test_row_count"] == 6
    assert m["class_counts"]["test"]["patients"] == {"0": 1, "1": 2}
    assert "feature preparation" in result["timing"]["timing_boundary"]
