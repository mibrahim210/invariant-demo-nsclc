"""Baseline split tests: sizes and row-level partition only."""
import pandas as pd
import pytest

from demo_repo.splits import make_split


def metadata_frame(n_patients=20, slices=8):
    rows = [{"slice_id": f"P{p:03d}_S{s:02d}", "patient_id": f"P{p:03d}", "label": p % 2}
            for p in range(n_patients) for s in range(slices)]
    return pd.DataFrame(rows).set_index("slice_id")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_split_sizes_and_row_partition(seed):
    df = metadata_frame()
    train_ids, test_ids = make_split(df, seed=seed, test_size=0.2)
    train_ids, test_ids = list(train_ids), list(test_ids)
    assert len(test_ids) == round(0.2 * len(df))
    assert len(train_ids) + len(test_ids) == len(df)
    assert set(train_ids) | set(test_ids) == set(df.index)
    assert not set(train_ids) & set(test_ids)


def test_split_is_reproducible():
    df = metadata_frame()
    assert list(make_split(df, seed=0)[1]) == list(make_split(df, seed=0)[1])
