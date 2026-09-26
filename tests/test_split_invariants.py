"""Tests for patient-level disjointness invariant (INV-1)."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from demo_repo.splits import make_split


# ---------------------------------------------------------------------------
# Fixed fixture: 20 patients × 3 slices each = 60 rows
# ---------------------------------------------------------------------------

def _make_df(n_patients: int = 20, slices_per_patient: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for p in range(n_patients):
        label = int(p < n_patients // 2)
        for s in range(slices_per_patient):
            records.append(
                {
                    "slice_id": f"S_P{p:03d}_{s}",
                    "patient_id": f"P{p:03d}",
                    "label": label,
                }
            )
    df = pd.DataFrame(records).set_index("slice_id")
    return df


def patient_overlap(df: pd.DataFrame, train_idx, test_idx) -> set:
    """Patients present in both partitions (must be empty under INV-1)."""
    return set(df.loc[train_idx, "patient_id"]) & set(df.loc[test_idx, "patient_id"])


@pytest.fixture()
def small_df():
    return _make_df(n_patients=20, slices_per_patient=3)


# ---------------------------------------------------------------------------
# Domain test: patient sets must be disjoint
# ---------------------------------------------------------------------------

def test_patient_disjointness(small_df):
    train_idx, test_idx = make_split(small_df, seed=0, test_size=0.2)
    overlap = patient_overlap(small_df, train_idx, test_idx)
    assert not overlap, f"Patient leakage detected: {overlap}"


def test_partition_covers_all_rows(small_df):
    train_idx, test_idx = make_split(small_df, seed=0, test_size=0.2)
    all_returned = set(train_idx) | set(test_idx)
    assert all_returned == set(small_df.index), "Partition does not cover all rows."
    assert len(set(train_idx) & set(test_idx)) == 0, "Duplicate rows across partitions."


def test_both_sets_nonempty(small_df):
    train_idx, test_idx = make_split(small_df, seed=0, test_size=0.2)
    assert len(train_idx) > 0, "Train set is empty."
    assert len(test_idx) > 0, "Test set is empty."


# ---------------------------------------------------------------------------
# Hypothesis property test
# ---------------------------------------------------------------------------

@given(
    n_patients=st.integers(min_value=5, max_value=50),
    slices_per_patient=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=999),
    # test_size must leave at least 1 patient on each side; use 0.1–0.5
    test_size=st.floats(min_value=0.1, max_value=0.5),
)
@settings(max_examples=50, deadline=None)
def test_no_patient_overlap_property(n_patients, slices_per_patient, seed, test_size):
    df = _make_df(n_patients=n_patients, slices_per_patient=slices_per_patient, seed=seed)
    train_idx, test_idx = make_split(df, seed=seed, test_size=test_size)
    overlap = patient_overlap(df, train_idx, test_idx)
    assert not overlap, f"Patient leakage: {overlap}"