"""Metadata and array loading for the synthetic NSCLC-inspired demo."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = ["slice_id", "patient_id", "study_id", "slice_index", "array_path", "label", "generator_version"]


def load_metadata(csv_path: str | Path = REPO_ROOT / "data" / "metadata.csv") -> pd.DataFrame:
    """Load metadata with slice_id as the unique DataFrame index."""
    df = pd.read_csv(csv_path, dtype={"slice_id": str, "patient_id": str, "study_id": str})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"metadata is missing columns: {missing}")
    if df["patient_id"].isna().any():
        raise ValueError("metadata contains missing patient_id values")
    return df.set_index("slice_id", verify_integrity=True)


def load_array(npy_path: str | Path, root: Path = REPO_ROOT) -> np.ndarray:
    """Load one (2, 64, 64) float32 sample; relative paths resolve against the repo root."""
    path = Path(npy_path)
    if not path.is_absolute():
        path = root / path
    return np.load(path, allow_pickle=False)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_image_manifest(manifest_path: str | Path = REPO_ROOT / "data" / "image_manifest.json",
                          root: Path = REPO_ROOT, metadata: pd.DataFrame | None = None) -> str:
    """Check every array against the manifest; return the manifest's SHA-256.

    Raises ValueError on any missing file, hash mismatch, or metadata/manifest disagreement.
    """
    manifest_path = Path(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    paths = [e["array_path"] for e in entries]
    if paths != sorted(paths) or len(set(paths)) != len(paths):
        raise ValueError("image manifest must be sorted and free of duplicate paths")
    bad = [e["array_path"] for e in entries
           if not (root / e["array_path"]).is_file() or _sha256(root / e["array_path"]) != e["sha256"]]
    if bad:
        raise ValueError(f"{len(bad)} arrays missing or not matching the manifest, e.g. {bad[:5]}")
    if metadata is not None and set(metadata["array_path"]) != set(paths):
        raise ValueError("metadata array_path values do not match the image manifest")
    return _sha256(manifest_path)
