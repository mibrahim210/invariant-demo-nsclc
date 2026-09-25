"""Inference CLI: class-1 probability for one or more slices.

The model is rebuilt from a recorded training run, so every prediction is traceable to
the run whose training rows it used:

    python -m demo_repo.predict --run-id <run_id> data/slices/NSCLC_P000_S00.npy [...]

Features are computed with the same deterministic transform as training and evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from demo_repo.dataset import REPO_ROOT, load_metadata
from demo_repo.train import ARTIFACT_ROOT, CONFIG_PATH, METADATA_PATH, extract_features


def load_run(run_id: str, artifact_root: Path = ARTIFACT_ROOT) -> dict:
    path = artifact_root / run_id / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"no run manifest at {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"run {run_id} has status {manifest.get('status')!r}; only completed runs can serve")
    return manifest


def build_model(manifest: dict, metadata, model_cfg: dict, root: Path = REPO_ROOT) -> KNeighborsClassifier:
    train_ids = manifest["split_assignments"]["train_slice_ids"]
    X = np.stack([extract_features(root / metadata.loc[i, "array_path"]) for i in train_ids])
    y = metadata.loc[train_ids, "label"].to_numpy()
    clf = KNeighborsClassifier(n_neighbors=model_cfg["n_neighbors"], weights=model_cfg["weights"],
                               algorithm=model_cfg["algorithm"], metric=model_cfg["metric"],
                               n_jobs=model_cfg["n_jobs"])
    return clf.fit(X, y)


def predict(clf: KNeighborsClassifier, array_paths: list[Path]) -> list[float]:
    X = np.stack([extract_features(p) for p in array_paths])
    return clf.predict_proba(X)[:, list(clf.classes_).index(1)].tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True, help="completed run in run_artifacts/")
    parser.add_argument("arrays", nargs="+", type=Path)
    args = parser.parse_args()

    model_cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["experiment"]["model"]
    try:
        manifest = load_run(args.run_id)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")
    clf = build_model(manifest, load_metadata(METADATA_PATH), model_cfg)
    for path, prob in zip(args.arrays, predict(clf, args.arrays)):
        print(json.dumps({"array": path.as_posix(), "run_id": args.run_id, "prob_class_1": round(prob, 6)}))


if __name__ == "__main__":
    main()
