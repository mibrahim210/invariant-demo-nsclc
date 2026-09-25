"""Train and evaluate the small CPU baseline with durable run provenance.

Usage (from the repository root, on a clean committed checkout):
    python -m demo_repo.train                    # configured seeds, the repository's split
    python -m demo_repo.train --seeds 0
    python -m demo_repo.train --split-file ../invariant/reference/reference_splits.py \\
                              --split-function reference_splits:grouped_split

Every seed writes run_artifacts/<run_id>/manifest.json with a status:
    completed | unavailable | invalid_partition | failed
A manifest is written even when the split, the model or MLflow fails.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import platform
import subprocess
import sys
import time
import traceback
import uuid
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from demo_repo.dataset import REPO_ROOT, load_array, load_metadata, verify_image_manifest

CONFIG_PATH = REPO_ROOT / "configs" / "data_generation.json"
METADATA_PATH = REPO_ROOT / "data" / "metadata.csv"
IMAGE_MANIFEST_PATH = REPO_ROOT / "data" / "image_manifest.json"
GENERATION_MANIFEST_PATH = REPO_ROOT / "data" / "generation_manifest.json"
ARTIFACT_ROOT = REPO_ROOT / "run_artifacts"
DEFAULT_SPLIT = "demo_repo.splits:make_split"
RUN_OUTPUT_PREFIXES = ("run_artifacts/", "mlruns/", "mlflow.db")
TIMING_BOUNDARY = ("split + partition validation + feature preparation (array load and block means) "
                   "+ fit + prediction + evaluation; excludes image-manifest verification, "
                   "MLflow logging and manifest writing")


# ---------------------------------------------------------------- provenance
def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()


def repo_state() -> dict:
    """HEAD and cleanliness of the demo repository; run outputs do not count as dirty."""
    try:
        sha = _git(REPO_ROOT, "rev-parse", "--verify", "HEAD")
        status = _git(REPO_ROOT, "status", "--porcelain")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"sha": None, "dirty": None, "dirty_paths": []}
    dirty = [line[3:] for line in status.splitlines() if not line[3:].startswith(RUN_OUTPUT_PREFIXES)]
    return {"sha": sha, "dirty": bool(dirty), "dirty_paths": dirty[:10]}


def code_binding(module) -> dict:
    """Bind a split module to its file hash and the commit of the repository that contains it."""
    path = Path(inspect.getsourcefile(module)).resolve()
    binding = {"module": module.__name__, "file_sha256": sha256_file(path),
               "repo_git_sha": None, "repo_relative_path": None, "repo_remote": None,
               "tracked": False, "clean": False}
    try:
        top = Path(_git(path.parent, "rev-parse", "--show-toplevel")).resolve()
        rel = path.relative_to(top).as_posix()
        binding["repo_relative_path"] = rel
        binding["repo_git_sha"] = _git(top, "rev-parse", "--verify", "HEAD")
        try:
            binding["repo_remote"] = _git(top, "config", "--get", "remote.origin.url") or None
        except subprocess.CalledProcessError:
            pass
        try:
            _git(top, "ls-files", "--error-unmatch", rel)
            binding["tracked"] = True
            binding["clean"] = _git(top, "status", "--porcelain", "--", rel) == ""
        except subprocess.CalledProcessError:
            pass
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass
    return binding


def dependency_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas", "scikit-learn", "mlflow"):
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def resolve_split(spec: str, split_file: Path | None):
    module_name, func_name = spec.split(":")
    if split_file is not None:
        split_file = split_file.resolve()
        if split_file.stem != module_name:
            sys.exit(f"ERROR: --split-file {split_file.name} does not define module '{module_name}'.")
        sys.path.insert(0, str(split_file.parent))
    module = importlib.import_module(module_name)
    if split_file is not None and Path(inspect.getsourcefile(module)).resolve() != split_file:
        sys.exit(f"ERROR: '{module_name}' resolved to {module.__file__}, not {split_file}.")
    return getattr(module, func_name), code_binding(module)


# ---------------------------------------------------------------- validation
def validate_partition(df: pd.DataFrame, train_ids: list, test_ids: list) -> list[str]:
    """Complete two-way partition contract; returns a list of violations (empty = valid)."""
    errors = []
    if not df.index.is_unique:
        errors.append("metadata row identifiers are not unique")
    if df["patient_id"].isna().any():
        errors.append("metadata has missing patient_id values")
    for name, ids in (("train", train_ids), ("test", test_ids)):
        if len(ids) == 0:
            errors.append(f"{name} set is empty")
        if len(set(ids)) != len(ids):
            errors.append(f"{name} set contains duplicate row identifiers")
        unknown = set(ids) - set(df.index)
        if unknown:
            errors.append(f"{name} set contains {len(unknown)} unknown row identifiers")
    shared = set(train_ids) & set(test_ids)
    if shared:
        errors.append(f"{len(shared)} row identifiers appear in both sets")
    omitted = set(df.index) - set(train_ids) - set(test_ids)
    if omitted:
        errors.append(f"{len(omitted)} rows are in neither set")
    return errors


def safe_auc(y_true, scores) -> tuple[float | None, str | None]:
    if len(np.unique(y_true)) < 2:
        return None, "only one class present in the evaluated set"
    return float(roc_auc_score(y_true, scores)), None


# ---------------------------------------------------------------- training
def extract_features(array_path: str) -> np.ndarray:
    """(2,64,64) -> non-overlapping 8x8 block means -> (2,8,8) -> 128 C-order features."""
    arr = load_array(array_path)
    return arr.reshape(2, 8, 8, 8, 8).mean(axis=(2, 4)).ravel(order="C")


def evaluate_seed(df, feature_fn, split_fn, seed, test_size, model_cfg) -> dict:
    """Pure computation for one seed. Never raises; the outcome is in 'status'."""
    result = {"status": None, "status_reason": None, "stage": None, "metrics": {}, "split_assignments": None}
    start_wall, start_cpu = time.time(), time.process_time()
    stage = "split"
    try:
        train_ids, test_ids = split_fn(df, seed=seed, test_size=test_size)
        train_ids, test_ids = [str(i) for i in train_ids], [str(i) for i in test_ids]

        stage = "partition_validation"
        errors = validate_partition(df, train_ids, test_ids)
        if errors:
            result["metrics"] = {"partition_valid": False}
            result.update(status="invalid_partition", status_reason="; ".join(errors), stage=stage)
            return result

        train_patients = sorted(set(df.loc[train_ids, "patient_id"]))
        test_patients = sorted(set(df.loc[test_ids, "patient_id"]))
        overlap = sorted(set(train_patients) & set(test_patients))
        result["split_assignments"] = {
            "train_slice_ids": sorted(train_ids), "test_slice_ids": sorted(test_ids),
            "train_patients": train_patients, "test_patients": test_patients, "overlap_patients": overlap,
        }
        def class_counts(ids):
            rows = df.loc[ids, "label"].value_counts()
            patients = df.loc[ids].groupby("patient_id")["label"].first().value_counts()
            return {"rows": {str(k): int(rows.get(k, 0)) for k in (0, 1)},
                    "patients": {str(k): int(patients.get(k, 0)) for k in (0, 1)}}

        result["metrics"] = {
            "partition_valid": True,
            "train_row_count": len(train_ids),
            "test_row_count": len(test_ids),
            "class_counts": {"train": class_counts(train_ids), "test": class_counts(test_ids)},
            "overlap_count": len(overlap),
            "train_patient_count": len(train_patients),
            "test_patient_count": len(test_patients),
            "test_patient_overlap_proportion": len(overlap) / len(test_patients),
            "patient_auc": None, "patient_auc_unavailable_reason": None,
            "slice_auc": None, "slice_auc_unavailable_reason": None,
            "patient_aggregation": "arithmetic mean of held-out slice probabilities per patient",
        }

        stage = "model"
        y_train = df.loc[train_ids, "label"].to_numpy()
        if set(np.unique(y_train)) != {0, 1}:
            reason = f"training set contains only class(es) {sorted(np.unique(y_train).tolist())}"
            result["metrics"].update(patient_auc_unavailable_reason=reason, slice_auc_unavailable_reason=reason)
            result.update(status="unavailable", status_reason=reason, stage=stage)
            return result

        stage = "features"
        X_train = np.stack([feature_fn(i) for i in train_ids])
        X_test = np.stack([feature_fn(i) for i in test_ids])

        stage = "model"
        clf = KNeighborsClassifier(n_neighbors=model_cfg["n_neighbors"], weights=model_cfg["weights"],
                                   algorithm=model_cfg["algorithm"], metric=model_cfg["metric"],
                                   n_jobs=model_cfg["n_jobs"])
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, list(clf.classes_).index(1)]

        stage = "evaluation"
        test_df = df.loc[test_ids, ["patient_id", "label"]].copy()
        test_df["prob"] = probs
        patient_df = test_df.groupby("patient_id").agg(true_label=("label", "first"), mean_prob=("prob", "mean"))
        p_auc, p_reason = safe_auc(patient_df["true_label"], patient_df["mean_prob"])
        s_auc, s_reason = safe_auc(test_df["label"], test_df["prob"])
        result["metrics"].update(patient_auc=p_auc, patient_auc_unavailable_reason=p_reason,
                                 slice_auc=s_auc, slice_auc_unavailable_reason=s_reason)
        if p_auc is None:
            result.update(status="unavailable", status_reason=p_reason, stage=stage)
        else:
            result.update(status="completed", stage="done")
    except Exception as exc:  # recorded, never swallowed silently
        result.update(status="failed", stage=stage, status_reason=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc(limit=5))
    finally:
        result["timing"] = {"wall_time_seconds": time.time() - start_wall,
                            "cpu_time_seconds": time.process_time() - start_cpu,
                            "timing_boundary": TIMING_BOUNDARY}
    return result


def log_to_mlflow(manifest: dict) -> tuple[str | None, str | None]:
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}")
        if mlflow.get_experiment_by_name("invariant-demo-nsclc") is None:
            mlflow.create_experiment("invariant-demo-nsclc", artifact_location=(REPO_ROOT / "mlruns").as_uri())
        mlflow.set_experiment("invariant-demo-nsclc")
        with mlflow.start_run(run_name=manifest["run_id"]) as run:
            mlflow.log_params({"run_id": manifest["run_id"], "split_seed": manifest["split_seed"],
                               "split_function": manifest["split_function"], "status": manifest["status"]})
            for key in ("overlap_count", "patient_auc", "slice_auc"):
                value = manifest["metrics"].get(key)
                if value is not None:
                    mlflow.log_metric(key, value)
            return run.info.run_id, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def write_manifest(manifest: dict, artifact_root: Path) -> Path:
    out_dir = artifact_root / manifest["run_id"]
    out_dir.mkdir(parents=True, exist_ok=False)
    path = out_dir / "manifest.json"
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return path


def summarize(values: list) -> dict:
    defined = [v for v in values if v is not None]
    return {"values": values, "defined": len(defined), "total": len(values),
            "mean": float(np.mean(defined)) if defined else None,
            "population_std": float(np.std(defined, ddof=0)) if defined else None}


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    exp = config["experiment"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=exp["split_seeds"])
    parser.add_argument("--split-function", default=DEFAULT_SPLIT, help="module:function")
    parser.add_argument("--split-file", type=Path, help="path to a committed module outside this repo")
    parser.add_argument("--allow-dirty", action="store_true", help="exploratory runs only; recorded as dirty")
    args = parser.parse_args()

    repo = repo_state()
    if repo["sha"] is None:
        sys.exit("ERROR: not a git checkout with a commit.")
    if repo["dirty"] and not args.allow_dirty:
        sys.exit(f"ERROR: uncommitted changes {repo['dirty_paths']}; commit them or pass --allow-dirty.")

    split_fn, split_binding = resolve_split(args.split_function, args.split_file)
    if args.split_function != DEFAULT_SPLIT and not (split_binding["tracked"] and split_binding["clean"]):
        sys.exit(f"ERROR: {split_binding['module']} must be committed and unmodified in its repository "
                 f"(tracked={split_binding['tracked']}, clean={split_binding['clean']}).")

    df = load_metadata(METADATA_PATH)
    try:
        image_manifest_sha = verify_image_manifest(IMAGE_MANIFEST_PATH, metadata=df)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}\nRun: python -m demo_repo.data_gen --materialize")
    gen = json.loads(GENERATION_MANIFEST_PATH.read_text(encoding="utf-8")) if GENERATION_MANIFEST_PATH.exists() else {}

    bindings = {
        "git_sha": repo["sha"], "git_dirty": repo["dirty"],
        "metadata_hash": sha256_file(METADATA_PATH),
        "image_manifest_hash": image_manifest_sha, "image_manifest_verified": True,
        "generation_config_hash": sha256_file(CONFIG_PATH),
        "generation_manifest_hash": sha256_file(GENERATION_MANIFEST_PATH),
        "generator_git_sha": gen.get("generator_git_sha"),
        "dependency_versions": dependency_versions(),
    }

    def feature_fn(slice_id: str) -> np.ndarray:
        return extract_features(df.loc[slice_id, "array_path"])

    sweep_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    manifests = []
    for seed in args.seeds:
        result = evaluate_seed(df, feature_fn, split_fn, seed, exp["test_size"], exp["model"])
        manifest = {"run_id": uuid.uuid4().hex, "sweep_id": sweep_id, **bindings,
                    "split_function": args.split_function, "split_code": split_binding,
                    "split_seed": seed, "test_size": exp["test_size"], **result}
        manifest["mlflow_run_id"], manifest["mlflow_error"] = log_to_mlflow(manifest)
        write_manifest(manifest, ARTIFACT_ROOT)
        manifests.append(manifest)
        m = manifest["metrics"]
        detail = (f"N={m['overlap_count']} of M={m['test_patient_count']} | patient AUC="
                  + (f"{m['patient_auc']:.4f}" if m.get("patient_auc") is not None else "unavailable")) if m else ""
        print(f"seed {seed}: {manifest['status']} {detail} {manifest['status_reason'] or ''} | run {manifest['run_id']}")

    summary = {
        "sweep_id": sweep_id, **bindings,
        "split_function": args.split_function, "split_code": split_binding,
        "seeds": args.seeds,
        "runs": [{"seed": m["split_seed"], "run_id": m["run_id"], "status": m["status"],
                  "status_reason": m["status_reason"]} for m in manifests],
        "overlap_count": [m["metrics"].get("overlap_count") for m in manifests],
        "patient_auc": summarize([m["metrics"].get("patient_auc") for m in manifests]),
        "slice_auc": summarize([m["metrics"].get("slice_auc") for m in manifests]),
    }
    sweep_dir = ARTIFACT_ROOT / "sweeps"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    with open(sweep_dir / f"{sweep_id}.json", "x", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    pa = summary["patient_auc"]
    print(f"patient AUC: {pa['defined']}/{pa['total']} defined"
          + (f", mean {pa['mean']:.4f} ± {pa['population_std']:.4f}" if pa["mean"] is not None else "")
          + f" -> run_artifacts/sweeps/{sweep_id}.json")
    print("Next: commit run_artifacts/ so the runs are durable evidence.")


if __name__ == "__main__":
    main()
