"""Procedural, fully synthetic NSCLC-inspired paired CT/PET-like slice generator.

All patients, images and labels are fictional. Every numeric rule is read from
configs/data_generation.json; the draw order documented there is normative.

Usage (from the repository root):
    python -m demo_repo.data_gen                 # one-time freeze: clean committed checkout, no existing outputs
    python -m demo_repo.data_gen --check         # regenerate into a temp dir and compare with committed outputs
    python -m demo_repo.data_gen --materialize   # --check, then write missing arrays into data/slices/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "data_generation.json"
SIZE = 64
GRID = 8


# ---------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True,
                                   stderr=subprocess.DEVNULL).strip()


def require_committed_generator(config_path: Path) -> str:
    """Return HEAD's SHA; exit unless the checkout is clean and generator + config are tracked."""
    try:
        sha = _git("rev-parse", "--verify", "HEAD")
        status = _git("status", "--porcelain")
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit("ERROR: not a git checkout with a commit; generation requires a committed generator.")
    if len(sha) != 40:
        sys.exit("ERROR: could not resolve a full generator commit SHA.")
    if status:
        sys.exit("ERROR: working tree is not clean; commit or remove these first:\n" + status)
    for path in (Path(__file__).resolve(), config_path):
        if not path.is_relative_to(REPO_ROOT):
            sys.exit(f"ERROR: {path} is outside the repository; the config must be committed in it.")
        try:
            _git("ls-files", "--error-unmatch", path.relative_to(REPO_ROOT).as_posix())
        except subprocess.CalledProcessError:
            sys.exit(f"ERROR: {path.relative_to(REPO_ROOT).as_posix()} is not tracked by git.")
    return sha


def _dependency_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas", "scikit-learn"):
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            out[pkg] = None
    return out


# ---------------------------------------------------------------- geometry
_YY, _XX = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)  # pixel centers, x = column, y = row


def _ellipse(cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    return ((_XX - cx) / rx) ** 2 + ((_YY - cy) / ry) ** 2 <= 1.0


def _gaussian(cx: float, cy: float, sigma: float) -> np.ndarray:
    return np.exp(-((_XX - cx) ** 2 + (_YY - cy) ** 2) / (2.0 * sigma ** 2))


def _render_slice(g: dict, label: int, patient: dict, shift: np.ndarray,
                  position: float, ct_noise: np.ndarray, pet_noise: np.ndarray) -> np.ndarray:
    anat, tex, les, var, inten = g["anatomy"], g["texture"], g["lesion"], g["slice_variation"], g["intensity"]
    dx, dy = shift

    bcx, bcy = np.asarray(anat["body_center_xy"]) + patient["body_jitter"] + shift
    body = _ellipse(bcx, bcy, *anat["body_radii_xy_px"])
    lungs = np.zeros_like(body)
    for lx, ly in anat["lung_centers_xy"]:
        lungs |= _ellipse(lx + patient["body_jitter"][0] + dx, ly + patient["body_jitter"][1] + dy,
                          *anat["lung_radii_xy_px"])
    lungs &= body

    lesion_cx, lesion_cy = patient["lesion_center"] + shift
    scale = 1.0 - 0.15 * abs(position)
    lesion = _gaussian(lesion_cx, lesion_cy, les["radius_px_by_label"][str(label)] * scale)

    texture = patient["texture"] * body  # fixed in the pixel grid, masked to body

    ct = np.full((SIZE, SIZE), inten["ct_background"])
    ct[body] = inten["ct_body"]
    ct[lungs] = inten["ct_lung"]
    ct += tex["ct_amplitude"] * texture
    ct += les["ct_amplitude_by_label"][str(label)] * lesion
    ct += var["ct_noise_std"] * ct_noise

    pet = np.full((SIZE, SIZE), inten["pet_background"])
    pet[body] = inten["pet_body"]
    pet += tex["pet_amplitude"] * texture
    pet += les["pet_amplitude_by_label"][str(label)] * lesion
    pet += var["pet_noise_std"] * pet_noise

    return np.clip(np.stack([ct, pet]), 0.0, 1.0).astype(np.float32, order="C")


# ---------------------------------------------------------------- generation
def existing_outputs(config: dict, out_root: Path) -> list[str]:
    """Output files that already exist under out_root (arrays counted as one entry)."""
    outputs = config["outputs"]
    found = [outputs[k] for k in ("metadata", "image_manifest", "generation_manifest")
             if (out_root / outputs[k]).exists()]
    slices = out_root / "data" / "slices"
    if slices.is_dir() and any(slices.iterdir()):
        found.append("data/slices/*")
    return found


def generate_dataset(config: dict, out_root: Path = REPO_ROOT, patient_count: int | None = None) -> pd.DataFrame:
    """Generate arrays, metadata and the image manifest under out_root.

    Refuses to write if any output already exists. patient_count overrides the
    cohort size (tests only); it must be even so the two classes stay balanced.
    """
    out_root = Path(out_root)
    clash = existing_outputs(config, out_root)
    if clash:
        raise FileExistsError(f"refusing to overwrite existing outputs under {out_root}: {clash}")
    rnd, cohort, g = config["randomness"], config["cohort"], config["generation"]
    ident, outputs = config["identity"], config["outputs"]
    n = patient_count if patient_count is not None else cohort["patient_count"]
    if n % 2:
        raise ValueError("patient_count must be even for balanced labels")
    n_slices = cohort["slices_per_patient"]
    positions = np.linspace(-1.0, 1.0, n_slices)
    version = config["generator_version"]

    (out_root / "data" / "slices").mkdir(parents=True, exist_ok=True)

    # Draw order (1): labels, from an independent RNG, before any patient stream.
    labels = np.random.default_rng(rnd["label_seed"]).permutation(np.array([0] * (n // 2) + [1] * (n // 2)))

    records, manifest = [], []
    for p_idx in range(n):
        label = int(labels[p_idx])
        rng = np.random.default_rng(np.random.SeedSequence([rnd["data_seed"], p_idx]))
        j = g["anatomy"]["body_center_jitter_px"]

        # Draw order (2-5): patient-level parameters, independent of label.
        body_jitter = rng.uniform(-j, j, size=2)
        grid = rng.standard_normal((GRID, GRID))
        lung_idx = 0 if rng.random() < 0.5 else 1
        offset = rng.uniform(-3.0, 3.0, size=2)
        patient = {
            "body_jitter": body_jitter,
            "texture": np.kron(np.clip(grid, -3, 3) / 3.0, np.ones((SIZE // GRID, SIZE // GRID))),
            "lesion_center": np.asarray(g["anatomy"]["lung_centers_xy"][lung_idx]) + body_jitter + offset,
        }

        patient_id = ident["patient_id_template"].format(patient_index=p_idx)
        study_id = ident["study_id_template"].format(patient_id=patient_id)
        t = g["slice_variation"]["shared_translation_max_px"]
        for s_idx in range(n_slices):
            # Draw order (6): per-slice translation, then CT noise, then PET noise.
            shift = rng.uniform(-t, t, size=2)
            ct_noise = rng.standard_normal((SIZE, SIZE))
            pet_noise = rng.standard_normal((SIZE, SIZE))
            arr = _render_slice(g, label, patient, shift, positions[s_idx], ct_noise, pet_noise)

            slice_id = ident["slice_id_template"].format(patient_id=patient_id, slice_index=s_idx)
            rel_path = ident["array_path_template"].format(slice_id=slice_id)
            np.save(out_root / rel_path, arr, allow_pickle=False)
            records.append({"slice_id": slice_id, "patient_id": patient_id, "study_id": study_id,
                            "slice_index": s_idx, "array_path": rel_path, "label": label,
                            "generator_version": version})
            manifest.append({"array_path": rel_path, "sha256": sha256_file(out_root / rel_path)})

    df = pd.DataFrame(records, columns=outputs["metadata_columns"]).sort_values(outputs["metadata_sort"])
    meta_path = out_root / outputs["metadata"]
    df.to_csv(meta_path, index=False, lineterminator="\n")  # LF on every OS -> stable hash

    manifest.sort(key=lambda e: e["array_path"])
    img_manifest_path = out_root / outputs["image_manifest"]
    with open(img_manifest_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    return df


def write_generation_manifest(config_path: Path, out_root: Path, sample_count: int, generator_git_sha: str) -> dict:
    if not generator_git_sha or len(generator_git_sha) != 40:
        raise ValueError("generator_git_sha must be a full commit SHA")
    config = load_config(config_path)
    outputs = config["outputs"]
    gm = {
        "generator_version": config["generator_version"],
        "generator_git_sha": generator_git_sha,
        "configuration_path": config_path.relative_to(REPO_ROOT).as_posix(),
        "configuration_sha256": sha256_file(config_path),
        "metadata_sha256": sha256_file(out_root / outputs["metadata"]),
        "image_manifest_sha256": sha256_file(out_root / outputs["image_manifest"]),
        "dependency_versions": _dependency_versions(),
        "sample_count": sample_count,
    }
    path = out_root / outputs["generation_manifest"]
    with open(path, "x", encoding="utf-8", newline="\n") as f:  # "x": never overwrite
        json.dump(gm, f, indent=2)
        f.write("\n")
    return gm


# ---------------------------------------------------------------- modes
def freeze(config_path: Path) -> None:
    sha = require_committed_generator(config_path)
    config = load_config(config_path)
    clash = existing_outputs(config, REPO_ROOT)
    if clash:
        sys.exit(f"ERROR: outputs already exist: {clash}. The dataset is generated once. "
                 "Use --check or --materialize, or remove the old outputs in a separate commit.")
    # Generate into a staging directory, then move into place, so a crash never leaves partial data.
    staging = Path(tempfile.mkdtemp(prefix="data_gen_", dir=REPO_ROOT.parent))
    try:
        df = generate_dataset(config, staging)
        (REPO_ROOT / "data").mkdir(exist_ok=True)
        shutil.move(str(staging / "data" / "slices"), str(REPO_ROOT / "data" / "slices"))
        for key in ("metadata", "image_manifest"):
            shutil.move(str(staging / config["outputs"][key]), str(REPO_ROOT / config["outputs"][key]))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    gm = write_generation_manifest(config_path, REPO_ROOT, len(df), sha)
    print(f"Generated {len(df)} samples across {df['patient_id'].nunique()} patients at {sha[:12]}.")
    print(f"metadata sha256:       {gm['metadata_sha256']}")
    print(f"image manifest sha256: {gm['image_manifest_sha256']}")
    print("Next: commit data/metadata.csv, data/image_manifest.json and data/generation_manifest.json.")


def check(config_path: Path, materialize: bool) -> None:
    config = load_config(config_path)
    outputs = config["outputs"]
    for key in ("metadata", "image_manifest", "generation_manifest"):
        if not (REPO_ROOT / outputs[key]).is_file():
            sys.exit(f"ERROR: {outputs[key]} is missing; nothing to compare against.")
    gm = json.loads((REPO_ROOT / outputs["generation_manifest"]).read_text(encoding="utf-8"))

    problems = []
    if sha256_file(config_path) != gm["configuration_sha256"]:
        problems.append("configuration file differs from the one recorded at generation")
    if sha256_file(REPO_ROOT / outputs["metadata"]) != gm["metadata_sha256"]:
        problems.append("committed metadata differs from the generation manifest")
    if sha256_file(REPO_ROOT / outputs["image_manifest"]) != gm["image_manifest_sha256"]:
        problems.append("committed image manifest differs from the generation manifest")

    with tempfile.TemporaryDirectory(prefix="data_gen_check_") as tmp:
        tmp = Path(tmp)
        df = generate_dataset(config, tmp)
        if (tmp / outputs["metadata"]).read_bytes() != (REPO_ROOT / outputs["metadata"]).read_bytes():
            problems.append("regenerated metadata is not byte-identical")
        regen = json.loads((tmp / outputs["image_manifest"]).read_text(encoding="utf-8"))
        committed = json.loads((REPO_ROOT / outputs["image_manifest"]).read_text(encoding="utf-8"))
        if regen != committed:
            differing = sum(a != b for a, b in zip(regen, committed)) + abs(len(regen) - len(committed))
            problems.append(f"regenerated image hashes differ ({differing} entries); "
                            f"recorded deps {gm['dependency_versions']}, current {_dependency_versions()}")

        if problems:
            sys.exit("MISMATCH:\n- " + "\n- ".join(problems))
        print(f"OK: regeneration matches the frozen dataset ({len(df)} samples, "
              f"generated at {gm['generator_git_sha'][:12]}).")

        if materialize:
            dest = REPO_ROOT / "data" / "slices"
            dest.mkdir(parents=True, exist_ok=True)
            written = 0
            for entry in committed:
                target = REPO_ROOT / entry["array_path"]
                if target.exists():
                    if sha256_file(target) != entry["sha256"]:
                        sys.exit(f"ERROR: {entry['array_path']} exists with different bytes; not overwriting.")
                    continue
                shutil.copyfile(tmp / entry["array_path"], target)
                written += 1
            print(f"Materialized {written} missing arrays into data/slices/ (existing files verified).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="regenerate to a temp dir and compare; writes nothing")
    mode.add_argument("--materialize", action="store_true", help="--check, then write missing arrays")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.check or args.materialize:
        check(config_path, materialize=args.materialize)
    else:
        freeze(config_path)


if __name__ == "__main__":
    main()
