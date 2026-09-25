# Demo plan - Invariant review of invariant-demo-nsclc

This file describes how the repository is used to demonstrate Invariant. It is kept out of Bob's
review context (`.bobignore`) so the review works from the study protocol and the measured
evidence, not from this script. It is not part of the study protocol.

## Two split states

| State | Where | Implementation |
|---|---|---|
| Baseline under review | `demo_repo/splits.py` on the PR head | `train_test_split` over row index labels, `shuffle=True`, `stratify=df["label"]`, `random_state=seed` |
| Reference comparator | `invariant/reference/reference_splits.py` (committed in the `invariant` repo) | `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)` with `groups=df["patient_id"]`, mapped back to `df.index` |

The baseline violates protocol INV-1 by design. The reference comparator lives outside this
repository so the reviewed history contains no grouped split before Bob's fix. It is used only to
measure the patient-independent AUC (Y) before the recorded review. After the fix, the runs at
`fix_sha` are the authoritative Y; the two should agree because the implementations are identical.

## Freeze and measurement (all zero-coin)

1. Commit code, configuration, protocol Markdown + PDF, dataset card and tests on a clean tree.
2. Remove any earlier generated data in its own commit.
3. `uv run --locked python -m demo_repo.data_gen` then commit `data/metadata.csv`,
   `data/image_manifest.json`, `data/generation_manifest.json`.
4. `uv run --locked python -m demo_repo.data_gen --check`.
5. `uv run --locked python -m demo_repo.train` (baseline, seeds 0-4): N and X.
6. `uv run --locked python -m demo_repo.train --split-file ../invariant/reference/reference_splits.py --split-function reference_splits:grouped_split`: Y.
7. `uv run --locked python scripts/write_release_record.py`, then commit `run_artifacts/` and
   `docs/release_record.md`.
8. Open the PR (a plausible feature change on a branch). Baseline CI must be green on its head.

## Acceptance sequence

1. Existing baseline tests pass on the PR head.
2. `verify_split_overlap` measures N patients in both sets at `head_sha`.
3. `build_report` records `blocked` at `head_sha` (Report A), before any edit.
4. Bob writes the grouped split and `tests/test_split_invariants.py` (fixed fixture, domain test,
   Hypothesis property test). The new tests fail on the old splitter with the overlap message.
5. At `fix_sha`, overlap is 0 and the tests pass.
6. Report B at `fix_sha` records `no_findings`; its history still lists the affected runs.
7. A branch that reintroduces the row-level split fails the required check and cannot merge.

`tests/test_split_invariants.py` does not exist before Report A. Its configured absence contributes
`review_required`; measured overlap contributes `blocked`, which takes precedence.

## Frozen between Report A and Report B

The generator, configuration, dataset, protocol, model and metrics do not change. The only
intended change is the split function plus the new regression tests. After Report B, rerun
`demo_repo.train` at `fix_sha` for the official post-fix AUC and record both sweep IDs in the
comparison artifact.
