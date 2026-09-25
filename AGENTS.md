# invariant-demo-nsclc

Synthetic NSCLC-inspired paired CT/PET-like slice demo. All patients, images and labels are fictional.

## Folders
- demo_repo/: data generator, dataset, split, training, inference
- tests/: pytest suite
- docs/: study protocol (source of the invariants)
- invariant.toml: Invariant checks configuration (do not edit during a review)

## Rules
- Reviews edit files in this repository only.
- Reports are created only by the build_report tool; never write report files.
- Static checks are risk indicators; only measured results are findings.
- Do not run training; the user runs it.
- Never state that anything is compliant, non-compliant, safe or OK to merge.