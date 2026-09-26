---
name: data-leakage
description: Verify and fix patient-level train/test leakage against a study protocol — group keys such as patient_id appearing in both splits. Use when reviewing dataset, split, or training code in a repository with invariant.toml.
---

# Data leakage review

1. Protocol: list invariants; mark checked vs not_checked (core checks only
   patient independence).
2. `inspect_split`: risk only. `stratify=` balances labels, not patients.
3. `verify_split_overlap`: the finding is the measured count at this commit.
4. `find_invariant_tests`: three-state coverage.
5. `build_report` before any edit.
6. Fix with `GroupShuffleSplit` on the group key (mention `StratifiedGroupKFold`
   if label balance matters); keep the signature and return index labels.
7. Tests: fixed fixture (20 patients x 3 images, patient-level labels), a domain
   test, and a Hypothesis property test constrained to the splitter's domain.
8. Historical runs are reported by the builder; they never decide current status.
