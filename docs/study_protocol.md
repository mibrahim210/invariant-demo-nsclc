# Study protocol - Invariant synthetic imaging experiment
Version 1.0 | 25 September 2026 | Repository: invariant-demo-nsclc

## 1. Purpose and status
Evaluate whether a developer-tool check detects patient overlap in a deliberately buggy ML split, and verifies a patient-grouped correction. The application is an NSCLC-inspired synthetic imaging benchmark for ML correctness. The prediction target is a simulated binary imaging phenotype, not cancer diagnosis, survival or treatment response.

All patients, IDs, studies, images and labels are fictional and procedurally generated. No real patient records, personal information, client data, social-media data or public datasets are generator inputs. The CT-like and PET-like arrays are lightweight phantoms, not real scans, calibrated imaging or clinically validated volumes. This document specifies a planned experiment; no data generation, training, AUC, overlap or gate result is claimed here.

## 2. Frozen design and ownership
Use configs/data_generation.json as the machine-readable experiment specification. Keep invariant.toml limited to the proposal's analyzer contract; its split_seed=0 is the primary review seed. Training additionally evaluates seeds [0,1,2,3,4]. Additional JSON fields are a new contract to implement, not evidence that a current generator already supports them.

The cohort is 200 fictional patients, one study and eight related slices per patient: 1,600 samples. Each .npy sample is a C-order float32 array of shape (2,64,64): CT-like channel 0, PET-like channel 1. Values are finite and in [0,1]; there is no Hounsfield-unit or SUV calibration. The channels share anatomy, lesion coordinates and slice position.

Data seed: 20260925. Label seed: 20260926. Use NumPy Generator(PCG64), with independent patient streams from SeedSequence([data_seed, patient_index]). Shuffle 100 zero labels and 100 one labels with the independent label RNG, then assign to ascending patient indices. Every slice of a patient retains its patient's label. These counts and parameters are engineering choices, not clinical statistics.

## 3. Synthetic signal and identity
Persistent anatomy and patient-specific texture are independent of label. Controlled lesion size/intensity carries the simulated phenotype signal. Related slices retain patient texture and add modest geometry variation and independent noise. Paired channels use shared coordinates. Fixed intensity mappings require no cohort statistics.

The persistent texture deliberately enables patient memorization when slices from one patient enter both sets. This engineered correlation must be disclosed; it does not guarantee an AUC gap. The JSON fixes initial amplitudes and shapes. Before data freeze, document exact numerical implementation and RNG draw order in the versioned generator. Do not tune after observing final evaluation results.

Metadata columns: slice_id, patient_id, study_id, slice_index, array_path, label, generator_version. Load slice_id as the unique DataFrame index. Use neutral identity-based filenames; never encode labels in filenames or visible text. Model features contain image values only, never IDs, paths, seeds, labels or latent generation parameters.

## 4. Two-way split contract
Both states use the same frozen 1,600-row dataset and make_split(df, seed=0, test_size=0.2). Return index labels; consumers use .loc, never assume labels are positions. Sort input metadata by patient_id and slice_index before splitting.

Buggy baseline: train_test_split over row index labels, shuffle=True, stratify=df['label'], random_state=seed, test_size=0.2. This creates 1,280 train rows and 320 test rows; measure patient overlap rather than assuming a count.

Corrected state: GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed) with groups=df['patient_id']. Map returned positions back to df.index. With this equal-sized cohort, the planned partition is 160 train patients / 40 test patients, or 1,280 / 320 rows. GroupShuffleSplit does not stratify labels; record observed class counts and handle undefined AUC explicitly. Do not change seeds to obtain favorable balance.

Validate unique row IDs and nonmissing patient IDs before overlap measurement. Both output sets must be nonempty; all returned IDs must exist; neither set may duplicate IDs; no row may be shared; the union must cover every intended row exactly once. Invalid partitions are blocked, not interpreted as zero-overlap success. Grouping applies across all studies and derived samples of a patient. No validation or external cohort is silently removed from this two-way dataset.

## 5. Model and preprocessing
Use a small CPU-only KNeighborsClassifier with n_neighbors=5, weights='distance', algorithm='brute', metric='euclidean', n_jobs=1. This deterministic model has no random model seed. Use the same settings for both split states and every seed; no hyperparameter search.

For each channel, average each non-overlapping 8x8 pixel block, producing an 8x8 feature grid. Flatten the (2,8,8) result in C order to 128 features. This fixed per-image transform is not fitted on the cohort. Do not add PCA, scaling, learned normalization, augmentation or a CNN to this frozen experiment. Fit the neighbor classifier only on training rows; predict the probability associated with class label 1.

Preprocessing statistics, if introduced in a separately versioned experiment, must be fitted only on training data. Inference must apply the same deterministic preprocessing as evaluation. These are protocol requirements; core Invariant does not verify them.

## 6. Evaluation and aggregation
Primary metric: patient-level ROC AUC. For each patient with held-out rows, compute the arithmetic mean of ONLY that patient's held-out slice probabilities, paired with its single patient label. Never include training-slice predictions in this mean. Under the buggy split, these patients may already be represented in training: label this result as contaminated evaluation, not independent-patient performance.

Secondary metric: slice-level ROC AUC over held-out rows. Keep it separate from patient AUC. For either unit, fewer than two observed label classes makes AUC unavailable: store null and a reason, never zero. If training has one class, record that model evaluation is unavailable; do not reinterpret a probability column.

Report per seed: partition validity, train/test row and patient counts, class counts, overlapping patient count, and overlapping patients divided by unique test patients. Retain both AUC units, timings and run IDs. AUC is not accuracy.

## 7. Comparison rules
Seed 0 is the primary displayed comparison. Seeds 1-4 are predetermined sensitivity runs; retain all results, including failures and unavailable metrics. For each split state, report every seed's AUC plus the mean and population standard deviation of defined AUCs, with the defined/total count. Define the per-seed AUC difference as leaky minus grouped; summarize differences only for seeds with both values defined. Do not pool repeated predictions across seeds or describe seed variability as a confidence interval.

The two split strategies select different held-out populations. Their AUC difference illustrates this engineered benchmark; it is not a controlled causal estimate or clinical performance claim. Zero or reversed differences must also be reported. There is no target AUC or required AUC drop. The core acceptance evidence is measured overlap and regression prevention, independent of the metric gap.

## 8. Freeze and provenance
Before the review, implement and validate the generator, lock dependencies, and commit the configuration, generator, protocol Markdown and matching PDF, dataset card, metadata and sorted image manifest. Freeze them before Report A. Generator-code provenance refers to the clean generator commit used to generate bytes; the later snapshot commit may add metadata and manifests.

The image manifest is a JSON array sorted by relative POSIX array_path, with array_path and SHA-256 of each actual .npy file's bytes. Export data/generation_manifest.json with generator version and code SHA, configuration/metadata/image-manifest hashes, dependency versions and sample count. Never invent hashes. Regeneration must match the frozen manifest; any mismatch stops training. Training verifies all referenced image bytes before logging the manifest hash. Core overlap CI uses committed metadata and does not verify all image bytes.

Export durable run_artifacts/<run_id>/ records with clean code SHA, dependency versions, configuration hash, metadata hash, verified image-manifest hash, split function and settings, train/test row and patient assignments, metrics, timings, execution status and run ID. Record measured elapsed wall time and process CPU time around feature preparation, fit and evaluation, with the timing boundary included. Local MLflow output alone is insufficient evidence.

No generator, dataset, model or evaluation changes between Reports A and B. The intended correction changes the splitter and adds regression tests. Any later experimental change requires a new version and transparent rerun of both conditions. No external untouched cohort is included in v1.

## 9. Checked scope and acceptance sequence
Core scope: patient independence checked; train-only fitting and inference consistency not_checked. Modality alignment, clinical plausibility and clinical validity are not certified by these checks. The checked declaration is scope, not a successful result.

Keep ordinary dataset and split-shape tests on the buggy baseline. Do not add tests/test_split_invariants.py before Report A: its configured absence contributes review_required, while measured overlap contributes blocked and takes precedence. After Report A is preserved, Bob adds the group-aware correction and patient-independence regression tests. Measure and retain their failures on the old splitter and success on the corrected one.

The authoritative builder determines status and commit/hash bindings, refuses dirty target or analyzer trees, and preserves each report in an exclusive execution directory. Bob supplies explanation, not evidence values. Historical runs do not determine current status. Report A must not contain future fix results; a later comparison links explicit A/B report IDs and training-run IDs. Reintroducing the bug must fail the required CI check. no_findings means only that the implemented checks contributed no findings at that commit.

## 10. Readiness and limitations
Step 3 is complete when these specification files agree and the PDF matches its Markdown. This is a design freeze candidate, not an executed benchmark. Validate the generator and record actual evidence before claiming reproducibility, patient overlap, AUC values or passing checks. Do not describe any result as compliant, safe or permission to merge.
