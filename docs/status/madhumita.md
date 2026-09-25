# Madhumita — Status

## Day 1 (2026-09-24)
### Done
- Env setup: Python 3.13.0, venv, all packages installed, pytest 9/9 pass
- Repo cloned, editable install (`pip install -e .`)
- Colab T4 GPU confirmed (Python 3.13.15, smp imports OK)
- Dataset A downloaded and inspected (`scripts/inspect_dataset.py`)
- `docs/data_sources.md` written: dataset comparison, primary decision, known limitations
- `docs/DECISIONS.md`: 7 decisions logged (dataset choice, split strategy, G1 provisional, Python version, encoder)
- Org created (TwoMuchCode), repo transferred
- PR merged into main

### Blocked
- Nothing

### Next
- Day 2: audit.py, manifest, group-aware splits, C1 preprocess, G1 answer

---

## Day 2 (2026-09-25)
### Done
- `ovanet/data/audit.py`: full audit — md5 dedup, phash near-duplicate clustering (threshold 10, Union-Find), brightness/contrast stats, train/test overlap check
- Findings: 1,921 unique images, 417 phash groups, 6 large groups (largest: 491 images with mixed labels), test folder = exact copy of train, image size perfectly separates classes (shortcut)
- `data/processed/manifest.csv`: 3,856 rows with md5, phash, group_id, brightness, contrast
- `docs/data_audit.md`: full audit report with shortcut analysis and G1 confirmation
- `ovanet/data/make_splits.py`: group-aware stratified split with greedy image-count balancing (seed 42, 70/15/15)
- `data/processed/splits.csv`: train 1345 (70.0%) / val 289 (15.0%) / test 287 (14.9%), no leakage
- `ovanet/preprocessing.py`: C1 contract — RAW (grayscale+resize+normalize) and ENH (bilateral denoise+CLAHE+normalize), quality check (blur, darkness, blank)
- `tests/test_preprocess.py`: 13 tests (shape/dtype/range, blank/dark rejection, RGB/RGBA input, RAW vs ENH differ)
- `tests/test_splits.py`: 6 tests (no group leakage, both classes in all splits, ratio check)
- **28/28 tests pass**
- **G1 CONFIRMED**: full transvaginal frames, not tight ovary crops. Ovary outline annotation needed for masked radiomics.
- Contracts sign-off: OK on all shapes, v0.1.1 addendum sent to Sahina

### Blocked
- Nothing

### Next
- Day 3: radiomics C3, paper baseline, shortcut check, annotation lists

---

## Day 3 (2026-09-26)
### Done
- `ovanet/radiomics.py`: C3 contract — mean, std, entropy, GLCM contrast, GLCM homogeneity (32 levels, dist 1, 4 angles averaged, symmetric, normalised)
- `tests/test_radiomics.py`: 10 tests — contract validation, known values (constant, noise), masked extraction, NaN safety
- `data/processed/features_v1.csv`: 1,921 rows, 5 radiomics features computed on full RAW image (no mask)
- `scripts/baseline_validation.py`: paper baseline + shortcut check
- **Wilcoxon results**: 4/5 features match paper direction. **Homogeneity flipped** (paper says lower in PCOS, we see higher). Contrast not significant (p=0.086).
- **Paper baseline A0 on TEST**: accuracy 61.3%, sensitivity 92.7%, specificity 26.5%, F1 71.6%. Much lower than paper's 100% — honest result on proper split.
- **Shortcut B2 on TEST: 100% accuracy** from image metadata alone (width, height, brightness, contrast). Dataset has a severe shortcut — must report B2 next to every metric.
- `scripts/select_annotation_images.py`: dev-15 (val, for me) and gold-test-15 (test, for Sahina) selected, stratified, one per group
- `data/annotations/dev15_list.csv` and `gold_test15_list.csv` saved
- **38/38 tests pass** (9 contracts + 13 preprocess + 6 splits + 10 radiomics)

### Blocked
- Nothing

### Next
- Annotate dev-15 images with Labelme (ovary outline + follicle outlines) — ~1-2 hours
- Day 4: classical follicle detector, follicle features, graph builder