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
- Day 4: classical follicle detector, follicle features, graph builder

---

## Day 4 (2026-09-30)
### Done
- `ovanet/detector.py`: C2 classical follicle detection
  - Robust ultrasound sector estimation excluding machine metadata headers/footers
  - Morphological Black Top-Hat isolating hypoechoic round fluid structures
  - Adaptive/percentile thresholding within ovary ROI
  - Geometric filtering: area [12, 800], solidity >= 0.65, eccentricity <= 0.85, mean intensity <= 0.55
  - Returns `VisionResult` conforming to Contract C2 (`ovary_mask`, `follicle_labels`, `follicles`, `e_cnn`, `seg_method="classical"`)
- `ovanet/graph.py`: C4 spatial graph builder
  - 5-dim normalized follicle node features $[x_{rel}, y_{rel}, area_{rel}, r_{rel}, eccen]$
  - Symmetrized $k$-NN adjacency ($k = \min(3, n-1)$) with Gaussian distance weights $w = \exp(-d^2 / \tau)$
  - Exact 8 graph statistics matching `GRAPH_STAT_NAMES`: `n_follicles`, `mean_nn_dist_rel`, `std_nn_dist_rel`, `mean_degree`, `clustering_coeff`, `peripheral_index`, `size_cv`, `follicle_area_fraction`
  - Safe $n < 3$ edge case handling (returns zeros, empty edges, `valid=False`)
- `tests/test_detector.py`: 5 tests (sector mask estimation, synthetic dots detection, blank image handling, noise robustness, custom mask)
- `tests/test_graph.py`: 4 tests (normal graph properties, $n < 3$ edge cases, end-to-end integration test feeding directly into Sahina's `gcn_embed`)
- **51/51 tests pass** across the entire repository
- Tested on real gold test images: successfully detects 17–22 follicles on real PCOS ultrasound frames

### Blocked
- Nothing

### Next
- Day 5: Train U-Net on Colab GPU for automated ovary & follicle segmentation, compare against classical detector

---

## Day 5 (2026-10-01)
### Done
- `scripts/generate_pseudo_masks.py`: Weak-supervision mask generator using Day 4 classical detector
  - Generated 1,345 train and 289 val pseudo-masks (2 channels: ovary sector and detected follicles)
  - Collision-free naming `<folder_split>_<label>_<stem>.npz`
  - Packaged into `data/pseudo_masks_train_val.zip` (2.34 MB) for fast upload to Google Colab / Drive
  - Mean detected follicles: ~15.9 per train image, ~16.2 per val image
- `scripts/make_features.py`: Full C1 -> C2 -> C3 -> C4 pipeline
  - Extracted 5 masked radiomics features + 8 graph statistics across all 1,921 images
  - Saved `data/processed/features_v2.csv`
  - Verified clinical signal: PCOS images show higher follicle counts (17.95 vs 14.62) and higher entropy (4.31 vs 4.05)
- `ovanet/vision/unet.py`: ResNet-18 U-Net architecture (`smp.Unet`)
  - 2 output channels: ovary sector mask + follicle mask
  - Extract 512-dim bottleneck embedding ($E_{cnn}$) via Global Average Pooling matching Contract C2 (`E_CNN_DIM=512`)
- `ovanet/vision/dataset.py`: PyTorch dataset with geometric and photometric augmentations (flips, slight rotations, contrast)
- `ovanet/vision/train_unet.py`: Training engine with combined `BCEWithLogitsLoss` + `DiceLoss`
- `notebooks/colab/train_unet.ipynb`: Complete, standalone Colab GPU notebook for 15-epoch training and visualization
- `tests/test_unet.py`: 4 unit tests verifying U-Net forward pass, $E_{cnn}=512$ embedding shape, inference mask helper, and dataset augmentation
- **55/55 tests pass** across the entire repository

### Blocked
- Nothing

### Next
- Day 6: Evaluate U-Net vs Classical detector on Sahina's 15 gold-test annotations (Gate G2), export E_cnn embeddings, build C2 infer API, generate features_v3

---

## Day 6 (2026-10-01)
### Done
- `scripts/evaluate_gate_g2.py`: Head-to-head evaluation on 15 Gold-Test images (`data/gold_test_annotations/`)
  - **Gate G2 Resolved**: Classical CV detector wins with **0.8727 Ovary Mask Dice** (vs 0.0650 for U-Net) and **0.4622 Combined Mean Dice** (vs 0.0466)
  - Result documented in `experiments/results/gate_g2_evaluation.md` and `docs/DECISIONS.md`
- `ovanet/vision/infer.py`: Implemented unified Contract C2 entrypoint `analyze_vision(prep) -> VisionResult`
  - Runs classical detector for ovary sector & follicle instances
  - Extracts 512-dimensional bottleneck embedding $E_{cnn}$ from ResNet-18 encoder via Global Average Pooling
  - Sets `seg_method="classical"`
- `scripts/make_features_v3.py`: Full dataset feature extraction (1,921 images)
  - Extracted 512-dim $E_{cnn}$ embeddings and saved to `data/processed/ecnn_embeddings.npz` (3.56 MB)
  - Fit 10-component PCA on TRAIN split only (Rule L2, 79.22% variance explained)
  - Saved `data/processed/features_v3.csv` (723 KB) containing 5 masked radiomics + 8 graph stats + 10 PCA $E_{cnn}$ components
  - Compared unmasked vs masked radiomics in `experiments/results/radiomics_mask_comparison.md` (shows masking eliminates background border bias)
- `tests/test_vision.py`: 3 unit tests verifying Contract C2 return types, shapes, and blank image handling
- **58/58 tests pass** across the entire repository

### Blocked
- Nothing

### Next
- Day 7: Train standalone ResNet18 CNN classifier baseline (B1) on Colab + generate Grad-CAM heatmaps for demo panel

---

## Day 7 (2026-10-02)
### Done
- Fixed `ovanet/pipeline.py`: now uses `analyze_vision()` (real ResNet-18 E_cnn) instead of `detect_follicles()` (placeholder E_cnn)
- `scripts/run_real_ablation.py`: Full ablation table on REAL features (not synthetic)
  - A0-A2 (radiomics only): AUROC=0.450 — confirms paper baseline is barely above chance on proper splits
  - A3 (+ graph stats): AUROC=0.599 — graph features add ~15 points
  - A4 (+ PCA E_cnn): AUROC=1.000 — but matches B2 shortcut, contaminated
  - A5 (full fusion): AUROC=1.000, Acc=0.983 — slight drop from E_gnn zero-fill in test
  - B1 (CNN only): AUROC=1.000 — CNN embedding alone captures shortcut
  - B2 (shortcut): AUROC=1.000, Acc=1.000 — image metadata alone is perfect
  - **Key finding**: A0-A3 (radiomics + graph) are the ONLY honest results. A4+ are contaminated by the same acquisition bias B2 exploits.
- `experiments/results/ablation_table_real.md`: Full markdown report with bootstrap 95% CIs
- Rewrote `README.md`: full project documentation (pipeline diagram, setup, methodology, structure, limitations, team)
- Code review fixes from Day 6: fallback E_cnn uses local RandomState, torch.load weights_only, detector.py docstrings
- **60/60 tests pass**

### Blocked
- Nothing

### Key Insight for Judges
The honest story is NOT "we got 100% accuracy." The honest story is:
1. We DISCOVERED that the dataset has a critical acquisition confound (B2)
2. Our radiomics + graph pipeline (A3) achieves 0.60 AUROC on genuinely medical features
3. We built the architecture for multi-center validation where B2 wouldn't apply
4. This is what responsible AI looks like — transparent about limitations
