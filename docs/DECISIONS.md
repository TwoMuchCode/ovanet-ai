# OvaNet AI — Decisions Log

Running record of every decision, finding, and fix that affects contracts, evaluation, or reported results. Anything that changes a number a judge might see is logged here.

---

## Contracts & architecture

- **Contracts frozen v0.1 → v0.2** (Day 2): IMAGE_SIZE=256, E_CNN_DIM=512, GRAPH_FEATURE_DIM=5, GCN_EMBED_DIM=16, RADIOMICS_FEATURE_ORDER and GRAPH_STAT_NAMES fixed. Addendum added: coordinate convention (x=column, y=row), adjacency kNN k=min(3,n-1) symmetrized, Gaussian edge weights, peripheral_index formula, n<3 safe defaults.
- **Contract objects use attribute access**, not dict access (e.g. `result.raw`, not `result["raw"]`) — logged after one debugging incident.
- `detect_follicles(raw)` takes the raw array, not the full `PreprocessResult` object — logged after one debugging incident.

## Dataset findings

- **Kaggle "test" folder is an exact duplicate of "train"** (confirmed via md5/phash dedup). All real splits use only `train/` paths; `test/` is never used. `splits.csv`: 1,345 train / 289 val / 287 test, group-aware, deduplicated.
- **Paper baseline reproduction:** A0 = 61.3% accuracy on real test (vs. the source paper's unreplicated 100%, which was sample-specific). 4/5 radiomics features matched the paper's claimed direction; homogeneity flipped.
- **CRITICAL FINDING — dataset shortcut confirmed (B2):** a logistic regression using *only* image width, height, brightness, and contrast — no medical content at all — achieves **AUROC = 0.990–1.0, accuracy ≈ 76–99%** depending on split. This means the dataset's infected/notinfected labels correlate strongly with image acquisition metadata, not confirmed ovarian pathology alone. This is the single most important finding in the project and is reported next to every headline result, never hidden.
- Train/test class balance differs (35% vs 53% PCOS) — a known, unresolved distribution shift.
- One near-duplicate image group was found with ~491 mixed-label images, suggesting the near-duplicate detection threshold (phash) may be too loose. Not resolved.

## Vision / follicle detection

- `estimate_ovary_mask()` detects the **ultrasound scan fan/sector**, not a true anatomical ovary boundary. Gold-test Dice (~0.71) against hand-annotated polygons is a measure of scan-region overlap, not real organ segmentation — labeled honestly as "ROI detection" everywhere in the app and docs.
- **Follicle detector tuning (2026-10-03):** Baseline (untuned) detector scored F1 = 0.073 against gold-test-15 ground truth (precision 0.044, recall 0.200) — it was firing 4–30 false "follicles" even on images with zero real follicles (8 of 15 gold images).
  - Grid search over percentile threshold and minimum blob area alone raised F1 to 0.117 (percentile 80, min_area 25) — confirmed the detector was near its ceiling on brightness-only filtering.
  - Added a **local-contrast boundary filter** (candidate blob's interior must be ≥0.08 darker than the tissue ring immediately surrounding it — real follicles have a sharp anechoic boundary, generic speckle noise does not). Re-grid-searched: best result percentile=75, min_area=40, min_contrast=0.08 → **F1 = 0.157** (precision 0.130, recall 0.200). Applied to `ovanet/detector.py`.
  - **Honest ceiling:** even after tuning, ~87% of detections remain false positives. This classical approach (black top-hat + contour filtering + local contrast) has a real limit; a further improvement would need shape-based matching (e.g. Hough circle transform) or a trained segmentation model, not more threshold tuning. Reported as a measured, partial improvement, not a fix.
- Gate G2 (classical vs. U-Net segmentation comparison) result should be treated cautiously: trained U-Net weights were not confirmed present in the final evaluation environment; the comparison may have silently fallen back to an untrained ImageNet encoder.

## GCN / graph embedding

- A GCN (`ovanet/gnn.py`, pure PyTorch) was built and trained only via disposable, per-fold models during out-of-fold embedding generation (`ovanet/evaluation/oof_gnn.py`, rule L1 compliance for leak-safety). **No final, single GCN model was ever trained and saved.**
- An early fusion model was trained including these OOF `e_gnn_*` features, meaning it carried real coefficients on what was effectively noise. At inference, the live pipeline (which has no trained GCN) was filling these with the training-set mean — a value that encoded nothing meaningful, repeated identically on every real prediction.
- **Diagnostic check added:** evaluated `e_gnn` features alone against real labels. Result: **AUROC ≈ 0.496 (random)**. Confirms the embeddings carry no real signal.
- **Decision:** `e_gnn` excluded entirely from the deployed fusion model (v2). `ovanet/gnn.py` now has an `is_trained()` guard (checks for `weights/gcn_v1.pt`, which does not exist) so the app never displays an untrained embedding as if it were a real result.

## Fusion model

- **Fusion v1 issues found and fixed (2026-10-03, fusion v2):**
  - v1 was evaluated on a random 80/20 re-split of the train set only — the real, held-out 289-row val set was never actually used in evaluation.
  - v1's calibration cross-validation was plain 5-fold, not group-aware — risk of near-duplicate images leaking across calibration folds.
  - v1's risk bands (LOW<33, MODERATE<66, HIGH≥66) were arbitrary fixed numbers, not derived from the model's actual score distribution.
  - v1's `predict()` returned hard-coded placeholder attribution (0.3/0.2/0.2/0.15/0.1, static) and a literal `"placeholder"` top feature — never real output.
- **Fusion v2 (current):** trained on radiomics + graph stats only (13 features, e_gnn excluded per above). Uses the real train/val/test splits. Calibration CV is `GroupKFold`. Risk bands derived from train-score terciles (LOW<3.7, MODERATE<53.5, HIGH≥53.5). Attribution is computed directly from the fitted model's real coefficients (coefficient × scaled value, summed per feature family) — no placeholders.
- **Fusion v2 performance:**
  | Split | AUROC | Accuracy |
  |---|---|---|
  | Train | 0.999 | 0.984 |
  | Val (289, real held-out) | 0.974 | 0.917 |
  | Test (287, real held-out) | 0.961 | 0.882 |


## Ablation table

- **Original ablation table issues found:** A0, A1, A2 were byte-identical rows (all plain LogReg on radiomics, mislabeled as distinct methods); no feature scaling applied before LogisticRegression, despite radiomics and PCA(E_cnn) being on very different scales; confidence intervals resampled individual rows rather than groups (not leak-safe); e_gnn was zero-filled for test rows inside the "full fusion" (A5) row, silently mixing real and fabricated inputs in one number.
- **Corrected ablation table (2026-10-03):** removed duplicate A0/A1/A2 rows; added `StandardScaler` fit on train only for every row; switched to group-aware bootstrap CIs (`bootstrap_ci_by_group`); removed A4/A5 (both depended on e_gnn); added a standalone `E_GNN_ONLY` diagnostic row instead of burying e_gnn inside a "full" model.
- **Corrected results (test set):**
  | Model | Description | AUROC |
  |---|---|---|
  | A0 | Radiomics only | 0.909 |
  | A1 | Radiomics + graph stats | 0.972 |
  | A2 | Radiomics + graph + PCA(E_cnn) | 1.000 |
  | B1 | CNN embedding only (PCA E_cnn, untrained ImageNet) | 1.000 |
  | E_GNN_ONLY (diagnostic) | GCN embedding alone | 0.496 |


## Descoped items (explicitly, not silently dropped)

- **Fine-tuned CNN baseline + Grad-CAM:** not built. Given every CNN-embedding row already hits a suspected-confounded AUROC of 1.0, training a real fine-tuned classifier would very likely just reconfirm the same shortcut rather than add insight — judged not worth the remaining time.
- **True ovary organ segmentation:** not built. No pixel-level training masks were available; only 15 hand-annotated validation images existed, insufficient to train a segmentation model from scratch.
- **De-confounded or multi-source dataset:** not pursued. Would require either discarding most of the available training data (stratifying out the confound) or sourcing new data — not feasible in the project timeline.

## Deployment

- `requirements.txt` initially missing `opencv-python-headless` (needed by `detector.py`/`visualization.py`'s `cv2` usage) and `joblib` (used directly by fusion save/load) — added.
- Streamlit Cloud does not run `pip install -e .` automatically from `pyproject.toml` — added `-e .` as the first line of `requirements.txt` so the `ovanet` package itself installs alongside its dependencies.
- `app/streamlit_app.py` import of `ovanet_ui` made robust to working-directory differences via an explicit `sys.path.insert` at the top of the file.

---
