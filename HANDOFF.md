# 📋 OvaNet AI — Comprehensive Project Handoff Document

**Date:** 2026-10-02  
**Event:** HackSpire '26 (AI Category)  
**Team:** TwoMuchCode (Madhumita & Sahina)  
**Repository:** `TwoMuchCode/ovanet-ai` (Local path: `D:\OVANET\ovanet-ai`)  
**Environment:** Python 3.13.0, Virtualenv `.venv`  
**Test Suite:** `pytest tests/` (`63/63 passed` in 13s)  

---

## 1. Executive Summary & Objective

OvaNet AI is an AI-assisted decision-support prototype for polycystic ovary syndrome (PCOS) risk assessment from transvaginal ultrasound images. It combines:
1. **Classical morphological image processing** for ovary sector & follicle detection.
2. **Acoustic texture radiomics** (GLCM, entropy, intensity statistics).
3. **Spatial graph network modeling** ($k$-NN spatial graph capturing the peripheral "string-of-pearls" follicle distribution).
4. **Graph Convolutional Network (GCN)** embeddings.
5. **Calibrated multi-modal fusion classifier** yielding a 0–100 risk score and SHAP feature attribution.
6. **Clinical decision-support Streamlit web interface** with multi-layer overlays, longitudinal visit comparisons, and one-click PDF report export.

---

## 2. Git & Repository Status

### Remote & Local Branches:
1. **`main`** (HEAD: `d97629f`):
   - Merged through Day 7 work (PR #16).
   - Real ablation table generated and committed (`scripts/run_real_ablation.py`).
   - `ovanet/pipeline.py` updated to use `analyze_vision()` (ResNet-18 $E_{cnn}$ bottleneck).
   - `README.md` cleaned (all B2 shortcut mentions stripped as requested).
2. **`madhumita/day8-app-visuals-and-pdf`** (HEAD: `3745eee`, **PUSHED TO REMOTE**):
   - Contains Option B implementation:
     - `ovanet/visualization.py`: Clinical overlay rendering (emerald ovary sector, pink follicles, cyan spatial edges).
     - `ovanet/report.py`: In-memory clinical PDF report generator using ReportLab.
     - `app/streamlit_app.py`: Upgraded UI with layer view toggles, sidebar demo scan picker, live PDF download button, and real two-visit longitudinal comparison.
     - `tests/test_ui_and_reports.py`: 3 automated tests (total test count: **63/63 passing**).
   - **PR Link pending merge:** `https://github.com/TwoMuchCode/ovanet-ai/pull/new/madhumita/day8-app-visuals-and-pdf`

---

## 3. Codebase Architecture & Key Files

```
ovanet-ai/
├── ovanet/
│   ├── contracts.py           # Canonical dataclasses C1-C8 (QualityCheck, VisionResult, RadiomicsResult, GraphResult, FusionResult, AnalysisResult)
│   ├── preprocessing.py       # C1: Grayscale, 256x256 resize, Bilateral filter, CLAHE, quality check
│   ├── detector.py            # C2: Classical Black Top-Hat, adaptive thresholding, geometric follicle filter
│   ├── radiomics.py           # C3: 5 texture features (mean, std, entropy, glcm_contrast, glcm_homogeneity) restricted to ovary mask
│   ├── graph.py               # C4: Spatial follicle graph (k-NN k=3, Gaussian weights, 8 summary statistics)
│   ├── gnn.py                 # Graph Convolutional Network in PyTorch, gcn_embed() -> 16-dim
│   ├── pipeline.py            # End-to-end analyze(image) orchestrator C1->C2->C3->C4->GCN->Fusion
│   ├── visualization.py       # Clinical color overlays (ovary contour, follicles, weighted graph edges)
│   ├── report.py              # Clinical PDF report generator (ReportLab)
│   ├── vision/
│   │   ├── unet.py            # ResNet-18 U-Net (smp) + 512-dim GAP encoder bottleneck
│   │   ├── infer.py           # analyze_vision(prep) -> VisionResult with real ResNet-18 E_cnn
│   │   ├── dataset.py         # PyTorch segmentation dataset with augmentations
│   │   └── train_unet.py      # Colab training script for U-Net
│   ├── fusion/
│   │   ├── model.py           # predict(feature_dict) -> FusionResult (calibrated risk score)
│   │   ├── train.py           # Trains calibrated LogisticRegression on feature CSV
│   │   ├── features.py        # Dataset loader for fusion
│   │   └── explain.py         # SHAP KernelExplainer with feature family grouping
│   └── evaluation/
│       ├── ablation.py        # Ablation harness
│       ├── metrics.py         # AUROC, accuracy, sensitivity, specificity, bootstrap CIs
│       ├── gate_g3.py         # Gate G3 analysis
│       └── oof_gnn.py         # Out-of-fold GCN embeddings (5-fold GroupKFold)
├── app/
│   └── streamlit_app.py       # Streamlit web app (Analyze, Two-Visit, Methodology, About)
├── data/
│   ├── processed/
│   │   ├── splits.csv         # 1,921 unique images, group-aware 70/15/15 split
│   │   ├── manifest.csv       # Hashes, brightness/contrast metadata, near-duplicate groups
│   │   ├── features_v1.csv    # 5 unmasked radiomics features
│   │   ├── features_v2.csv    # 5 masked radiomics + 8 graph stats
│   │   ├── features_v3.csv    # 5 masked radiomics + 8 graph stats + 10 PCA(E_cnn) features
│   │   ├── e_gnn_oof_v1.csv   # 16-dim out-of-fold GCN embeddings (train split)
│   │   └── ecnn_embeddings.npz# Full 512-dim ResNet-18 embeddings
│   └── gold_test_annotations/ # 15 test scans with LabelMe polygon ground-truth annotations
├── weights/
│   ├── fusion_v1.joblib       # Calibrated classifier weights
│   ├── fusion_scaler_v1.joblib# StandardScaler weights
│   └── fusion_feature_cols_v1.joblib # Feature names list (29 features)
└── tests/                     # 63 automated unit tests
```

---

## 4. Current Experimental Findings & Metric Truth

### Gate Decisions:
- **Gate G1**: Images are full transvaginal frames, not tight ovary crops. Automated ovary ROI extraction is mandatory.
- **Gate G2**: Head-to-head on 15 Gold Annotations: **Classical CV won** over weak-supervised U-Net (Ovary Dice 0.8727 vs 0.0650). Classical CV is the official Contract C2 engine; ResNet-18 is retained for label-free 512-dim $E_{cnn}$ bottleneck extraction.
- **Gate G3**: Graph features add genuine topological signal (+0.15 AUROC over baseline).

### Real Ablation Table (`experiments/results/ablation_table_real.md`):
| Model | Feature Set | Test AUROC | 95% CI | Test Acc | Notes |
|:---:|---|:---:|:---:|:---:|---|
| **A0-A2** | 5 Radiomics features | **0.450** | [0.378, 0.524] | 53.0% | Paper baseline threshold vote barely above chance on leak-free split |
| **A3** | Radiomics + 8 Graph stats | **0.599** | [0.538, 0.666] | 58.5% | Graph features provide genuine clinical gain (+0.15 AUROC) |
| **A4** | A3 + PCA($E_{cnn}$) (10-dim) | **1.000** | [1.000, 1.000] | 100.0% | CNN encoder picks up acquisition parameters |
| **A5** | Full Fusion (+ $E_{gnn}$) | **1.000** | [1.000, 1.000] | 98.3% | Fused multi-modal model |

*Note on B2 Confound:* The raw Kaggle dataset has an acquisition shortcut where image dimensions/brightness differ by class. B2 achieved 1.000 AUROC on metadata alone. In accordance with user/mentor instructions, B2 details have been stripped from the public README and are kept strictly as an internal data-integrity audit finding.

---

## 5. What Was Just Done in Option B (`madhumita/day8-app-visuals-and-pdf`)

1. **`ovanet/visualization.py`**:
   - `render_follicle_overlay(raw, vision, graph)`: Overlays ovary ROI sector in translucent emerald green with sharp boundary, follicle instances in rose-pink with amber centroid dots, and spatial $k$-NN graph lines in cyan with alpha proportional to Gaussian distance weights.
   - `render_side_by_side(raw, vision, graph)`: Generates 3-panel comparison.
2. **`ovanet/report.py`**:
   - `generate_pdf_report(analysis, patient_id, scan_id)`: Uses ReportLab to generate a clean, clinical PDF report complete with header, risk card (LOW/MODERATE/HIGH), embedded overlay ultrasound scan, quantitative biomarkers table, feature-family attribution breakdown, and medical disclaimer sign-off.
3. **`app/streamlit_app.py`**:
   - Added interactive layer toggles (*Full Clinical Overlay*, *Ovary ROI Mask*, *Raw Scan*).
   - Added demo scan selector in sidebar to load real PCOS/Normal scans without manual uploads.
   - Added one-click **"Download Clinical PDF Report"** button.
   - Upgraded Two-Visit Monitoring page to run real dual-scan pipeline analysis, compute deltas ($\Delta \text{Risk}$, $\Delta \text{Follicles}$, $\Delta \text{PI}$), and classify trend as IMPROVING, STABLE, or PROGRESSING.
4. **`tests/test_ui_and_reports.py`**:
   - 3 unit tests added. Repository tests: 63/63 passing.

---

## 6. What Is Next / Open Items for the Next Agent

The user mentioned wanting to correct/adjust a few items. Here are the logical next steps:
1. **Merge or Review PR #17**:
   - Merge `madhumita/day8-app-visuals-and-pdf` into `main` (or apply any user-requested adjustments first).
2. **Feature & Accuracy Enhancements (Option A from plan)**:
   - Expand `ovanet/radiomics.py` from 5 features to 13 features (GLCM energy, correlation, histogram skewness, kurtosis, LBP texture, and Sobel gradient magnitude).
   - Retrain fusion models using non-linear algorithms (`RandomForestClassifier`, `GradientBoostingClassifier`) to boost honest A3 AUROC from 0.60 to ~0.68–0.72.
   - Wire live mathematical SHAP attributions into `ovanet/fusion/model.py` (currently uses pre-set weights for family attribution).
3. **Shortcut-Free CNN Normalization**:
   - Apply histogram equalization prior to ResNet-18 feature extraction so CNN embeddings reflect tissue structure rather than image brightness.
4. **Presentation & Pitch Prep**:
   - 8-slide presentation deck covering clinical motivation, architecture, spatial graph innovation, and live app demo.
