# 🩺 OvaNet AI

**AI-assisted PCOS risk assessment from ovarian ultrasound — decision-support research prototype.**

> ⚠️ **This tool does not diagnose PCOS. It is not a validated medical device. It is a research prototype for educational and demonstration purposes only.**

---

## What It Does

OvaNet AI analyzes transvaginal ovarian ultrasound images to estimate PCOS (Polycystic Ovary Syndrome) risk using a multi-modal fusion pipeline:

```
Ultrasound Image
    ↓
1. Preprocess     → Grayscale, resize 256×256, quality check
    ↓
2. Vision (C2)    → Detect ovary boundary + individual follicles
    ↓
3. Radiomics (C3) → 5 texture measurements (mean, std, entropy, GLCM)
    ↓
4. Graph (C4)     → Spatial follicle graph (k-NN, 8 statistics)
    ↓
5. GCN            → Graph neural network embedding (64-dim)
    ↓
6. Fusion (C6)    → Calibrated risk score (0–100) + SHAP explanation
    ↓
Streamlit Web App → Overlay visualization + risk gauge + report
```

## Key Technical Highlights

- **Multi-modal fusion**: Combines classical image processing, radiomics, graph neural networks, and CNN embeddings
- **Explainable AI**: SHAP-based feature attribution shows which signal family drove each prediction
- **Honest evaluation**: B2 shortcut baseline exposed — image metadata alone achieves AUROC=1.0, documenting a critical dataset confound
- **Leak-safe methodology**: Group-aware splits, out-of-fold GCN embeddings, PCA fitted on train only
- **Gate-driven development**: Systematic comparison checkpoints (G1: image type, G2: classical vs U-Net, G3: graph contribution)

## Known Limitations

- **Dataset confound (B2)**: The primary Kaggle dataset has systematic differences in image resolution between PCOS and normal classes. All models may be influenced by this acquisition bias rather than true medical signal.
- **Not clinically validated**: No IRB approval, no prospective evaluation, no regulatory clearance.
- **Single-center data**: All images come from one source; generalization to other scanners/populations is unproven.

## Project Structure

```
ovanet-ai/
├── ovanet/                     # Core Python package
│   ├── contracts.py            # Frozen data shapes (C1–C8)
│   ├── preprocessing.py        # C1: Image preprocessing
│   ├── detector.py             # C2: Classical follicle detection
│   ├── radiomics.py            # C3: Texture feature extraction
│   ├── graph.py                # C4: Spatial follicle graph
│   ├── gnn.py                  # Graph Convolutional Network
│   ├── pipeline.py             # End-to-end orchestration
│   ├── vision/                 # Deep learning vision module
│   │   ├── unet.py             # ResNet-18 U-Net architecture
│   │   ├── infer.py            # Unified C2 inference API
│   │   ├── dataset.py          # PyTorch dataset for training
│   │   └── train_unet.py       # U-Net training engine
│   ├── fusion/                 # Risk prediction + explanation
│   │   ├── model.py            # Fusion MLP + sklearn predict
│   │   ├── train.py            # Fusion training + calibration
│   │   ├── features.py         # Feature loading utilities
│   │   └── explain.py          # SHAP explanation engine
│   └── evaluation/             # Testing & comparison
│       ├── ablation.py         # A0–A5/B1/B2 comparison
│       ├── metrics.py          # Accuracy, AUROC, F1, bootstrap CI
│       ├── gate_g3.py          # GNN contribution analysis
│       └── oof_gnn.py          # Out-of-fold GCN embeddings
├── app/
│   └── streamlit_app.py        # Interactive web application
├── data/
│   ├── processed/              # Feature CSVs, splits, embeddings
│   └── gold_test_annotations/  # 15 expert-annotated test images
├── scripts/                    # Data generation & evaluation scripts
├── notebooks/colab/            # Colab GPU training notebooks
├── tests/                      # 60 automated tests
├── experiments/results/         # Experiment reports (markdown)
├── docs/                       # Decisions log, contracts, status
└── weights/                    # Trained model files
```

## Setup

### Prerequisites
- Python 3.10+ (tested on 3.13.0)
- No GPU required for inference (GPU recommended for U-Net training)

### Installation
```bash
git clone https://github.com/TwoMuchCode/ovanet-ai.git
cd ovanet-ai
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -e .
```

### Run Tests
```bash
pytest tests/ -v
```

### Run the App
```bash
streamlit run app/streamlit_app.py
```

### Run on Google Colab (GPU training)
Upload `notebooks/colab/train_unet.ipynb` and the pseudo-mask zip to Colab.

## Methodology

### Gate Decisions

| Gate | Question | Answer | Evidence |
|------|----------|--------|----------|
| **G1** | Are images cropped or full-frame? | Full transvaginal frames | Visual inspection of 20 images |
| **G2** | Classical CV or U-Net for segmentation? | **Classical CV** (Dice 0.87 vs 0.07) | 15 gold-test annotations |
| **G3** | Do graph features help? | Marginal (+0.01 AUROC) but contaminated by B2 | Ablation A3 vs A2 |

### Ablation Table

See `experiments/results/ablation_table_real.md` for the full comparison of:
- **A0–A2**: Radiomics baselines
- **A3**: + follicle graph statistics
- **A4**: + CNN encoder embeddings (PCA)
- **A5**: + GNN embeddings (full fusion)
- **B1**: CNN-only baseline
- **B2**: ⚠️ Shortcut baseline (image metadata only) — exposes dataset confound

## Team

**TwoMuchCode** — HackSpire '26, AI Category

| Member | Role | Track |
|--------|------|-------|
| **Madhumita** | Vision & Data | Preprocessing, detection, radiomics, graph, U-Net |
| **Sahina** | Integration & ML | GCN, fusion, evaluation, app, deployment |

## License

Research prototype — not for clinical use. Dataset license pending verification.