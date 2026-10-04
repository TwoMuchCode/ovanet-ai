# OvaNet AI

Explainable, decision-support research prototype for PCOS risk estimation from ovarian ultrasound images.

Built by Team TwoMuchCode for HackSpire '26.

> **Not a diagnosis. Not clinically validated. Not a medical device.** Research and education only.

## What it does

1. **Preprocess** — grayscale, resize to 256×256, quality check
2. **Vision** — classical CV detects the ultrasound scan region and individual follicles
3. **Radiomics** — texture features (mean, std, entropy, GLCM contrast/homogeneity), reproducing a published baseline method
4. **Spatial graph** — follicles become nodes in a kNN graph; 8 statistics describe their spatial distribution
5. **Fusion** — a calibrated logistic regression model combines radiomics + graph features into a 0–100 risk score with LOW/MODERATE/HIGH bands
6. **Explainability** — every score comes with real, coefficient-based feature-family attribution, not a black box
7. **Clinical checklist** — a separate, rule-based Rotterdam-criteria checklist for reviewer cross-reference (never fed into the model)
8. **Longitudinal tracking** — compares two visits for risk trend
9. **PDF export** — structured report for the record

## Honest findings (reported, not hidden)

- **Dataset confound:** labels in the underlying public dataset correlate strongly with image-acquisition metadata (width, height, brightness, contrast) — a model using *only* that metadata reaches AUROC ≈ 0.99. Our real radiomics+graph model (AUROC ≈ 0.96 on held-out test) does not exceed this baseline. We report the shortcut baseline next to every headline number rather than hiding it.
- **Follicle detection** uses classical computer vision (no pretrained segmentation model was available). Validated against 15 expert-annotated images: precision ≈ 0.13, recall ≈ 0.20 after tuning. A real limitation, not a solved problem.
- **"Ovary region"** is the detected ultrasound scan area, not a verified anatomical ovary boundary.
- A graph neural network (GCN) embedding was tested separately and found to carry no real signal (AUROC ≈ 0.50 alone) — excluded from the deployed model for this reason.

Full detail: `docs/DECISIONS.md`.

## Tech stack

Python, Streamlit, OpenCV, scikit-learn, PyTorch, ReportLab.

## Running locally

```bash
pip install -e .
streamlit run app/streamlit_app.py
```

## Team

TwoMuchCode — Sahina (contracts, GNN, fusion, evaluation, app) · Madhumita (data, preprocessing, radiomics, follicle/graph detection, vision)

## Disclaimer

OvaNet AI is a research prototype for decision support only. It does not diagnose, rule out, or treat any condition. All outputs require review by a qualified clinician.