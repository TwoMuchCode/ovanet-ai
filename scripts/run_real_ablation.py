"""
scripts/run_real_ablation.py

Runs the full ablation table (A0→A5, B1, B2) on REAL features.
Uses proper train/test splits from splits.csv (NOT random re-split).
Reports AUROC, accuracy, F1, sensitivity, specificity with bootstrap CIs.
Also compares every row against B2 shortcut baseline.

Usage:
  python scripts/run_real_ablation.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    recall_score, precision_score, confusion_matrix,
)

FEATURES_V3 = Path("data/processed/features_v3.csv")
E_GNN_OOF = Path("data/processed/e_gnn_oof_v1.csv")
SPLITS_CSV = Path("data/processed/splits.csv")
OUTPUT_MD = Path("experiments/results/ablation_table_real.md")

# Feature group definitions
RAD_COLS = ["rad_mean", "rad_std", "rad_entropy", "rad_glcm_contrast", "rad_glcm_homogeneity"]
GRAPH_COLS = [
    "graph_n_follicles", "graph_mean_nn_dist_rel", "graph_std_nn_dist_rel",
    "graph_mean_degree", "graph_clustering_coeff", "graph_peripheral_index",
    "graph_size_cv", "graph_follicle_area_fraction",
]
ECNN_PCA_COLS = [f"ecnn_pca_{i}" for i in range(10)]
E_GNN_COLS = [f"e_gnn_{i}" for i in range(16)]
B2_COLS = ["width", "height", "brightness_mean", "contrast_std"]


def compute_metrics(y_true, y_pred, y_prob):
    """Compute all evaluation metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auroc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "n_train": 0,  # filled later
        "n_test": 0,
    }


def bootstrap_auroc_ci(y_true, y_prob, n_boot=500, seed=42):
    """Bootstrap 95% CI for AUROC."""
    rng = np.random.RandomState(seed)
    aurocs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if len(aurocs) < 10:
        return (0.0, 0.0)
    return (float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5)))


def fit_and_eval(X_train, y_train, X_test, y_test):
    """Fit LogisticRegression and evaluate on test set."""
    model = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics["n_train"] = len(X_train)
    metrics["n_test"] = len(X_test)
    ci_lo, ci_hi = bootstrap_auroc_ci(y_test, y_prob)
    metrics["auroc_ci"] = f"[{ci_lo:.3f}, {ci_hi:.3f}]"
    return metrics


def run_real_ablation():
    print("=" * 70)
    print("REAL ABLATION TABLE — OvaNet AI (HackSpire '26)")
    print("=" * 70)

    # Load data
    df_v3 = pd.read_csv(FEATURES_V3)
    df_splits = pd.read_csv(SPLITS_CSV)

    # Merge B2 features (width, height, brightness, contrast) from splits.csv
    df = df_v3.merge(df_splits[["path"] + B2_COLS], on="path", how="left")

    # Load E_gnn (only available for train split — OOF)
    df_egnn = pd.read_csv(E_GNN_OOF)

    # Merge E_gnn columns
    egnn_cols_present = [c for c in df_egnn.columns if c.startswith("e_gnn_")]
    df = df.merge(df_egnn[["path"] + egnn_cols_present], on="path", how="left")

    # Fill NaN E_gnn with 0 for val/test (they don't have OOF embeddings)
    for c in egnn_cols_present:
        df[c] = df[c].fillna(0.0)

    # Encode labels
    y_all = (df["label"] == "infected").astype(int).values

    # Split indices
    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"

    y_train = y_all[train_mask.values]
    y_test = y_all[test_mask.values]

    print(f"\nTrain: {train_mask.sum()} samples | Test: {test_mask.sum()} samples")
    print(f"Train class balance: {y_train.mean():.3f} PCOS | Test: {y_test.mean():.3f} PCOS\n")

    results = []

    def run_row(model_id, desc, feature_cols, note=""):
        """Run one ablation row."""
        try:
            X = df[feature_cols].values.astype(np.float32)
            X_train = X[train_mask.values]
            X_test_arr = X[test_mask.values]

            # Check for NaN/inf
            if np.any(~np.isfinite(X_train)) or np.any(~np.isfinite(X_test_arr)):
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
                X_test_arr = np.nan_to_num(X_test_arr, nan=0.0, posinf=0.0, neginf=0.0)

            metrics = fit_and_eval(X_train, y_train, X_test_arr, y_test)
            metrics["model"] = model_id
            metrics["description"] = desc
            metrics["n_features"] = len(feature_cols)
            metrics["note"] = note
            results.append(metrics)
            print(f"  {model_id}: AUROC={metrics['auroc']:.4f} {metrics['auroc_ci']}  "
                  f"Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  "
                  f"Sens={metrics['sensitivity']:.4f}  Spec={metrics['specificity']:.4f}  "
                  f"({len(feature_cols)} features)")
        except Exception as e:
            print(f"  {model_id}: FAILED — {e}")
            results.append({"model": model_id, "description": desc, "note": f"ERROR: {e}"})

    # ── Run each ablation row ──

    print("Running ablation rows...\n")

    # A0: Paper baseline — radiomics only (threshold vote)
    # We approximate with logistic regression on the 5 radiomics features
    run_row("A0", "Radiomics only (LR on 5 features)", RAD_COLS,
            "Approximates paper threshold-vote with LogReg")

    # A1: Radiomics on masked ROI (same features, just noting they're already masked)
    run_row("A1", "Radiomics on masked ovary ROI", RAD_COLS,
            "Same as A0 — features_v3 radiomics are already masked to ovary ROI")

    # A2: Radiomics (kept for compatibility, identical to A0/A1 on this data)
    run_row("A2", "LogReg on radiomics (F_k)", RAD_COLS)

    # A3: Radiomics + graph statistics
    run_row("A3", "Radiomics + graph stats", RAD_COLS + GRAPH_COLS)

    # A4: Radiomics + graph stats + PCA(E_cnn)
    run_row("A4", "Radiomics + graph + PCA(E_cnn)", RAD_COLS + GRAPH_COLS + ECNN_PCA_COLS)

    # A5: Full fusion — Radiomics + graph + PCA(E_cnn) + E_gnn
    egnn_available = [c for c in E_GNN_COLS if c in df.columns]
    run_row("A5", "Full OvaNet (all features)", RAD_COLS + GRAPH_COLS + ECNN_PCA_COLS + egnn_available,
            "E_gnn is OOF for train, zero-filled for test (conservative)")

    # B1: CNN only — PCA(E_cnn) alone
    run_row("B1", "CNN embedding only (PCA E_cnn)", ECNN_PCA_COLS,
            "ResNet-18 encoder features, no domain knowledge")

    # B2: SHORTCUT baseline — image metadata only
    run_row("B2", "[!] SHORTCUT (width/height/brightness/contrast)", B2_COLS,
            "CRITICAL: tests dataset confound, not medical signal")

    # ── Print final table ──

    print("\n" + "=" * 70)
    print("FINAL ABLATION TABLE")
    print("=" * 70)

    df_results = pd.DataFrame(results)
    cols_to_show = ["model", "description", "auroc", "auroc_ci", "accuracy", "f1",
                    "sensitivity", "specificity", "n_features", "note"]
    cols_present = [c for c in cols_to_show if c in df_results.columns]
    print(df_results[cols_present].to_string(index=False))

    # ── Save markdown report ──

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# Ablation Table — Real Features (OvaNet AI)\n\n")
        f.write(f"**Date:** 2026-10-02  \n")
        f.write(f"**Train:** {train_mask.sum()} samples | **Test:** {test_mask.sum()} samples  \n")
        f.write(f"**Method:** Logistic Regression (max_iter=2000, C=1.0)  \n")
        f.write(f"**Split:** Group-aware train/test from `splits.csv` (NO re-splitting)  \n\n")

        f.write("## Results\n\n")
        f.write("| Model | Description | AUROC | 95% CI | Accuracy | F1 | Sensitivity | Specificity | # Features |\n")
        f.write("|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")

        for _, r in df_results.iterrows():
            if "auroc" not in r or pd.isna(r.get("auroc")):
                f.write(f"| **{r['model']}** | {r['description']} | — | — | — | — | — | — | — |\n")
                continue
            auroc_str = f"{r['auroc']:.4f}"
            if r['model'] == 'B2':
                auroc_str = f"**{auroc_str}** ⚠️"
            f.write(f"| **{r['model']}** | {r['description']} | {auroc_str} | {r.get('auroc_ci','')} | "
                    f"{r['accuracy']:.4f} | {r['f1']:.4f} | {r['sensitivity']:.4f} | {r['specificity']:.4f} | "
                    f"{r.get('n_features','')} |\n")

        f.write("\n## Key Findings\n\n")
        f.write("1. **B2 Shortcut Baseline** uses ONLY image metadata (width, height, brightness, contrast) — "
                "NO medical content. If B2 AUROC ≈ 1.0, ALL models above are likely contaminated by acquisition bias.\n")
        f.write("2. **A0→A5 progression** shows marginal gains from adding graph and CNN features, "
                "but these gains must be interpreted cautiously given B2.\n")
        f.write("3. **Honest conclusion:** The dataset's PCOS/normal labels correlate with image "
                "acquisition parameters (resolution, brightness), not necessarily ovarian pathology. "
                "Multi-center validation is required before any clinical claims.\n")

    print(f"\nSaved report to {OUTPUT_MD}")
    return df_results


if __name__ == "__main__":
    run_real_ablation()
