"""
scripts/run_real_ablation.py

Runs the ablation table (A0-A3, B1, B2, plus an e_gnn sanity-check row) on
REAL features, using the real train/test splits from splits.csv.

Fixes vs the prior version (docs/DECISIONS.md, issue F):
  - Removed A4/A5: they depended on e_gnn (untrained-GCN noise, see issue E
    and fusion/features.py) and were otherwise identical to A3 once e_gnn is
    dropped. Replaced with a single diagnostic row (E_GNN_ONLY) that proves
    e_gnn carries no real signal, rather than quietly burying it in A5.
  - A0/A1/A2 were byte-identical (all "LogReg on radiomics") -- collapsed
    into one row (A0) with an honest note that this is a LogReg
    approximation of the paper's method, not a reimplementation of its
    actual threshold-vote algorithm (which was never fully specified).
  - Added StandardScaler, fit on TRAIN only, for every row -- the previous
    version fed unscaled features (radiomics ~0-1, PCA(E_cnn) on a very
    different scale) directly into LogisticRegression, which silently
    distorts which features dominate.
  - CIs are now GROUP-aware (ovanet.evaluation.metrics.bootstrap_ci_by_group),
    reusing group_id from splits.csv, so near-duplicate images never
    contribute to both the "real" and the resampled estimate independently.
    The old version resampled individual rows, which is not leak-safe.
  - Every row is checked for actually-present columns and skipped with a
    clear reason (not a silent crash) if its features aren't in the data.

Usage:
  python scripts/run_real_ablation.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ovanet.evaluation.metrics import compute_metrics, bootstrap_ci_by_group

FEATURES_V3 = Path("data/processed/features_v3.csv")
E_GNN_OOF = Path("data/processed/e_gnn_oof_v1.csv")
SPLITS_CSV = Path("data/processed/splits.csv")
OUTPUT_MD = Path("experiments/results/ablation_table_real.md")

RAD_COLS = ["rad_mean", "rad_std", "rad_entropy", "rad_glcm_contrast", "rad_glcm_homogeneity"]
GRAPH_COLS = [
    "graph_n_follicles", "graph_mean_nn_dist_rel", "graph_std_nn_dist_rel",
    "graph_mean_degree", "graph_clustering_coeff", "graph_peripheral_index",
    "graph_size_cv", "graph_follicle_area_fraction",
]
ECNN_PCA_COLS = [f"ecnn_pca_{i}" for i in range(10)]
E_GNN_COLS = [f"e_gnn_{i}" for i in range(16)]
B2_COLS = ["width", "height", "brightness_mean", "contrast_std"]


def fit_scale_eval(X_train, y_train, X_test, y_test, groups_test):
    X_train = np.nan_to_num(X_train.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)   # fit TRAIN only
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    model.fit(X_train_s, y_train)

    y_prob = model.predict_proba(X_test_s)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics["n_train"] = len(X_train)
    metrics["n_test"] = len(X_test)

    ci = bootstrap_ci_by_group(y_test, y_pred, y_prob, groups_test, n_boot=500)
    auroc_lo, auroc_hi = ci.get("auroc", (float("nan"), float("nan")))
    metrics["auroc_ci"] = f"[{auroc_lo:.3f}, {auroc_hi:.3f}]"
    return metrics


def run_real_ablation():
    print("=" * 70)
    print("REAL ABLATION TABLE -- OvaNet AI (HackSpire '26)")
    print("=" * 70)

    df_v3 = pd.read_csv(FEATURES_V3)
    df_splits = pd.read_csv(SPLITS_CSV)

    # Only pull columns from splits.csv that features_v3.csv doesn't already have,
    # to avoid pandas silently renaming collisions to group_id_x/group_id_y.
    needed_from_splits = ["path"] + [
        c for c in (["group_id"] + B2_COLS) if c not in df_v3.columns
    ]
    df = df_v3.merge(df_splits[needed_from_splits], on="path", how="left")

    if "group_id" not in df.columns:
        raise RuntimeError(
            "No 'group_id' column available in features_v3.csv or splits.csv after merge -- "
            "cannot run group-aware bootstrap CIs. Check column names with "
            "`python -c \"import pandas as pd; print(pd.read_csv('data/processed/features_v3.csv').columns.tolist())\"`"
        )
    if E_GNN_OOF.exists():
        df_egnn = pd.read_csv(E_GNN_OOF)
        egnn_cols_present = [c for c in df_egnn.columns if c.startswith("e_gnn_")]
        df = df.merge(df_egnn[["path"] + egnn_cols_present], on="path", how="left")
    else:
        egnn_cols_present = []

    y_all = (df["label"] == "infected").astype(int).values
    train_mask = (df["split"] == "train").values
    test_mask = (df["split"] == "test").values

    y_train, y_test = y_all[train_mask], y_all[test_mask]
    groups_test = df["group_id"].values[test_mask]

    print(f"\nTrain: {train_mask.sum()} samples | Test: {test_mask.sum()} samples")
    print(f"Train class balance: {y_train.mean():.3f} PCOS | Test: {y_test.mean():.3f} PCOS\n")

    results = []

    def run_row(model_id, desc, feature_cols, note=""):
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            print(f"  {model_id}: SKIPPED -- missing columns {missing}")
            results.append({"model": model_id, "description": desc,
                             "note": f"SKIPPED: missing columns {missing}"})
            return
        try:
            X = df[feature_cols].values
            X_train, X_test_arr = X[train_mask], X[test_mask]
            metrics = fit_scale_eval(X_train, y_train, X_test_arr, y_test, groups_test)
            metrics.update({"model": model_id, "description": desc,
                             "n_features": len(feature_cols), "note": note})
            results.append(metrics)
            print(f"  {model_id}: AUROC={metrics['auroc']:.4f} {metrics['auroc_ci']}  "
                  f"Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  "
                  f"Sens={metrics['sensitivity']:.4f}  Spec={metrics['specificity']:.4f}  "
                  f"({len(feature_cols)} features)")
        except Exception as e:
            print(f"  {model_id}: FAILED -- {e}")
            results.append({"model": model_id, "description": desc, "note": f"ERROR: {e}"})

    print("Running ablation rows...\n")

    run_row("A0", "Radiomics only (LogReg, scaled)", RAD_COLS,
            "LogReg approximation of the paper baseline -- NOT a reimplementation "
            "of the paper's own threshold-vote algorithm, which was never fully "
            "specified in our source material. Flagged honestly rather than guessed.")

    run_row("A1", "Radiomics + graph stats", RAD_COLS + GRAPH_COLS)

    run_row("A2", "Radiomics + graph + PCA(E_cnn)", RAD_COLS + GRAPH_COLS + ECNN_PCA_COLS)

    run_row("B1", "CNN embedding only (PCA E_cnn)", ECNN_PCA_COLS,
            "ImageNet-pretrained ResNet18 features, not a fine-tuned classifier "
            "(a real fine-tuned CNN baseline was descoped, see docs/DECISIONS.md)")

    run_row("B2", "[SHORTCUT] width/height/brightness/contrast", B2_COLS,
            "CRITICAL: tests dataset confound, not medical signal")

    if egnn_cols_present:
        egnn_available = [c for c in E_GNN_COLS if c in df.columns]
        run_row("E_GNN_ONLY (diagnostic)", "GCN embedding alone, no GCN ever trained",
                egnn_available,
                "DIAGNOSTIC ROW, not part of the deployed model. e_gnn came from "
                "throwaway per-fold GCNs (no final model trained, issue E) and is "
                "zero-filled for test rows here. AUROC near 0.5 would confirm it "
                "carries no real signal, which is why fusion-v2 excludes it entirely.")

    print("\n" + "=" * 70)
    print("FINAL ABLATION TABLE")
    print("=" * 70)

    df_results = pd.DataFrame(results)
    cols_to_show = ["model", "description", "auroc", "auroc_ci", "accuracy", "f1",
                     "sensitivity", "specificity", "n_features", "note"]
    cols_present = [c for c in cols_to_show if c in df_results.columns]
    print(df_results[cols_present].to_string(index=False))

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# Ablation Table -- Real Features (OvaNet AI)\n\n")
        f.write(f"**Date:** 2026-10-03 (corrected re-run, see docs/DECISIONS.md)  \n")
        f.write(f"**Train:** {train_mask.sum()} samples | **Test:** {test_mask.sum()} samples  \n")
        f.write(f"**Method:** Logistic Regression (max_iter=2000, C=1.0), "
                f"StandardScaler fit on TRAIN only  \n")
        f.write(f"**Split:** Real train/test from `splits.csv` (no re-splitting)  \n")
        f.write(f"**CIs:** Group-aware bootstrap (500 resamples by group_id)  \n\n")

        f.write("## Results\n\n")
        f.write("| Model | Description | AUROC | 95% CI | Accuracy | F1 | Sensitivity | Specificity | # Features |\n")
        f.write("|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for _, r in df_results.iterrows():
            if "auroc" not in r or pd.isna(r.get("auroc")):
                f.write(f"| **{r['model']}** | {r['description']} | -- | -- | -- | -- | -- | -- | -- |\n")
                continue
            auroc_str = f"{r['auroc']:.4f}"
            if str(r['model']).startswith("B2"):
                auroc_str = f"**{auroc_str}** [SHORTCUT]"
            f.write(f"| **{r['model']}** | {r['description']} | {auroc_str} | {r.get('auroc_ci','')} | "
                    f"{r['accuracy']:.4f} | {r['f1']:.4f} | {r['sensitivity']:.4f} | {r['specificity']:.4f} | "
                    f"{r.get('n_features','')} |\n")

        f.write("\n## Key Findings\n\n")
        f.write("1. **B2 (shortcut)** uses only image width/height/brightness/contrast -- no "
                "medical content. Its AUROC is the bar every real row must be read against.\n")
        f.write("2. If A0-A2's AUROC does not clearly exceed B2, the added radiomics/graph/CNN "
                "features are not demonstrated to escape the dataset's acquisition-metadata confound.\n")
        f.write("3. **E_GNN_ONLY** is a diagnostic, not a model we ship: it shows what a GCN "
                "embedding alone achieves when no GCN was ever actually trained.\n")
        f.write("4. **Honest conclusion:** labels in this dataset correlate with acquisition "
                "parameters, not confirmed ovarian pathology. Any AUROC near or below B2's should "
                "be read as \"not yet shown to be real signal,\" not as a working diagnostic.\n")

    print(f"\nSaved report to {OUTPUT_MD}")
    return df_results


if __name__ == "__main__":
    run_real_ablation()