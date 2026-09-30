"""
ovanet/evaluation/gate_g3.py

Gate G3 decision: does the GCN embedding (E_gnn) improve on graph-statistics-only?
Compares A3 (radiomics + graph stats) vs A3+E_gnn on VAL AUROC.
Features are standardized (StandardScaler, fit on TRAIN only, rule L2) before
logistic regression -- unscaled features with very different magnitudes can break
the solver and produce nonsensical (sub-chance) results.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

from ovanet.evaluation.metrics import compute_metrics

RADIOMICS_COLS = ["rad_mean", "rad_std", "rad_entropy", "rad_glcm_contrast", "rad_glcm_homogeneity"]
GRAPH_STAT_COLS = [
    "graph_n_follicles", "graph_mean_nn_dist_rel", "graph_std_nn_dist_rel",
    "graph_mean_degree", "graph_clustering_coeff", "graph_peripheral_index",
    "graph_size_cv", "graph_follicle_area_fraction",
]


def main():
    features = pd.read_csv("data/processed/features_v2.csv")
    oof = pd.read_csv("data/processed/e_gnn_oof_v1.csv")

    merged = features.merge(oof, on=["path", "label", "group_id"], suffixes=("", "_oof"))
    print(f"Merged rows: {len(merged)} (expect close to {len(features[features['split']=='train'])})")

    y = (merged["label"] == "infected").astype(int).values
    groups = merged["group_id"].values
    print(f"Label balance: {np.bincount(y)}")

    X_a3 = merged[RADIOMICS_COLS + GRAPH_STAT_COLS].values
    e_gnn_cols = [c for c in merged.columns if c.startswith("e_gnn_")]
    X_a5 = merged[RADIOMICS_COLS + GRAPH_STAT_COLS + e_gnn_cols].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X_a3, y, groups))

    def fit_eval(X):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X[train_idx])  # fit on TRAIN only, rule L2
        X_val_scaled = scaler.transform(X[val_idx])

        model = LogisticRegression(max_iter=2000)
        model.fit(X_train_scaled, y[train_idx])
        prob = model.predict_proba(X_val_scaled)[:, 1]
        pred = (prob > 0.5).astype(int)
        return compute_metrics(y[val_idx], pred, prob)

    m_a3 = fit_eval(X_a3)
    m_a5 = fit_eval(X_a5)

    print("\nA3 (radiomics + graph stats only):")
    print(f"  AUROC = {m_a3['auroc']:.4f}, accuracy = {m_a3['accuracy']:.4f}")
    print("\nA3 + E_gnn (graph stats + GCN embedding):")
    print(f"  AUROC = {m_a5['auroc']:.4f}, accuracy = {m_a5['accuracy']:.4f}")

    delta = m_a5["auroc"] - m_a3["auroc"]
    print(f"\nDelta AUROC (A5 - A3) = {delta:+.4f}")

    if delta > 0.01:
        print("\n>>> GATE G3 DECISION: E_gnn HELPS. Keep it in final fusion.")
    else:
        print("\n>>> GATE G3 DECISION: E_gnn does NOT meaningfully help. "
              "Drop E_gnn from final fusion, keep graph stats + overlay only.")


if __name__ == "__main__":
    main()