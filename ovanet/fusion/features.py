"""
ovanet/fusion/features.py

Assembles the fused feature vector for the DEPLOYED model: radiomics (5) +
follicle/graph stats (8) only -- 13 real features.

e_gnn_* columns are deliberately EXCLUDED here. They were produced by
throwaway per-fold GCNs during ovanet/evaluation/oof_gnn.py -- no final GCN
was ever trained (see docs/DECISIONS.md, issue E), so those 16 columns are
untrained noise, not real embeddings. Training on them and then filling
missing values with the training mean at inference (the prior approach)
meant the model carried small but real coefficients on pure noise, and
every live prediction silently added back a constant that encoded nothing.
Dropping them is both more honest and strictly simpler.
"""

import numpy as np
import pandas as pd

RADIOMICS_COLS = ["rad_mean", "rad_std", "rad_entropy", "rad_glcm_contrast", "rad_glcm_homogeneity"]
GRAPH_STAT_COLS = [
    "graph_n_follicles", "graph_mean_nn_dist_rel", "graph_std_nn_dist_rel",
    "graph_mean_degree", "graph_clustering_coeff", "graph_peripheral_index",
    "graph_size_cv", "graph_follicle_area_fraction",
]
FEATURE_COLS = RADIOMICS_COLS + GRAPH_STAT_COLS


def load_fusion_dataset(features_path="data/processed/features_v2.csv", split=None):
    """
    Returns (X, y, groups, feature_names).

    split: None (all rows), or one of "train"/"val"/"test" to filter to that
    split only (uses the real split column from splits.csv, merged into
    features_v2.csv -- NOT a random re-split).
    """
    df = pd.read_csv(features_path)
    if split is not None:
        if "split" not in df.columns:
            raise ValueError("features_v2.csv has no 'split' column -- cannot filter by split")
        df = df[df["split"] == split].reset_index(drop=True)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = (df["label"] == "infected").astype(int).values
    groups = df["group_id"].values

    return X, y, groups, FEATURE_COLS