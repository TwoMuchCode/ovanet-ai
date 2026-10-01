"""
ovanet/fusion/features.py

Assembles the real fused feature vector per contract C6:
radiomics (5) + follicle/graph stats (8) + E_gnn (16) for now.
PCA(E_cnn) is added once Madhumita's U-Net export lands (Day 6/7 per her side) --
left as a hook, not yet wired.
"""

import numpy as np
import pandas as pd

RADIOMICS_COLS = ["rad_mean", "rad_std", "rad_entropy", "rad_glcm_contrast", "rad_glcm_homogeneity"]
GRAPH_STAT_COLS = [
    "graph_n_follicles", "graph_mean_nn_dist_rel", "graph_std_nn_dist_rel",
    "graph_mean_degree", "graph_clustering_coeff", "graph_peripheral_index",
    "graph_size_cv", "graph_follicle_area_fraction",
]


def load_fusion_dataset(features_path="data/processed/features_v2.csv",
                         oof_path="data/processed/e_gnn_oof_v1.csv"):
    """Returns (X, y, groups, feature_names) for the TRAIN split, with E_gnn included."""
    features = pd.read_csv(features_path)
    oof = pd.read_csv(oof_path)
    merged = features.merge(oof, on=["path", "label", "group_id"], suffixes=("", "_oof"))

    e_gnn_cols = [c for c in merged.columns if c.startswith("e_gnn_")]
    feature_cols = RADIOMICS_COLS + GRAPH_STAT_COLS + e_gnn_cols

    X = merged[feature_cols].values.astype(np.float32)
    y = (merged["label"] == "infected").astype(int).values
    groups = merged["group_id"].values

    return X, y, groups, feature_cols