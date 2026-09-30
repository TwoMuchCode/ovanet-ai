"""
ovanet/evaluation/ablation.py

Runs the A0-A5 / B1 / B2 comparison table (master doc Section 5).
Day 4: works end-to-end on SYNTHETIC placeholder data so the harness itself
is proven correct before real features exist. Rows get swapped to real data
as each piece lands (radiomics=ready, graph=pending, e_cnn=pending, gnn=pending).

Never report these Day 4 numbers as results -- this is a plumbing test only (rule S1).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ovanet.evaluation.metrics import compute_metrics, bootstrap_ci_by_group

MODEL_IDS = ["A0", "A1", "A2", "A3", "A4", "A5", "B1", "B2"]

MODEL_DESCRIPTIONS = {
    "A0": "Paper baseline: threshold vote on F_k, central crop",
    "A1": "Paper baseline on automatic masked ROI",
    "A2": "Logistic regression on F_k",
    "A3": "A2 + follicle features + graph statistics",
    "A4": "A3 + PCA(E_cnn)",
    "A5": "A4 + E_gnn = full OvaNet fusion MLP",
    "B1": "Plain fine-tuned CNN classifier (ResNet18)",
    "B2": "Shortcut baseline (size/brightness/contrast only)",
}


def make_synthetic_dataset(n=200, seed=42):
    """
    PLUMBING ONLY. Generates fake features + labels so the ablation runner
    can be built and tested before real feature files exist for every stage.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    groups = rng.integers(0, 40, size=n)

    features = {
        "radiomics": rng.normal(size=(n, 5)) + y[:, None] * 0.3,
        "follicle_graph": rng.normal(size=(n, 8)) + y[:, None] * 0.2,
        "e_cnn_pca": rng.normal(size=(n, 10)) + y[:, None] * 0.25,
        "e_gnn": rng.normal(size=(n, 16)) + y[:, None] * 0.15,
        "size_brightness_contrast": rng.normal(size=(n, 3)) + y[:, None] * 0.6,  # deliberately strong shortcut signal
    }
    return features, y, groups


def _fit_and_eval(X_train, y_train, X_test, y_test, groups_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    point = compute_metrics(y_test, y_pred, y_prob)
    cis = bootstrap_ci_by_group(y_test, y_pred, y_prob, groups_test)
    return point, cis


def run_ablation(features: dict, y: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=0.3, random_state=42, stratify=y
    )
    y_train, y_test = y[idx_train], y[idx_test]
    groups_test = groups[idx_test]

    rows = []

    def add_row(model_id, X):
        if X is None:
            rows.append({"model": model_id, "description": MODEL_DESCRIPTIONS[model_id], "status": "pending"})
            return
        X_train, X_test = X[idx_train], X[idx_test]
        point, cis = _fit_and_eval(X_train, y_train, X_test, y_test, groups_test)
        row = {"model": model_id, "description": MODEL_DESCRIPTIONS[model_id], "status": "done"}
        row.update(point)
        row["ci"] = cis
        rows.append(row)

    # A0/A1: paper's threshold-vote method isn't logistic regression -- placeholder until wired to real baseline vote logic
    add_row("A0", None)
    add_row("A1", None)

    add_row("A2", features.get("radiomics"))

    if "radiomics" in features and "follicle_graph" in features:
        add_row("A3", np.hstack([features["radiomics"], features["follicle_graph"]]))
    else:
        add_row("A3", None)

    if all(k in features for k in ["radiomics", "follicle_graph", "e_cnn_pca"]):
        add_row("A4", np.hstack([features["radiomics"], features["follicle_graph"], features["e_cnn_pca"]]))
    else:
        add_row("A4", None)

    if all(k in features for k in ["radiomics", "follicle_graph", "e_cnn_pca", "e_gnn"]):
        add_row("A5", np.hstack([features["radiomics"], features["follicle_graph"],
                                   features["e_cnn_pca"], features["e_gnn"]]))
    else:
        add_row("A5", None)

    add_row("B1", None)  # CNN classifier -- separate training pipeline, not logistic regression
    add_row("B2", features.get("size_brightness_contrast"))

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Ablation runner smoke test -- SYNTHETIC DATA, plumbing only, never a reported result.\n")
    features, y, groups = make_synthetic_dataset()
    table = run_ablation(features, y, groups)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 150)
    print(table[["model", "description", "status", "accuracy", "auroc", "f1"]].to_string(index=False))