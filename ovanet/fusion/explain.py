"""
ovanet/fusion/explain.py

SHAP (KernelExplainer, small k-means background) grouped by feature family.
This is the chart clinicians/judges actually read -- shows WHICH family
(radiomics/follicle/graph/cnn/gnn) drove a given risk score.
"""

import numpy as np
import joblib
import shap

FAMILY_MAP = {
    "rad_": "radiomics",
    "graph_n_follicles": "follicle",
    "graph_mean_nn_dist_rel": "graph",
    "graph_std_nn_dist_rel": "graph",
    "graph_mean_degree": "graph",
    "graph_clustering_coeff": "graph",
    "graph_peripheral_index": "graph",
    "graph_size_cv": "follicle",
    "graph_follicle_area_fraction": "follicle",
    "e_gnn_": "gnn",
    "e_cnn_": "cnn",
}


def family_of(feature_name: str) -> str:
    for prefix, family in FAMILY_MAP.items():
        if feature_name.startswith(prefix):
            return family
    return "unknown"


def explain_prediction(x_row: np.ndarray, X_background: np.ndarray, feature_cols: list) -> dict:
    """
    x_row: [1, n_features] scaled feature vector for one sample.
    X_background: small representative sample for KernelExplainer background (e.g. k-means centers).
    Returns grouped SHAP contribution per family.
    """
    model = joblib.load("weights/fusion_v1.joblib")

    background = shap.kmeans(X_background, 10)
    explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], background)
    shap_values = explainer.shap_values(x_row, nsamples=100)

    family_totals = {}
    for i, name in enumerate(feature_cols):
        fam = family_of(name)
        family_totals[fam] = family_totals.get(fam, 0.0) + float(shap_values[0][i])

    return family_totals


if __name__ == "__main__":
    from ovanet.fusion.features import load_fusion_dataset
    from sklearn.preprocessing import StandardScaler

    X, y, groups, feature_cols = load_fusion_dataset()
    scaler = joblib.load("weights/fusion_scaler_v1.joblib")
    X_scaled = scaler.transform(X)

    print("Running SHAP smoke test on one sample (this can take ~30-60s)...")
    result = explain_prediction(X_scaled[:1], X_scaled[:50], feature_cols)
    print("Family-level SHAP contribution for sample 0:")
    print(result)