"""
ovanet/fusion/attribution.py

Real, per-prediction explanation for the fusion model (no torch / shap needed).

The deployed fusion model is a calibrated *linear* classifier on standardised features,
so each feature's contribution to the log-odds is exactly  coef_i * x_scaled_i.
Contributions are summed per feature family. Positive = pushes the risk estimate up.
(Platt calibration is a monotonic map applied afterwards, so ranking/direction hold.)
"""

import numpy as np

from ovanet.contracts import FEATURE_FAMILIES, TopFeature
from ovanet.fusion.explain import family_of


def mean_coefficients(calibrated_model) -> np.ndarray | None:
    """Average linear coefficients over the calibration folds. None if not a linear model."""
    coefs = []
    for cc in getattr(calibrated_model, "calibrated_classifiers_", []):
        est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if est is not None and hasattr(est, "coef_"):
            coefs.append(np.asarray(est.coef_, dtype=float).ravel())
    if not coefs and hasattr(calibrated_model, "coef_"):
        coefs.append(np.asarray(calibrated_model.coef_, dtype=float).ravel())
    return np.mean(coefs, axis=0) if coefs else None


def explain_linear(calibrated_model, x_scaled: np.ndarray, feature_cols: list, top_k: int = 5):
    """
    x_scaled: [1, n_features]. Returns (family_attribution, top_features).
    Families with no input features (e.g. cnn) are reported as exactly 0.0.
    If the model is not linear, returns zeros and no top features (never invented values).
    """
    family = {f: 0.0 for f in FEATURE_FAMILIES}
    coef = mean_coefficients(calibrated_model)
    if coef is None or len(coef) != len(feature_cols):
        return family, []

    contrib = coef * np.asarray(x_scaled, dtype=float).reshape(-1)
    for name, c in zip(feature_cols, contrib):
        fam = family_of(name)
        family[fam] = family.get(fam, 0.0) + float(c)

    order = np.argsort(-np.abs(contrib))
    top = [
        TopFeature(name=feature_cols[i], family=family_of(feature_cols[i]), contribution=float(contrib[i]))
        for i in order[:top_k]
        if abs(contrib[i]) > 1e-9
    ]
    return family, top
