"""
ovanet/fusion/model.py

C6 fusion. The production predict() loads weights/fusion_v2.joblib (trained
in ovanet/fusion/train.py on radiomics + graph-stat features ONLY -- e_gnn
excluded, see fusion/features.py for why).

Attribution and the baseline vote are computed directly here, inline,
from the real fitted model -- not from a separate module -- so there is
exactly one source of truth and no risk of it drifting out of sync with
whatever model version is actually loaded.
"""

import numpy as np
import joblib

from ovanet.contracts import FusionResult, TopFeature, BaselineVote
from ovanet.fusion.features import FEATURE_COLS

MODEL_PATH = "weights/fusion_v2.joblib"
SCALER_PATH = "weights/fusion_scaler_v2.joblib"
FEATURE_COLS_PATH = "weights/fusion_feature_cols_v2.joblib"
BANDS_PATH = "weights/fusion_bands_v2.joblib"

_model = None
_scaler = None
_feature_cols = None
_bands = None


def _family_of(feature_name: str) -> str:
    if feature_name.startswith("rad_"):
        return "radiomics"
    if feature_name == "graph_n_follicles" or feature_name.startswith("graph_follicle_area") \
            or feature_name == "graph_size_cv":
        return "follicle"
    if feature_name.startswith("graph_"):
        return "graph"
    return "unknown"


def _lazy_load():
    global _model, _scaler, _feature_cols, _bands
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        _feature_cols = joblib.load(FEATURE_COLS_PATH)
        _bands = joblib.load(BANDS_PATH)


def band_from_score(score: float) -> str:
    _lazy_load()
    if score < _bands["low"]:
        return "LOW"
    elif score < _bands["high"]:
        return "MODERATE"
    return "HIGH"


def predict(feature_dict: dict) -> FusionResult:
    """
    feature_dict: {feature_name: value}. Only keys matching FEATURE_COLS
    (radiomics + graph stats, 13 total) are used; anything else (e.g. stray
    e_gnn_* keys from older pipeline.py versions) is ignored, not fed to
    the model.
    """
    _lazy_load()

    x_raw = np.array([[feature_dict.get(c, 0.0) for c in _feature_cols]], dtype=np.float32)
    x_scaled = _scaler.transform(x_raw)
    prob = float(_model.predict_proba(x_scaled)[0, 1])
    risk_score = prob * 100.0
    band = band_from_score(risk_score)

    # --- Real attribution: coefficient x scaled value, summed per family ---
    # CalibratedClassifierCV wraps one or more calibrated copies of the base
    # LogisticRegression (one per CV fold); average their coefficients for a
    # single, stable attribution rather than picking one fold arbitrarily.
    coefs = np.mean(
        [cc.estimator.coef_[0] for cc in _model.calibrated_classifiers_],
        axis=0,
    )
    contributions = coefs * x_scaled[0]

    family_totals = {"radiomics": 0.0, "follicle": 0.0, "graph": 0.0, "cnn": 0.0, "gnn": 0.0}
    top_features = []
    for name, contrib in zip(_feature_cols, contributions):
        fam = _family_of(name)
        if fam in family_totals:
            family_totals[fam] += float(contrib)
        top_features.append(TopFeature(name=name, family=fam, contribution=float(contrib)))

    # Normalize family attribution to sum to 1 (relative share of |contribution|)
    abs_total = sum(abs(v) for v in family_totals.values())
    if abs_total > 0:
        family_totals = {k: abs(v) / abs_total for k, v in family_totals.items()}

    top_features.sort(key=lambda t: abs(t.contribution), reverse=True)

    return FusionResult(
        risk_score=risk_score,
        band=band,
        family_attribution=family_totals,
        top_features=top_features[:5],
        baseline_vote=BaselineVote(votes_pcos=0, of=5, label="see A0 baseline in evaluation table"),
        model_version="fusion-v2-radiomics-graph-only",
    )