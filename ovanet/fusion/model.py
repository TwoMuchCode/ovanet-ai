"""
ovanet/fusion/model.py

Deployed model = logistic regression on standardized features + Platt calibration
(fit on VAL only). Built by scripts/final_run.py. This is NOT an MLP.
Attribution = exact linear contribution (coef * standardized value) to the log-odds.
"""
from pathlib import Path

import joblib
import numpy as np

from ovanet.contracts import BaselineVote, FusionResult, TopFeature

MODEL_PATH = Path(__file__).resolve().parents[2] / "weights" / "fusion_v2.joblib"
LOW_THRESHOLD = 33.0
HIGH_THRESHOLD = 66.0
FOLLICLE_STATS = {"graph_n_follicles", "graph_size_cv", "graph_follicle_area_fraction"}

_bundle = None


def band_from_score(score: float) -> str:
    if score < LOW_THRESHOLD:
        return "LOW"
    if score < HIGH_THRESHOLD:
        return "MODERATE"
    return "HIGH"


def family_of(name: str) -> str:
    if name.startswith("rad_"):
        return "radiomics"
    if name.startswith("e_gnn_"):
        return "gnn"
    if name.startswith("ecnn_"):
        return "cnn"
    if name in FOLLICLE_STATS:
        return "follicle"
    return "graph"


def load_bundle() -> dict:
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"{MODEL_PATH} missing. Run: python scripts/final_run.py")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict(feature_dict: dict) -> FusionResult:
    b = load_bundle()
    cols = b["cols"]
    missing = [c for c in cols if c not in feature_dict]
    if missing:
        raise KeyError(f"Missing features: {missing}")

    x = np.array([[float(feature_dict[c]) for c in cols]], dtype=np.float64)
    z = b["scaler"].transform(np.nan_to_num(x))
    dec = b["lr"].decision_function(z)
    prob = float(b["cal"].predict_proba(dec.reshape(-1, 1))[0, 1])
    score = prob * 100.0

    contrib = b["lr"].coef_[0] * z[0]
    fam = {k: 0.0 for k in ("radiomics", "follicle", "graph", "cnn", "gnn")}
    for c, v in zip(cols, contrib):
        fam[family_of(c)] += abs(float(v))
    total = sum(fam.values()) or 1.0
    fam = {k: v / total for k, v in fam.items()}

    order = np.argsort(-np.abs(contrib))[:5]
    top = [TopFeature(name=cols[i], family=family_of(cols[i]), contribution=float(contrib[i])) for i in order]

    v = b["vote"]
    votes = sum(1 for c in v["cols"] if (float(feature_dict[c]) - v["mid"][c]) * v["dir"][c] > 0)
    label = "majority on dataset-positive side" if votes >= 3 else "majority on dataset-negative side"

    return FusionResult(
        risk_score=score,
        band=band_from_score(score),
        family_attribution=fam,
        top_features=top,
        baseline_vote=BaselineVote(votes_pcos=votes, of=len(v["cols"]), label=label),
        model_version=b["version"],
    )