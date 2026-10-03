"""
ovanet/fusion/baseline_vote.py

The paper's baseline (model A0): midpoint-threshold majority vote over the 5 radiomics
features. Thresholds are the midpoint between the two class means on the TRAIN split of
data/processed/features_v2.csv (masked radiomics, same as the app). The direction per feature
is the paper's, fixed (higher mean/std/entropy/contrast, LOWER homogeneity in PCOS).
On this dataset homogeneity actually runs the other way, so that vote is the paper's rule,
not a data-driven one.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ovanet.contracts import BaselineVote

FEATURES_CSV = Path(__file__).resolve().parents[2] / "data" / "processed" / "features_v2.csv"
COLS = ["rad_mean", "rad_std", "rad_entropy", "rad_glcm_contrast", "rad_glcm_homogeneity"]
PAPER_HIGHER_IN_PCOS = {"rad_mean": True, "rad_std": True, "rad_entropy": True,
                        "rad_glcm_contrast": True, "rad_glcm_homogeneity": False}

_thresholds = None


def _load_thresholds():
    global _thresholds
    if _thresholds is None:
        df = pd.read_csv(FEATURES_CSV)
        tr = df[df["split"] == "train"]
        pos, neg = tr[tr["label"] == "infected"], tr[tr["label"] != "infected"]
        _thresholds = {c: float((pos[c].mean() + neg[c].mean()) / 2.0) for c in COLS}
    return _thresholds


def baseline_vote(feature_dict: dict) -> BaselineVote:
    try:
        th = _load_thresholds()
        votes = sum(
            (feature_dict[c] > th[c]) if PAPER_HIGHER_IN_PCOS[c] else (feature_dict[c] < th[c])
            for c in COLS
        )
    except Exception:
        return BaselineVote(votes_pcos=0, of=5, label="unavailable")
    label = "PCOS-pattern majority" if votes >= 3 else "non-PCOS-pattern majority"
    return BaselineVote(votes_pcos=int(votes), of=5, label=f"paper baseline: {label}")
