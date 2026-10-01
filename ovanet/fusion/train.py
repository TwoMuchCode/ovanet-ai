"""
ovanet/fusion/train.py

Trains the fusion classifier on real features, calibrates probabilities
(Platt scaling via CalibratedClassifierCV), defines LOW/MODERATE/HIGH bands,
and saves the fitted model + scaler for use by predict().

Calibration and scaling are fit on TRAIN only (rule L2).
Thresholds are documented here, not invented ad hoc.
"""

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

from ovanet.fusion.features import load_fusion_dataset
from ovanet.evaluation.metrics import compute_metrics

MODEL_PATH = "weights/fusion_v1.joblib"
SCALER_PATH = "weights/fusion_scaler_v1.joblib"

# Risk band thresholds -- documented here, used consistently everywhere.
# Chosen as simple terciles on predicted probability; revisit if val performance suggests otherwise.
LOW_THRESHOLD = 33.0
HIGH_THRESHOLD = 66.0


def band_from_score(score: float) -> str:
    if score < LOW_THRESHOLD:
        return "LOW"
    elif score < HIGH_THRESHOLD:
        return "MODERATE"
    return "HIGH"


def main():
    X, y, groups, feature_cols = load_fusion_dataset()
    print(f"Loaded {len(X)} samples, {len(feature_cols)} features")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])  # fit on TRAIN only
    X_val = scaler.transform(X[val_idx])

    base_model = LogisticRegression(max_iter=2000)
    calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
    calibrated.fit(X_train, y[train_idx])

    val_prob = calibrated.predict_proba(X_val)[:, 1]
    val_pred = (val_prob > 0.5).astype(int)
    metrics = compute_metrics(y[val_idx], val_pred, val_prob)

    print("\nCalibrated fusion model -- VAL performance:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    import os
    os.makedirs("weights", exist_ok=True)
    joblib.dump(calibrated, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_cols, "weights/fusion_feature_cols_v1.joblib")
    print(f"\nSaved model to {MODEL_PATH}, scaler to {SCALER_PATH}")


if __name__ == "__main__":
    main()