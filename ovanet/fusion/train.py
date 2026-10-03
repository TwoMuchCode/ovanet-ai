"""
ovanet/fusion/train.py

Trains the fusion classifier on radiomics + graph-stat features ONLY
(see fusion/features.py for why e_gnn is excluded).

Fixes vs the prior version (docs/DECISIONS.md, issue G):
  - Uses the REAL train/val/test splits from splits.csv (not a random 80/20
    re-split of train, which left the actual 289-row val set unused).
  - Calibration CV is GROUP-AWARE (GroupKFold), so near-duplicate images
    never appear on both sides of a calibration fold.
  - Scaler fit on TRAIN only (rule L2).
  - Risk bands are set from the TRAIN score distribution's own tercile
    boundaries, not arbitrary fixed numbers.
  - Reports VAL and TEST metrics separately, plus the B2 shortcut baseline
    on the same TEST rows, so every headline number sits next to the
    confound it needs to be read against.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

from ovanet.fusion.features import load_fusion_dataset, FEATURE_COLS
from ovanet.evaluation.metrics import compute_metrics

MODEL_PATH = "weights/fusion_v2.joblib"
SCALER_PATH = "weights/fusion_scaler_v2.joblib"
FEATURE_COLS_PATH = "weights/fusion_feature_cols_v2.joblib"
BANDS_PATH = "weights/fusion_bands_v2.joblib"


def band_from_score(score: float, low_cut: float, high_cut: float) -> str:
    if score < low_cut:
        return "LOW"
    elif score < high_cut:
        return "MODERATE"
    return "HIGH"


def b2_shortcut_baseline(test_df_path="data/processed/splits.csv"):
    """Re-confirms the image-metadata shortcut on the TEST split specifically,
    so it sits right next to the real model's TEST numbers below."""
    df = pd.read_csv(test_df_path)
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    Xtr = train[["width", "height", "brightness_mean", "contrast_std"]].values
    ytr = (train["label"] == "infected").astype(int).values
    Xte = test[["width", "height", "brightness_mean", "contrast_std"]].values
    yte = (test["label"] == "infected").astype(int).values

    scaler = StandardScaler().fit(Xtr)
    model = LogisticRegression(max_iter=2000).fit(scaler.transform(Xtr), ytr)
    prob = model.predict_proba(scaler.transform(Xte))[:, 1]
    pred = (prob > 0.5).astype(int)
    return compute_metrics(yte, pred, prob)


def main():
    X_train, y_train, groups_train, _ = load_fusion_dataset(split="train")
    X_val, y_val, _, _ = load_fusion_dataset(split="val")
    X_test, y_test, _, _ = load_fusion_dataset(split="test")

    print(f"train={len(X_train)}  val={len(X_val)}  test={len(X_test)}  features={len(FEATURE_COLS)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)   # fit TRAIN only
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Group-aware calibration CV: near-duplicate images never split across folds
    n_groups = len(np.unique(groups_train))
    n_splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=n_splits)
    cv_folds = list(gkf.split(X_train_s, y_train, groups_train))

    base_model = LogisticRegression(max_iter=2000)
    calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv=cv_folds)
    calibrated.fit(X_train_s, y_train)

    def evaluate(name, X_s, y):
        prob = calibrated.predict_proba(X_s)[:, 1]
        pred = (prob > 0.5).astype(int)
        m = compute_metrics(y, pred, prob)
        print(f"\n{name} (n={len(y)}):")
        for k, v in m.items():
            if k != "confusion_matrix":
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        return prob, m

    train_prob, _ = evaluate("TRAIN", X_train_s, y_train)
    val_prob, val_metrics = evaluate("VAL (real held-out, 289 rows)", X_val_s, y_val)
    test_prob, test_metrics = evaluate("TEST (real held-out)", X_test_s, y_test)

    print("\nB2 shortcut baseline on the SAME test split (for comparison):")
    b2_metrics = b2_shortcut_baseline()
    for k, v in b2_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Bands from TRAIN score distribution's own terciles (not arbitrary 33/66)
    train_scores = train_prob * 100
    low_cut, high_cut = np.percentile(train_scores, [33.3, 66.6])
    print(f"\nRisk bands set from TRAIN score terciles: LOW < {low_cut:.1f}, "
          f"MODERATE < {high_cut:.1f}, HIGH >= {high_cut:.1f}")

    os.makedirs("weights", exist_ok=True)
    joblib.dump(calibrated, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(FEATURE_COLS, FEATURE_COLS_PATH)
    joblib.dump({"low": float(low_cut), "high": float(high_cut)}, BANDS_PATH)
    print(f"\nSaved model/scaler/feature_cols/bands to weights/*_v2.joblib")
    print("\nHONEST SUMMARY: compare VAL vs TEST vs B2 above before trusting these numbers.")
    print("If TEST AUROC is far below VAL, that confirms the group-split/device-leakage")
    print("hypothesis in docs/DECISIONS.md (issue A). If TEST AUROC is close to B2's,")
    print("that confirms the model is still mostly riding the dataset's shortcut.")


if __name__ == "__main__":
    main()