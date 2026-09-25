"""
Day 3: Paper baseline validation + shortcut check.

1. Computes the 5 radiomics features (C3) for every image in splits.csv.
2. Saves data/processed/features_v1.csv.
3. Wilcoxon rank-sum per feature (train), compares direction to the paper.
4. Midpoint-threshold majority vote (thresholds from train, evaluate on test).
5. Shortcut check: logistic regression on [width, height, brightness, contrast].
6. Prints all results and saves experiments/results/baseline_results.md.

Usage:
    python scripts/baseline_validation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
import cv2

# Add project root so ovanet is importable (if not pip-installed)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ovanet.radiomics import extract
from ovanet.contracts import RADIOMICS_FEATURE_ORDER

RAW_ROOT = Path("data/raw/pcos_primary/data")
SPLITS = Path("data/processed/splits.csv")
FEATURES_OUT = Path("data/processed/features_v1.csv")
RESULTS_OUT = Path("experiments/results/baseline_results.md")

# Paper's reported direction: PCOS has higher mean, std, entropy, contrast
# and LOWER homogeneity.
PAPER_DIRECTION = {
    "mean": "higher",
    "std": "higher",
    "entropy": "higher",
    "glcm_contrast": "higher",
    "glcm_homogeneity": "lower",
}


def compute_features(splits_df: pd.DataFrame) -> pd.DataFrame:
    """Compute C3 radiomics for every image.

    Uses a fast path: grayscale -> resize -> normalise -> extract.
    Skips ENH (bilateral filter + CLAHE) since radiomics needs only RAW.
    """
    from ovanet.contracts import IMAGE_SIZE
    import time

    records = []
    n = len(splits_df)
    t0 = time.time()

    for idx, (i, row) in enumerate(splits_df.iterrows()):
        img_path = RAW_ROOT / row["path"]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  WARNING: could not read {img_path}")
            continue

        # Fast RAW: resize + normalise only (no bilateral/CLAHE)
        raw = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        raw = raw.astype(np.float32) / 255.0
        rad = extract(raw)

        records.append({
            "path": row["path"],
            "label": row["label"],
            "split": row["split"],
            "group_id": row["group_id"],
            "mean": rad.mean,
            "std": rad.std,
            "entropy": rad.entropy,
            "glcm_contrast": rad.glcm_contrast,
            "glcm_homogeneity": rad.glcm_homogeneity,
        })

        done = idx + 1
        if done % 200 == 0 or done == n:
            elapsed = time.time() - t0
            eta = (elapsed / done) * (n - done)
            print(f"  {done}/{n}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s left)")

    return pd.DataFrame(records)


def wilcoxon_analysis(features: pd.DataFrame) -> list[str]:
    """Wilcoxon rank-sum per feature on TRAIN split, compare to paper."""
    train = features[features["split"] == "train"]
    inf = train[train["label"] == "infected"]
    notinf = train[train["label"] == "notinfected"]

    lines = [
        "## Wilcoxon rank-sum tests (train split)\n",
        "| Feature | Infected mean | Not-infected mean | Direction | Paper says | Match? | p-value |",
        "|---------|--------------|-------------------|-----------|------------|--------|---------|",
    ]

    for feat in RADIOMICS_FEATURE_ORDER:
        a = inf[feat].values
        b = notinf[feat].values
        stat, pval = stats.mannwhitneyu(a, b, alternative="two-sided")

        inf_mean = float(a.mean())
        notinf_mean = float(b.mean())
        direction = "higher" if inf_mean > notinf_mean else "lower"
        paper = PAPER_DIRECTION[feat]
        match = "YES" if direction == paper else "**NO**"
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))

        lines.append(
            f"| {feat} | {inf_mean:.4f} | {notinf_mean:.4f} | {direction} "
            f"| {paper} | {match} | {pval:.2e} {sig} |"
        )

    return lines


def midpoint_voting(features: pd.DataFrame) -> tuple[list[str], dict]:
    """
    Paper baseline A0: midpoint-threshold majority vote.

    For each feature, threshold = (train_infected_mean + train_notinfected_mean) / 2.
    Vote PCOS if the feature's value is on the PCOS side (higher or lower per paper).
    Majority vote (>= 3 of 5) → predicted PCOS.
    Thresholds computed on TRAIN only.
    """
    train = features[features["split"] == "train"]
    test = features[features["split"] == "test"]

    inf_means = train[train["label"] == "infected"][RADIOMICS_FEATURE_ORDER].mean()
    notinf_means = train[train["label"] == "notinfected"][RADIOMICS_FEATURE_ORDER].mean()
    thresholds = (inf_means + notinf_means) / 2

    lines = [
        "\n## Midpoint-threshold majority vote (A0)\n",
        "Thresholds computed on TRAIN only. Evaluated on TEST.\n",
        "| Feature | Threshold | PCOS side |",
        "|---------|-----------|-----------|",
    ]
    for feat in RADIOMICS_FEATURE_ORDER:
        side = PAPER_DIRECTION[feat]
        lines.append(f"| {feat} | {thresholds[feat]:.4f} | {side} |")

    # Vote on TEST
    y_true = (test["label"] == "infected").astype(int).values
    votes = np.zeros(len(test), dtype=int)
    for feat in RADIOMICS_FEATURE_ORDER:
        vals = test[feat].values
        t = thresholds[feat]
        if PAPER_DIRECTION[feat] == "higher":
            votes += (vals > t).astype(int)
        else:
            votes += (vals < t).astype(int)

    y_pred = (votes >= 3).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    lines += [
        f"\n### TEST results (n={len(test)})\n",
        f"- Accuracy: {acc:.3f}",
        f"- Sensitivity (recall): {sens:.3f}",
        f"- Specificity: {spec:.3f}",
        f"- F1: {f1:.3f}",
        f"- Confusion matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}",
        "",
        "**Note:** This is the paper's original method applied to our new "
        "group-aware split. The paper's 100% was on its own small sample — "
        "do not compare directly.",
    ]

    metrics = {"accuracy": acc, "sensitivity": sens, "specificity": spec, "f1": f1}
    return lines, metrics


def shortcut_check(splits_df: pd.DataFrame, features: pd.DataFrame) -> list[str]:
    """
    Baseline B2: logistic regression on image metadata only.

    If this predicts the label well, the dataset has shortcuts.
    Uses original width, height, brightness_mean, contrast_std from manifest.
    """
    # Merge metadata from splits_df (which has width, height, brightness_mean, contrast_std)
    meta_cols = ["path", "width", "height", "brightness_mean", "contrast_std"]
    available = [c for c in meta_cols if c in splits_df.columns]
    meta = splits_df[available].copy()
    merged = features.merge(meta, on="path", how="left")

    feature_cols = [c for c in ["width", "height", "brightness_mean", "contrast_std"]
                    if c in merged.columns]

    if not feature_cols:
        return ["\n## Shortcut check (B2)\n", "Metadata columns not available — skipped."]

    train = merged[merged["split"] == "train"]
    test = merged[merged["split"] == "test"]

    X_train = train[feature_cols].fillna(0).values
    y_train = (train["label"] == "infected").astype(int).values
    X_test = test[feature_cols].fillna(0).values
    y_test = (test["label"] == "infected").astype(int).values

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    try:
        y_prob = clf.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, y_prob)
    except Exception:
        auroc = float("nan")

    lines = [
        "\n## Shortcut check (B2)\n",
        f"Logistic regression on: {feature_cols}",
        f"Trained on TRAIN, evaluated on TEST.\n",
        f"- Accuracy: {acc:.3f}",
        f"- F1: {f1:.3f}",
        f"- AUROC: {auroc:.3f}",
        "",
    ]

    if acc > 0.90:
        lines.append(
            "**WARNING: image metadata alone predicts the label with "
            f"{acc:.0%} accuracy. This dataset has a severe shortcut. "
            "Baseline B2 must be reported next to every headline metric.**"
        )
    else:
        lines.append("Metadata does not strongly predict the label — good.")

    return lines


def main():
    print("=" * 60)
    print("  BASELINE VALIDATION — Day 3")
    print("=" * 60)

    # Load splits
    splits_df = pd.read_csv(SPLITS)
    print(f"\nLoaded {len(splits_df)} images from {SPLITS}")

    # Compute features
    print("\nComputing radiomics features (C3) for all images …")
    features = compute_features(splits_df)

    # Save features_v1
    FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_OUT, index=False, encoding="utf-8")
    print(f"Saved → {FEATURES_OUT} ({len(features)} rows)")

    # Wilcoxon analysis
    print("\nWilcoxon rank-sum tests …")
    wilcoxon_lines = wilcoxon_analysis(features)
    for line in wilcoxon_lines:
        print(line)

    # Midpoint voting
    print("\nMidpoint-threshold voting …")
    voting_lines, metrics = midpoint_voting(features)
    for line in voting_lines:
        print(line)

    # Shortcut check
    print("\nShortcut check (B2) …")
    shortcut_lines = shortcut_check(splits_df, features)
    for line in shortcut_lines:
        print(line)

    # Save report
    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    all_lines = [
        "# Baseline Validation Results (Day 3)\n",
        f"Dataset: {len(features)} images, group-aware split (seed 42).",
        f"Features computed on full RAW image (no mask, no ovary outline yet).\n",
    ] + wilcoxon_lines + voting_lines + shortcut_lines + [
        "\n## What this means\n",
        "- The paper baseline is model A0. It uses the paper's exact method.",
        "- If the Wilcoxon directions match the paper, the finding replicates.",
        "- If the shortcut baseline B2 is high, results must be interpreted with caution.",
        "- These numbers are HONEST — they go in the pitch as-is.",
    ]
    RESULTS_OUT.write_text("\n".join(all_lines), encoding="utf-8")
    print(f"\nFull report → {RESULTS_OUT}")
    print("Done.")


if __name__ == "__main__":
    main()
