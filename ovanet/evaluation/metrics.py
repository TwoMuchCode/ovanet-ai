"""
ovanet/evaluation/metrics.py

Classification metrics + group-aware bootstrap confidence intervals.
Built now against synthetic labels so this harness is ready the moment
real predictions exist -- Day 8's final evaluation uses this unchanged.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, brier_score_loss, confusion_matrix,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """y_true, y_pred: 0/1 arrays. y_prob: predicted probability of class 1."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auroc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def bootstrap_ci_by_group(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, groups: np.ndarray,
    n_boot: int = 1000, ci: float = 0.95, seed: int = 42,
) -> dict:
    """
    Resamples by GROUP (near-duplicate cluster id), never by row -- this is
    what makes the split leak-safe. Returns {metric_name: (lo, hi)}.
    """
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    boot_results = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled_groups])
        if len(set(y_true[idx])) < 2:
            continue  # skip degenerate single-class resamples
        boot_results.append(compute_metrics(y_true[idx], y_pred[idx], y_prob[idx]))

    if not boot_results:
        return {}

    alpha = (1 - ci) / 2
    out = {}
    for key in boot_results[0]:
        if key == "confusion_matrix":
            continue
        vals = np.array([r[key] for r in boot_results if not np.isnan(r[key])])
        out[key] = (float(np.quantile(vals, alpha)), float(np.quantile(vals, 1 - alpha))) if len(vals) else (float("nan"), float("nan"))
    return out


if __name__ == "__main__":
    # Smoke test on SYNTHETIC labels -- confirms the harness works before any
    # real predictions exist. These numbers are never reported as a result.
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_prob = np.clip(y_true * 0.6 + rng.normal(0, 0.25, size=n) + 0.2, 0, 1)
    y_pred = (y_prob > 0.5).astype(int)
    groups = rng.integers(0, 40, size=n)

    print("Point metrics (SYNTHETIC smoke test):")
    print(compute_metrics(y_true, y_pred, y_prob))
    print("\nBootstrap 95% CIs by group (SYNTHETIC smoke test):")
    print(bootstrap_ci_by_group(y_true, y_pred, y_prob, groups))