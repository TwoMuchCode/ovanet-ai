# Baseline Validation Results (Day 3)

Dataset: 1921 images, group-aware split (seed 42).
Features computed on full RAW image (no mask, no ovary outline yet).

## Wilcoxon rank-sum tests (train split)

| Feature | Infected mean | Not-infected mean | Direction | Paper says | Match? | p-value |
|---------|--------------|-------------------|-----------|------------|--------|---------|
| mean | 0.2785 | 0.1898 | higher | higher | YES | 1.79e-165 *** |
| std | 0.2287 | 0.1643 | higher | higher | YES | 1.14e-187 *** |
| entropy | 4.2056 | 3.5977 | higher | higher | YES | 1.53e-184 *** |
| glcm_contrast | 3.5687 | 3.4935 | higher | higher | YES | 8.57e-02 ns |
| glcm_homogeneity | 0.6591 | 0.6449 | higher | lower | **NO** | 3.77e-13 *** |

## Midpoint-threshold majority vote (A0)

Thresholds computed on TRAIN only. Evaluated on TEST.

| Feature | Threshold | PCOS side |
|---------|-----------|-----------|
| mean | 0.2341 | higher |
| std | 0.1965 | higher |
| entropy | 3.9016 | higher |
| glcm_contrast | 3.5311 | higher |
| glcm_homogeneity | 0.6520 | lower |

### TEST results (n=287)

- Accuracy: 0.613
- Sensitivity (recall): 0.927
- Specificity: 0.265
- F1: 0.716
- Confusion matrix: TP=140, FP=100, FN=11, TN=36

**Note:** This is the paper's original method applied to our new group-aware split. The paper's 100% was on its own small sample — do not compare directly.

## Shortcut check (B2)

Logistic regression on: ['width', 'height', 'brightness_mean', 'contrast_std']
Trained on TRAIN, evaluated on TEST.

- Accuracy: 1.000
- F1: 1.000
- AUROC: 1.000

**WARNING: image metadata alone predicts the label with 100% accuracy. This dataset has a severe shortcut. Baseline B2 must be reported next to every headline metric.**

## What this means

- The paper baseline is model A0. It uses the paper's exact method.
- If the Wilcoxon directions match the paper, the finding replicates.
- If the shortcut baseline B2 is high, results must be interpreted with caution.
- These numbers are HONEST — they go in the pitch as-is.