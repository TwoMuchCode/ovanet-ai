# Ablation Table -- Real Features (OvaNet AI)

**Date:** 2026-10-03 (corrected re-run, see docs/DECISIONS.md)  
**Train:** 1345 samples | **Test:** 287 samples  
**Method:** Logistic Regression (max_iter=2000, C=1.0), StandardScaler fit on TRAIN only  
**Split:** Real train/test from `splits.csv` (no re-splitting)  
**CIs:** Group-aware bootstrap (500 resamples by group_id)  

## Results

| Model | Description | AUROC | 95% CI | Accuracy | F1 | Sensitivity | Specificity | # Features |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A0** | Radiomics only (LogReg, scaled) | 0.9091 | [0.852, 0.956] | 0.7770 | 0.8118 | 0.9139 | 0.6250 | 5 |
| **A1** | Radiomics + graph stats | 0.9722 | [0.946, 0.991] | 0.8746 | 0.8889 | 0.9536 | 0.7868 | 13 |
| **A2** | Radiomics + graph + PCA(E_cnn) | 1.0000 | [1.000, 1.000] | 0.9930 | 0.9934 | 1.0000 | 0.9853 | 23 |
| **B1** | CNN embedding only (PCA E_cnn) | 1.0000 | [1.000, 1.000] | 0.9965 | 0.9967 | 1.0000 | 0.9926 | 10 |
| **B2** | [SHORTCUT] width/height/brightness/contrast | **0.9902** [SHORTCUT] | [0.973, 0.999] | 0.7631 | 0.8162 | 1.0000 | 0.5000 | 4 |
| **E_GNN_ONLY (diagnostic)** | GCN embedding alone, no GCN ever trained | 0.4963 | [0.487, 0.500] | 0.4739 | 0.0000 | 0.0000 | 1.0000 | 16 |

## Key Findings

1. **B2 (shortcut)** uses only image width/height/brightness/contrast -- no medical content. Its AUROC is the bar every real row must be read against.
2. If A0-A2's AUROC does not clearly exceed B2, the added radiomics/graph/CNN features are not demonstrated to escape the dataset's acquisition-metadata confound.
3. **E_GNN_ONLY** is a diagnostic, not a model we ship: it shows what a GCN embedding alone achieves when no GCN was ever actually trained.
4. **Honest conclusion:** labels in this dataset correlate with acquisition parameters, not confirmed ovarian pathology. Any AUROC near or below B2's should be read as "not yet shown to be real signal," not as a working diagnostic.
