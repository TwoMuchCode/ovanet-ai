# Ablation Table — Real Features (OvaNet AI)

**Date:** 2026-10-02  
**Train:** 1345 samples | **Test:** 287 samples  
**Method:** Logistic Regression (max_iter=2000, C=1.0)  
**Split:** Group-aware train/test from `splits.csv` (NO re-splitting)  

## Results

| Model | Description | AUROC | 95% CI | Accuracy | F1 | Sensitivity | Specificity | # Features |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A0** | Radiomics only (LR on 5 features) | 0.4497 | [0.378, 0.524] | 0.5296 | 0.6197 | 0.7285 | 0.3088 | 5 |
| **A1** | Radiomics on masked ovary ROI | 0.4497 | [0.378, 0.524] | 0.5296 | 0.6197 | 0.7285 | 0.3088 | 5 |
| **A2** | LogReg on radiomics (F_k) | 0.4497 | [0.378, 0.524] | 0.5296 | 0.6197 | 0.7285 | 0.3088 | 5 |
| **A3** | Radiomics + graph stats | 0.5986 | [0.538, 0.666] | 0.5854 | 0.6722 | 0.8079 | 0.3382 | 13 |
| **A4** | Radiomics + graph + PCA(E_cnn) | 1.0000 | [1.000, 1.000] | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 23 |
| **A5** | Full OvaNet (all features) | 1.0000 | [1.000, 1.000] | 0.9826 | 0.9837 | 1.0000 | 0.9632 | 39 |
| **B1** | CNN embedding only (PCA E_cnn) | 1.0000 | [1.000, 1.000] | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 10 |
| **B2** | [!] SHORTCUT (width/height/brightness/contrast) | **1.0000** ⚠️ | [1.000, 1.000] | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 4 |

## Key Findings

1. **B2 Shortcut Baseline** uses ONLY image metadata (width, height, brightness, contrast) — NO medical content. If B2 AUROC ≈ 1.0, ALL models above are likely contaminated by acquisition bias.
2. **A0→A5 progression** shows marginal gains from adding graph and CNN features, but these gains must be interpreted cautiously given B2.
3. **Honest conclusion:** The dataset's PCOS/normal labels correlate with image acquisition parameters (resolution, brightness), not necessarily ovarian pathology. Multi-center validation is required before any clinical claims.
