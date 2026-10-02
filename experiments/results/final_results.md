# Final results (2026-10-02)

Split: `splits.csv` (group-aware near-duplicate groups, seed 42). Train 1345 / Val 289 / Test 287. Test touched only by this table. 95% CIs resample groups. LogReg everywhere; scaler/calibration fit on train/val only.

**G3 (VAL AUROC):** radiomics+graph = 0.982, +E_gnn = 0.993 -> E_gnn in deployed model: **True**.

| ID | Model | AUROC [95% CI] | Accuracy | Sensitivity | Specificity | F1 | Brier |
|---|---|---|---|---|---|---|---|
| A1 | Paper vote on masked-ROI radiomics (features_v3) | 0.446 [0.323, 0.622] | 0.516 | 0.709 | 0.301 | 0.606 | n/a (vote) |
| A2 | LogReg on radiomics | 0.909 [0.852, 0.956] | 0.777 | 0.914 | 0.625 | 0.812 | 0.164 [0.096, 0.237] |
| A3 | A2 + follicle/graph stats | 0.972 [0.946, 0.991] | 0.875 | 0.954 | 0.787 | 0.889 | 0.086 [0.045, 0.136] |
| A4 | A3 + PCA(E_cnn) [ImageNet encoder] | 1.000 [1.000, 1.000] | 0.993 | 1.000 | 0.985 | 0.993 | 0.006 [0.001, 0.012] |
| A5 | A4 + E_gnn (OOF train / final-GCN val,test) | 1.000 [1.000, 1.000] | 0.997 | 1.000 | 0.993 | 0.997 | 0.006 [0.002, 0.011] |
| B1 | PCA(E_cnn) only (no plain ResNet18 classifier trained) | 1.000 [1.000, 1.000] | 0.997 | 1.000 | 0.993 | 0.997 | 0.010 [0.005, 0.015] |
| B2 | SHORTCUT: width/height/brightness/contrast | 0.990 [0.973, 0.999] | 0.763 | 1.000 | 0.500 | 0.816 | 0.135 [0.078, 0.196] |
| DEPLOYED | fusion-v2-radiomics+graph+gnn + Platt (val) | 0.991 [0.976, 0.998] | 0.941 | 0.947 | 0.934 | 0.944 | 0.038 [0.020, 0.061] |

## Conclusion (read this first)

- **B2 (image size/brightness/contrast only) reaches AUROC 0.990.** Acquisition format alone separates the two classes in this dataset, so no row above is evidence of PCOS detection. Rows near B2 (especially A4, A5, B1 using CNN embeddings) mostly measure image format.
- The low rows are the more honest ones: they show texture/follicle-graph features add little beyond format on this data.
- Label meaning ('infected') is unverified; single source dataset; not clinically validated; not a diagnosis.
- Follicle detector not validated: follicle Dice 0.05 and count MAE 14.6 on 15 gold-test images; the 'ovary mask' is close to the whole frame (Dice 0.873 vs 0.793 for a full-frame mask).
- Gate G2 (classical vs U-Net) was NOT run validly: no trained U-Net weights exist; classical is used by default, untested against a trained U-Net.
- E_cnn comes from an ImageNet-pretrained encoder, not a label-free trained U-Net; it is not used by the deployed model.
- Not done: dev-15 annotation/detector tuning, flip/rotation near-duplicate check (possible residual leakage), external dataset, plain CNN classifier, PDF report.