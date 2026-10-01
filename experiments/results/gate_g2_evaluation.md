# Gate G2 Evaluation: Classical Detector vs U-Net on 15 Gold-Test Images

**Date:** 2026-10-01  
**Evaluator:** Madhumita (Vision Track)  
**Ground Truth Annotations:** 15 hand-annotated test scans by Sahina (`data/gold_test_annotations/`)

## Summary Comparison Table

| Metric | Classical CV Detector | ResNet-18 U-Net | Winner |
|---|:---:|:---:|:---:|
| **Ovary Mask Dice** | **0.8727** | 0.0650 | Classical |
| **Follicle Mask Dice** | **0.0518** | 0.0282 | Classical |
| **Follicle Count MAE** | **14.60** | 2.27 | U-Net |
| **Combined Mean Dice** | **0.4622** | 0.0466 | **CLASSICAL** |

## Gate G2 Resolution

- Selected Segmentation Method in Contract C2: **`classical`**
- Both results will be transparently documented in `docs/DECISIONS.md` and the final pitch deck.
- Label-free encoder embedding $E_{cnn}$ (512-dim) will be exported via ResNet-18 bottleneck.

## Per-Image Breakdown

| Image | GT Follicles | Classical Count | U-Net Count | Classical Ovary Dice | Classical Follicle Dice |
|---|:---:|:---:|:---:|:---:|:---:|
| `img_0_1195` | 4 | 15 | 1 | 0.828 | 0.000 |
| `img_0_1590` | 4 | 21 | 1 | 0.885 | 0.270 |
| `img_0_1903` | 0 | 9 | 1 | 0.941 | 0.000 |
| `img_0_2739` | 4 | 20 | 1 | 0.899 | 0.174 |
| `img_0_3117` | 0 | 14 | 1 | 0.888 | 0.000 |
| `img_0_4200` | 0 | 31 | 1 | 0.946 | 0.000 |
| `img_0_4564` | 0 | 10 | 1 | 0.888 | 0.000 |
| `img_0_5812` | 4 | 18 | 1 | 0.720 | 0.133 |
| `img_0_6676` | 5 | 21 | 1 | 0.766 | 0.157 |
| `img_0_7453` | 6 | 16 | 1 | 0.774 | 0.030 |
| `img_0_794` | 0 | 11 | 1 | 0.906 | 0.000 |
| `img_0_820` | 1 | 22 | 1 | 0.864 | 0.000 |
| `img_0_821` | 0 | 17 | 1 | 0.958 | 0.000 |
| `img_0_9172` | 7 | 17 | 1 | 0.900 | 0.013 |
| `img_0_9342` | 0 | 12 | 1 | 0.927 | 0.000 |
