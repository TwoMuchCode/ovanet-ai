# Masked vs Unmasked Radiomics Comparison (Day 6)

**Comparison of feature separation (Infected Mean − Normal Mean) between:**
- **v1 (Unmasked):** Whole ultrasound frame (includes scanner metadata banner, bottom calibration bar, and black background margins)
- **v3 (Masked):** Masked specifically to the ovary tissue sector ROI via `ovanet.detector.detect_follicles`

## Comparison Table

| Feature | v1 Unmasked Diff | v3 Masked Diff | Delta | Effect of Masking |
|---|:---:|:---:|:---:|---|
| `mean` | +0.0781 | +0.0136 | −0.0645 | Removes black outer borders that biased whole-image mean |
| `std` | +0.0653 | +0.0673 | +0.0021 | Preserves internal stromal variance |
| `entropy` | +0.5097 | +0.2664 | −0.2432 | True tissue entropy without artificial background zero-bins |
| `glcm_contrast` | +0.2738 | −0.3542 | −0.6281 | Massive difference: unmasked contrast was heavily inflated by outer edges |
| `glcm_homogeneity`| +0.0107 | +0.0569 | +0.0462 | Masking boosts internal stromal homogeneity separation |

### Clinical & Algorithmic Findings:
1. **Elimination of Edge Artifacts:** In unmasked ultrasound frames (`features_v1`), the hard black-to-white transition at the edge of the ultrasound fan artificially dominated GLCM contrast. Masking to the actual tissue ROI (`features_v3`) eliminates this artifact.
2. **True Parenchymal Echogenicity:** In the masked ROI, infected ovaries show consistently higher entropy and homogeneity shifts that reflect true multi-follicular stroma rather than machine layout differences.
