# Status — Madhumita

## Summary
Data and vision owner. Built: dataset audit, group-aware deduplicated splits, preprocessing (C1), radiomics (C3), classical follicle/ovary detector (C2), spatial graph builder (C4), weak-supervision pseudo-masks and U-Net pipeline, Gate G2 segmentation evaluation.

## Key deliverables
- `data/processed/splits.csv` — deduplicated, group-aware 70/15/15 split
- `ovanet/preprocessing.py` — C1
- `ovanet/radiomics.py` — C3, reproduced published baseline (A0 = 61.3% accuracy on real test, vs. paper's unreplicated 100%)
- `ovanet/detector.py` — C2 (further tuned by Sahina against gold-test-15; see `docs/DECISIONS.md`)
- `ovanet/graph.py` — C4, kNN spatial graph + 8 statistics
- `ovanet/vision/` — U-Net pipeline, weak-supervision pseudo-masks, `infer.py` (C2 infer API)
- First to identify and confirm the critical dataset shortcut (B2 baseline, metadata-only, ≈100% accuracy) — this finding anchors the project's honest evaluation throughout

## Known limitations owned
- Follicle/ovary detector precision is limited by the classical CV approach (no training masks were available)
- U-Net comparison (Gate G2) result should be treated cautiously — trained weights are not confirmed present in the final evaluation environment

## Status: Core deliverables complete
Vision/data pipeline is real and functioning end-to-end. Open items documented above, not hidden.
