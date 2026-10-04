# Status — Sahina

## Summary
Repo/contracts owner. Built: contract definitions and validators, mocks, evaluation/metrics harness, GCN (pure PyTorch), fusion model (training, calibration, attribution), real end-to-end pipeline orchestration, Streamlit app (full redesign), clinical checklist, longitudinal comparison, PDF export, ablation table, follicle-detector tuning, deployment.

## Key deliverables (final state)
- `ovanet/contracts.py`, `ovanet/mocks.py` — frozen interface contracts, C1–C8
- `ovanet/gnn.py` — GCN implementation; `is_trained()` guard since no final model is trained
- `ovanet/fusion/` — features.py, train.py, model.py (v2: radiomics+graph only, real attribution)
- `ovanet/pipeline.py` — real end-to-end `analyze()`
- `ovanet/checklist.py`, `ovanet/longitudinal.py`, `ovanet/report.py`
- `ovanet/evaluation/` — metrics, ablation, detector tuning/eval scripts
- `scripts/run_real_ablation.py` — corrected ablation table
- `app/streamlit_app.py`, `app/ovanet_ui/` — full frontend redesign
- Gold-test-15 annotations (ovary + follicle polygons) for ground-truth evaluation

## Known limitations owned
- No fine-tuned CNN baseline / Grad-CAM built (descoped)

## Status: Complete for submission
All Tier-1 items delivered and verified against real data. Open items are documented, not hidden.