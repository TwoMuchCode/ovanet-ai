# Decisions Log

Format: Date | Who | Decision | Why | Alternatives considered
2026-09-24 | Madhumita | Primary dataset = Kaggle "PCOS detection using ultrasound images" (anaghachoudhari), provisional | In hand and inspected; its flaws are known and handled by our protocol | Kaggle "PCOS-XAI Ultrasound" (ibadeus): not downloaded, provenance unverifiable, page calls it simulated; revisit after Day 2 audit
2026-09-24 | Madhumita | Do not use the provided train/test folders; build our own group-aware split on unique images | Preliminary: every test file is an exact md5 copy of a train file (about 1,921 unique images); augmented near-duplicates also exist | Provided split (rejected: leakage); random split (rejected: leakage)
2026-09-24 | Madhumita | Exclude the 10 zero-byte unreadable files (all in test/) | Cannot be opened | None
2026-09-24 | Madhumita | Provisional G1: images are full transvaginal frames, not tight ovary crops; ovary outline needed; final answer on Day 2 | Visual inspection of about 10 images per class | Treat whole image as ovary ROI (rejected)
2026-09-24 | Madhumita | Report shortcut baseline B2 next to every headline metric | Preliminary: image size alone separates the classes; brightness and burned-in marks also differ by class | None
2026-09-24 | Madhumita | Python: laptops 3.13.0, Colab 3.13.15, code must run on both; pin python_version in the Spaces README at deploy (Spaces default is 3.10) | All stack packages install and import on 3.13; matches Colab | 3.10 (source-only releases, end of support Oct 2026); 3.12 (not needed)
2026-09-24 | Madhumita | U-Net encoder = ResNet18 | Contract fixes E_CNN_DIM = 512, which ResNet18/34 give and MobileNetV3 does not | MobileNetV3 (rejected: dimension mismatch)
---

2026-09-23 | Madhumita + Sahina | Froze CONTRACTS.md v0.2 (coordinate rules, adjacency math, peripheral_index formula, n<3 handling, ResNet18 encoder, ovary_mask = sector region) | Needed exact math before C4/fusion code | N/A

2026-09-23 | Madhumita | G1 provisional: images are full frames, not crops | Per master doc, this means an ovary outline/head must be added to segmentation | N/A

2026-09-23 | Madhumita | Confirmed dataset issue: test folder duplicates train (~1,921 unique images), and image size alone predicts the label (shortcut) | Confirms what master doc told us to assume — group-aware split is essential | N/A

2026-09-23 | Sahina | Repo stays PRIVATE, no real dataset images in public demo, until licence is confirmed | Licence is "Unknown" — playing it safe | N/A

2026-09-23 | Sahina | Confirmed PreprocessResult (and by extension all contract objects) use attribute access (.raw), not dict access (["raw"]) | Verified against Madhumita's real preprocess() output; avoids a TypeError bug when wiring real modules into app/pipeline | N/A

2026-09-25 (Day 5) | Sahina | OOF GCN embeddings generated (rule L1 compliant, 5-fold GroupKFold) for all 1,345 train images -> data/processed/e_gnn_oof_v1.csv | Needed label-safe embeddings for fusion | N/A
2026-09-25 (Day 5) | Sahina | Gate G3 initial result: A3 AUROC=0.974, A3+E_gnn AUROC=0.984 (delta +0.010). Both suspiciously high vs paper's 61.3% baseline -- likely shortcut contamination (confirmed via B2 re-check: AUROC [FILL IN FROM A1 OUTPUT]). DECISION: E_gnn's marginal gain not trusted standalone; final G3 call deferred until shortcut-adjusted evaluation (Day 8 final table will report B2 alongside every other model, per master doc) | Matches B2=100% shortcut Madhumita found on raw pixels | N/A
2026-09-25 (Day 5) | Sahina | CP2 confirmed: real C1-C4 overlays (ovary mask, follicle graph, radiomics, GCN embedding) visible end-to-end in Streamlit app; fusion/risk score still clearly marked MOCK | Checkpoint per master doc Day 5 row | N/A
2026-09-26 (Day 6) | Sahina | Fusion v1 trained + calibrated: VAL AUROC=0.986, accuracy=0.625, sensitivity=1.0, specificity=0.589, F1=0.319. High AUROC + low accuracy/F1 + sensitivity=1.0 indicates threshold miscalibration and likely continued shortcut contamination (same root cause flagged Day 5 Gate G3). NOT a trustworthy final result -- full honest evaluation with B2 comparison deferred to Day 8 per master doc. Pipeline plumbing (preprocess->detect->radiomics->graph->gcn->fusion) is confirmed working end-to-end.
2026-09-26 | Sahina | DEFINITIVE: B2 shortcut baseline (width/height/brightness/contrast only) achieves AUROC=1.0, accuracy=98.8% on this dataset. This means the Kaggle "infected"/"notinfected" labels are confounded with image acquisition/technical metadata, not (necessarily) ovarian pathology. ALL downstream models (radiomics, graph, fusion) likely inherit this shortcut since their features are computed from these same images. This is the single most important finding of the project and must be the headline of the Day 8 evaluation, Day 9 deck, and limitations section -- not hidden or minimized.