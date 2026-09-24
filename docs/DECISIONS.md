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