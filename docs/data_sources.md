# Data sources (v1)

Author: Madhumita. Written 2026-09-24 (Day 1). Owner: Madhumita.

Every fact below is tagged as one of:
- **Page states**: copied from the dataset's Kaggle page. Not independently verified.
- **Verified**: checked by us on the downloaded files.
- **Preliminary**: found in a first analysis; `ovanet/data/audit.py` (Day 2) must reproduce it before it is treated as verified.

## 1. Decision (provisional)

- **Primary dataset: Candidate A**, Kaggle "PCOS detection using ultrasound images" (author anaghachoudhari).
- **Candidate B** (Kaggle "PCOS-XAI Ultrasound: Real-World Training Dataset", author ibadeus) is **not downloaded and not used**. It stays a possible external test set only after the checks in Section 5.
- Candidate A is weak. The reasons are in Section 4, and they shape how every result is reported ("on this dataset", never "PCOS detection accuracy").

## 2. Candidate A: PCOS detection using ultrasound images

Link: (https://www.kaggle.com/datasets/anaghachoudhari/pcos-detection-using-ultrasound-images)

**Page states**
- Folders `data/train` and `data/test`, each with `infected` and `notinfected`.
- `infected`: "Images of ovaries having PCOS". `notinfected`: "Images of healthy ovaries".
- Version 1, 136.3 MB, 3,856 files. Licence: **Unknown**. Update frequency: not specified. Usability 5.63.
- Tags: Image, Health Conditions, CNN, TensorFlow, Keras.
- The page text we saw names no source, clinic, patient information, annotator or linked paper. [Madhumita: confirm the Discussion and Code tabs say nothing more about the source.]

**Verified** (`scripts/inspect_dataset.py`, 2026-09-24)
- 3,856 files: train infected 781, train notinfected 1,143, test infected 787, test notinfected 1,145.
- Extensions: 3,846 `.jpg`, 8 `.jpeg`, 2 `.png`. Colour modes: RGB 3,842, P 2, L 2.
- **10 files are unreadable zero-byte files**, all in `test/` (6 infected, 4 notinfected). Names include `OIP.jpg`, `R.jpg`, `WhatsApp Image 2022-04-01 ...`.
- **11 distinct image sizes** in the whole dataset.
- Visual inspection of about 10 images per class (Madhumita):
  - All are full transvaginal-style frames with the probe arc and black borders. None is a tight ovary crop.
  - Some have white or grey padding strips, and pairs in the same size group look like the same scene shifted or flipped.
  - Burned-in marks differ by class. Some infected frames have a small green dot. Some notinfected frames have a grey-scale bar, ruler-like ticks or small symbols.
  - Infected frames are close-ups with many dark follicle-like circles. Notinfected frames are wider fields, and an ovary is not clearly visible in several of them.

**Preliminary** (to be reproduced by `audit.py`)
- Every readable test file is a byte-for-byte (md5) copy of a train file. The two folders hold the same images.
- Only **1,921 unique images**: 779 infected, 1,142 notinfected.
- Image sizes (width x height) do not overlap between classes:
  - infected: 225x225, 233x216, 280x259, 299x266, 300x300, 320x320, 343x343
  - notinfected: 315x160, 630x436, 850x578, 984x848
- Mean brightness is about 73 (infected) versus 53 (notinfected). Mean file size is about 14 KB versus 50 KB.
- A rough perceptual-hash grouping (arbitrary threshold) suggests only a few hundred distinct scenes among the 1,921 unique files. The real number depends on the threshold justified in `audit.py`.

**Unknown**: who labelled the images, how PCOS was confirmed, whether "infected" means confirmed diagnosis or a cystic-looking appearance, the number of patients, the scanners used, the original source.
Filenames such as `OIP.jpg` and `WhatsApp Image ...` suggest images collected from the web or messaging apps rather than exported from a hospital system. This is an inference, not a stated fact.

## 3. Candidate B: PCOS-XAI Ultrasound: Real-World Training Dataset

Link: (https://www.kaggle.com/datasets/ibadeus/pcos-xai-ultrasound-dataset) **Not downloaded. Everything below is what the page states.**

- 11,784 images: `infected` 6,784 ("PCOS-positive cases"), `noninfected` 5,000 ("Healthy ovaries").
- Licence: CC0: Public Domain. Update frequency: annually. Usability 8.75.
- Provenance stated: "retrospective ultrasound studies across 3 clinics (2018-2022)", patient identifiers removed, DICOM metadata stripped, "annotated by radiology residents under consultant supervision". No clinical patient data.
- Also stated: it "simulates real-world medical data complexities", is "intentionally uncleaned", and has 1,956+ groups of identical images within and across classes.
- Resolutions stated: 255x247 up to 984x848 (infected minimum 255x247, noninfected maximum 800x600). Formats JPEG about 95%, PNG about 5%.

Things to check before trusting it:
- The page describes itself as a simulated, practice dataset and also claims clinical provenance. It is unclear which is true.
- Nothing on the page lets us verify the clinics or annotator claims.
- Identical images across classes means some images carry conflicting labels.
- The page mixes English and Spanish.
- Whether it overlaps with Candidate A (same images or derived copies) is unchecked. An overlap would make it useless as an external test set.

## 4. Side by side

| | A: anaghachoudhari | B: ibadeus |
|---|---|---|
| Images | 3,856 files, **1,921 unique** (verified/preliminary) | 11,784 (page states) |
| Classes | infected 779 / notinfected 1,142 unique | infected 6,784 / noninfected 5,000 (page states) |
| Licence | Unknown | CC0 (page states) |
| Label meaning | Unverified | "PCOS-positive" / "Healthy" (page states; unverified) |
| Patient IDs | None | None (page states) |
| Provenance | Not stated | 3 clinics (page states; unverified) |
| Train/test folders | Test is a copy of train | Not stated |
| Frame type | Full frames (verified) | Not checked |
| Status | Downloaded, audited | Not downloaded |

Why A is primary: it is in hand and inspected, its flaws are known and can be handled by our protocol, and B cannot be judged without downloading it and its own page calls it simulated.

## 5. Consequences for the project

1. The provided train/test folders are unusable. We build our own group-aware split on unique images, grouping near-duplicate augmented copies (Day 2). There are no patient IDs, so we call it a "group-aware split (near-duplicate groups)", never "patient-wise".
2. The effective sample is 1,921 unique files and probably a few hundred distinct scenes. Confidence intervals will be wide.
3. The classes look like they come from different sources (sizes, borders, burned-in marks, scale, view). Results may reflect source rather than pathology. The shortcut baseline B2 is reported next to every headline number, and the CNN baseline B1 is read with that in mind.
4. The label meaning is unverified, so the app and the pitch never say "PCOS detected". They say "risk estimate for clinician review".
5. Licence is Unknown. Raw images are never committed or redistributed. Whether bundled demo images (`data/sample_scans/`) can be public is an open decision (Section 6).
6. Gate G1 (provisional): the images are full frames, so the ovary is not the whole image. An ovary outline is needed (annotation plus, later, an ovary head). To be confirmed on Day 2.
7. The 10 zero-byte files are excluded.

## 6. Open items

- [ ] Paste both Kaggle URLs; confirm the Discussion/Code tabs of A say nothing about source or labelling.
- [ ] Reproduce all "Preliminary" numbers in `audit.py` and update this file.
- [ ] Decide with Sahina: can `data/sample_scans/` images from A be bundled in a public repo/Space, given "Unknown" licence? Options: keep the repo private, use only user-uploaded images in the live demo, or use B's images if B is verified and licensed.
- [ ] After the Day 2 audit: download B in a time-boxed check (`inspect_dataset.py`, md5/phash overlap with A, image look). Then decide whether it is usable as an external set.
- [ ] Final G1 answer (crop vs full frame) on Day 2.