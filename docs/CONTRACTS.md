# OvaNet Interface Contracts — v0.1

Status: DRAFT pending Madhumita's sign-off (requested Day 2, target: before Day 3).
Any change after freeze must be logged in DECISIONS.md and announced to both members.

## Fixed constants
- IMAGE_SIZE = 256 (raw/enh are 256×256)
- E_CNN_DIM = 512
- GRAPH_FEATURE_DIM = 5 (per-node: area_rel, eccentricity, mean_intensity, dist_to_centroid_rel, dist_to_boundary_rel)
- GCN_EMBED_DIM = 16
- RADIOMICS_FEATURE_ORDER = [mean, std, entropy, glcm_contrast, glcm_homogeneity]
- GRAPH_STAT_NAMES (fixed order) = [n_follicles, mean_nn_dist_rel, std_nn_dist_rel, mean_degree, clustering_coeff, peripheral_index, size_cv, follicle_area_fraction]
- Graph adjacency: kNN k=3, weight = exp(-d²/τ)

## C1 preprocess
preprocess(image) -> {"raw": float32[256,256] in [0,1], "enh": float32[256,256] in [0,1],
"quality": {"ok": bool, "reasons": [str], "blur_var": float}}
RAW = grayscale + resize + normalise only. ENH = denoise + CLAHE for the CNN.

## C2 vision
analyze_vision(prep) -> {"ovary_mask": uint8[256,256], "follicle_labels": int32[256,256],
"follicles": [{"id","cx","cy","area_px","eq_radius_px","eccentricity","solidity","mean_intensity"}],
"e_cnn": float32[512], "seg_method": "classical"|"unet", "warnings": [str]}

## C3 radiomics
extract(raw, mask=None) -> {"mean","std","entropy","glcm_contrast","glcm_homogeneity"}
Vector order fixed per RADIOMICS_FEATURE_ORDER. With a mask, only pixels/pairs inside it are used.

## C4 graph
build_graph(follicles, ovary_mask) -> {"n": int, "x": float32[n,5], "adj": float32[n,n],
"edges": [{"source","target","weight"}], "nodes": [{"id","x","y"}], "stats": {name: float}}
Must return valid zero/default stats when n<3.
gcn_embed(graph) -> float32[16] (zeros + flag when n<3)

## C5 clinical
{"age": float|None, "bmi": float|None, "cycle_irregular": bool|None, "hyperandrogenism": bool|None,
"lh_fsh": float|None, "amh": float|None, "years_since_menarche": float|None,
"source": "user_entered"|"synthetic_demo"}
Used only by the rule-based checklist. Not fed to the ML model in the MVP.

## C6 fusion
predict(features) -> {"risk_score": float 0-100, "band": "LOW"|"MODERATE"|"HIGH",
"family_attribution": {"radiomics","follicle","graph","cnn","gnn": float},
"top_features": [{"name","family","contribution"}],
"baseline_vote": {"votes_pcos": int, "of": 5, "label": str}, "model_version": str}

## C7 longitudinal
compare(visit_a, visit_b) -> {"trend": "IMPROVING"|"STABLE"|"WORSENING"|"INSUFFICIENT_DATA",
"deltas": {...}, "report_text": str, "is_demo_pair": bool}

## C8 pipeline
analyze(image, clinical=None) -> AnalysisResult with preprocess quality, C2 overlays,
C3, C4, C6, timings, warnings, is_mock.

## Sign-off log
- 2026-09-22 (Day 2): Sahina drafted and froze v0.1 per master prompt Section 7. Awaiting Madhumita's review.

## Addendum v0.2 (Day 1–2, from Madhumita, signed off by Sahina)

**Coordinates:** x = column, y = row, in 256×256 pixel space.

**Normaliser:** R = sqrt(ovary_area_px / π). All `_rel` values are expressed in units of R.

**Adjacency (C4):** kNN with k = min(3, n-1), symmetrised (union), no self-loops.
Weight = exp(-d²/τ), τ = median of d² over edges (τ = 1.0 if no edges exist).

**peripheral_index:** = 1 - mean(dist_to_boundary / d_max), range [0, 1].
Higher = more peripheral distribution of follicles.

**n < 3 handling:** all graph stats = 0.0 except `n_follicles`.
`GraphResult` also carries `"valid": False` in this case.

**Vision encoder (C2):** ResNet18-based encoder, output dim 512 (matches E_CNN_DIM).

**ovary_mask definition:** until a real ovary outline exists, `ovary_mask` = the
ultrasound sector/fan region, used as the analysis ROI.

**Status:** Contracts v0.1 + this addendum = v0.2, confirmed OK by both members.
Frozen as of 2026-09-23 (Day 2).

**Access pattern clarification:** All contract result objects (PreprocessResult, VisionResult,
RadiomicsResult, GraphResult, FusionResult, LongitudinalResult, AnalysisResult) are accessed
via attributes (e.g. `result.raw`, `result.quality.ok`), NOT dict keys (`result["raw"]` will
raise TypeError). Confirmed against Madhumita's real preprocess() output, Day 2.