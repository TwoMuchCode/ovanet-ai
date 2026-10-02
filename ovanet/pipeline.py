"""ovanet/pipeline.py -- real end-to-end analyze() (C1 -> C2 -> C3 -> C4 -> [GCN] -> C6)."""
import time

import numpy as np

from ovanet.contracts import GRAPH_STAT_NAMES, AnalysisResult
from ovanet.fusion.model import load_bundle, predict as fusion_predict
from ovanet.gnn import gcn_embed
from ovanet.graph import build_graph
from ovanet.preprocessing import preprocess
from ovanet.radiomics import extract
from ovanet.vision.infer import analyze_vision

SHORTCUT_WARNING = (
    "Training data has a size/brightness shortcut: image size and brightness alone separate the two "
    "classes perfectly (B2 AUROC=1.0). This score is NOT evidence of PCOS and must not be read as a "
    "diagnosis or risk."
)
DETECTOR_WARNING = (
    "Follicle detection is not validated (follicle Dice 0.05 on 15 gold images). "
    "Follicle count and graph features are indicative only."
)


def analyze(image: np.ndarray, clinical=None) -> AnalysisResult:
    t0 = time.time()
    warnings = [SHORTCUT_WARNING, DETECTOR_WARNING]

    prep = preprocess(image)
    if not prep.quality.ok:
        warnings.append(f"Quality check: {prep.quality.reasons}")

    vision = analyze_vision(prep)
    radiomics = extract(prep.raw, mask=vision.ovary_mask)
    graph = build_graph(vision.follicles, vision.ovary_mask)

    fd = {
        "rad_mean": radiomics.mean, "rad_std": radiomics.std, "rad_entropy": radiomics.entropy,
        "rad_glcm_contrast": radiomics.glcm_contrast, "rad_glcm_homogeneity": radiomics.glcm_homogeneity,
    }
    for name in GRAPH_STAT_NAMES:
        fd[f"graph_{name}"] = graph.stats.get(name, 0.0)
    if load_bundle()["include_gnn"]:
        for i, val in enumerate(gcn_embed(graph)):
            fd[f"e_gnn_{i}"] = float(val)

    fusion = fusion_predict(fd)
    return AnalysisResult(
        quality=prep.quality, vision=vision, radiomics=radiomics, graph=graph, fusion=fusion,
        timings={"total_s": time.time() - t0}, warnings=warnings, is_mock=False,
    )