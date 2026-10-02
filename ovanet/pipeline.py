"""
ovanet/pipeline.py

Real end-to-end analyze() function, replacing mocks entirely.
Orchestrates C1 -> C2 (via analyze_vision) -> C3 -> C4 -> GCN -> C6.
"""

import time
import numpy as np

from ovanet.preprocessing import preprocess
from ovanet.vision.infer import analyze_vision
from ovanet.radiomics import extract
from ovanet.graph import build_graph
from ovanet.gnn import gcn_embed
from ovanet.fusion.model import predict as fusion_predict
from ovanet.contracts import AnalysisResult, GRAPH_STAT_NAMES


def analyze(image: np.ndarray, clinical=None) -> AnalysisResult:
    t0 = time.time()
    warnings = []

    prep = preprocess(image)
    if not prep.quality.ok:
        warnings.append(f"Quality check: {prep.quality.reasons}")

    # C2: analyze_vision() uses classical detection (Gate G2 winner)
    # AND extracts real 512-dim E_cnn via ResNet-18 encoder
    vision = analyze_vision(prep)
    radiomics = extract(prep.raw, mask=vision.ovary_mask)
    graph = build_graph(vision.follicles, vision.ovary_mask)
    e_gnn = gcn_embed(graph)

    feature_dict = {
        "rad_mean": radiomics.mean, "rad_std": radiomics.std, "rad_entropy": radiomics.entropy,
        "rad_glcm_contrast": radiomics.glcm_contrast, "rad_glcm_homogeneity": radiomics.glcm_homogeneity,
    }
    for name in GRAPH_STAT_NAMES:
        feature_dict[f"graph_{name}"] = graph.stats.get(name, 0.0)
    for i, val in enumerate(e_gnn):
        feature_dict[f"e_gnn_{i}"] = float(val)

    fusion = fusion_predict(feature_dict)

    return AnalysisResult(
        quality=prep.quality,
        vision=vision,
        radiomics=radiomics,
        graph=graph,
        fusion=fusion,
        timings={"total_s": time.time() - t0},
        warnings=warnings,
        is_mock=False,
    )