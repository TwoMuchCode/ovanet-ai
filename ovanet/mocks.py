"""
ovanet/mocks.py

Mock implementations of every contract (C1-C8), used until the real modules land.
Everything here is fake data. AnalysisResult.is_mock=True is the flag the app
uses to show a MOCK badge (rule S3: mock outputs are never shown as real results).
"""

import time
import numpy as np

from ovanet.contracts import (
    IMAGE_SIZE, E_CNN_DIM, GRAPH_FEATURE_DIM, GCN_EMBED_DIM, GRAPH_STAT_NAMES,
    QualityCheck, PreprocessResult,
    Follicle, VisionResult,
    RadiomicsResult,
    GraphEdge, GraphNode, GraphResult,
    TopFeature, BaselineVote, FusionResult,
    LongitudinalResult,
    AnalysisResult,
)

_RNG = np.random.default_rng(42)


def mock_preprocess() -> PreprocessResult:
    raw = _RNG.random((IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    enh = _RNG.random((IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    return PreprocessResult(
        raw=raw, enh=enh,
        quality=QualityCheck(ok=True, reasons=[], blur_var=123.4),
    )


def mock_vision(n_follicles: int = 6) -> VisionResult:
    ovary_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    ovary_mask[60:200, 60:200] = 1
    follicle_labels = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32)
    follicles = []
    for i in range(1, n_follicles + 1):
        cx, cy = _RNG.uniform(70, 190), _RNG.uniform(70, 190)
        r = _RNG.uniform(4, 14)
        follicles.append(Follicle(
            id=i, cx=float(cx), cy=float(cy),
            area_px=float(np.pi * r * r), eq_radius_px=float(r),
            eccentricity=float(_RNG.uniform(0, 0.6)),
            solidity=float(_RNG.uniform(0.8, 1.0)),
            mean_intensity=float(_RNG.uniform(0.2, 0.6)),
        ))
        y0, y1 = int(cy - r), int(cy + r)
        x0, x1 = int(cx - r), int(cx + r)
        follicle_labels[max(0, y0):y1, max(0, x0):x1] = i
    return VisionResult(
        ovary_mask=ovary_mask,
        follicle_labels=follicle_labels,
        follicles=follicles,
        e_cnn=_RNG.normal(size=E_CNN_DIM).astype(np.float32),
        seg_method="classical",
        warnings=["MOCK: synthetic follicles, not from a real image"],
    )


def mock_radiomics() -> RadiomicsResult:
    return RadiomicsResult(
        mean=float(_RNG.uniform(0.3, 0.6)),
        std=float(_RNG.uniform(0.1, 0.3)),
        entropy=float(_RNG.uniform(3.0, 6.0)),
        glcm_contrast=float(_RNG.uniform(5.0, 50.0)),
        glcm_homogeneity=float(_RNG.uniform(0.2, 0.8)),
    )


def mock_graph(n: int = 6) -> GraphResult:
    x = _RNG.random((n, GRAPH_FEATURE_DIM)).astype(np.float32)
    adj = _RNG.random((n, n)).astype(np.float32)
    adj = (adj + adj.T) / 2
    np.fill_diagonal(adj, 0.0)
    edges = [GraphEdge(source=i, target=(i + 1) % n, weight=float(adj[i, (i + 1) % n]))
             for i in range(n)] if n > 0 else []
    nodes = [GraphNode(id=i, x=float(_RNG.uniform(0, IMAGE_SIZE)), y=float(_RNG.uniform(0, IMAGE_SIZE)))
             for i in range(n)]
    stats = {name: float(_RNG.uniform(0, 1)) for name in GRAPH_STAT_NAMES}
    stats["n_follicles"] = float(n)
    return GraphResult(n=n, x=x, adj=adj, edges=edges, nodes=nodes, stats=stats)


def mock_gcn_embed(n: int) -> np.ndarray:
    if n < 3:
        return np.zeros(GCN_EMBED_DIM, dtype=np.float32)
    return _RNG.normal(size=GCN_EMBED_DIM).astype(np.float32)


def mock_fusion() -> FusionResult:
    risk = float(_RNG.uniform(0, 100))
    band = "LOW" if risk < 33 else "MODERATE" if risk < 66 else "HIGH"
    return FusionResult(
        risk_score=risk,
        band=band,
        family_attribution={"radiomics": 0.3, "follicle": 0.25, "graph": 0.2, "cnn": 0.15, "gnn": 0.1},
        top_features=[
            TopFeature(name="glcm_contrast", family="radiomics", contribution=0.18),
            TopFeature(name="follicle_count", family="follicle", contribution=0.14),
            TopFeature(name="peripheral_index", family="graph", contribution=0.11),
        ],
        baseline_vote=BaselineVote(votes_pcos=3, of=5, label="MOCK"),
        model_version="mock-v0",
    )


def mock_longitudinal() -> LongitudinalResult:
    return LongitudinalResult(
        trend="STABLE",
        deltas={"risk_score": 0.0},
        report_text="MOCK/DEMO: synthetic two-visit comparison, not a real patient.",
        is_demo_pair=True,
    )


def mock_analyze(image=None, clinical=None) -> AnalysisResult:
    """Mock C8 pipeline. Same signature/shape as the future ovanet.pipeline.analyze()."""
    t0 = time.time()
    prep = mock_preprocess()
    vision = mock_vision()
    radiomics = mock_radiomics()
    graph = mock_graph(n=len(vision.follicles))
    fusion = mock_fusion()
    return AnalysisResult(
        quality=prep.quality,
        vision=vision,
        radiomics=radiomics,
        graph=graph,
        fusion=fusion,
        timings={"total_s": time.time() - t0},
        warnings=["MOCK DATA -- not a real analysis"],
        is_mock=True,
    )