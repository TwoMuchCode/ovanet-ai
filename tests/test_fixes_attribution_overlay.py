"""Regression tests for: fake attribution, random GCN embedding, graph-overlay index bug."""

import numpy as np
import pytest

from ovanet.contracts import FEATURE_FAMILIES, Follicle
from ovanet.graph import build_graph
from ovanet.visualization import graph_edge_endpoints, draw_graph_overlay


def _follicles(n, seed=1):
    rng = np.random.RandomState(seed)
    return [Follicle(id=i, cx=float(rng.uniform(60, 190)), cy=float(rng.uniform(60, 190)),
                     area_px=50.0, eq_radius_px=4.0, eccentricity=0.3, solidity=0.9,
                     mean_intensity=0.2) for i in range(1, n + 1)]  # ids start at 1, like the detector


def test_graph_edges_connect_the_right_follicles():
    fol = _follicles(12)
    g = build_graph(fol, np.ones((256, 256), np.uint8))
    assert g.edges, "expected edges"
    ends = graph_edge_endpoints(g)
    assert len(ends) == len(g.edges)  # none silently dropped
    for e, ((x1, y1), (x2, y2)) in zip(g.edges, ends):
        assert (x1, y1) == (fol[e.source].cx, fol[e.source].cy)
        assert (x2, y2) == (fol[e.target].cx, fol[e.target].cy)
    draw_graph_overlay(np.zeros((256, 256), np.float32), fol, g)  # must not raise


def test_baseline_vote_is_real_and_bounded():
    from ovanet.fusion.baseline_vote import baseline_vote
    hi = baseline_vote({"rad_mean": 9, "rad_std": 9, "rad_entropy": 99, "rad_glcm_contrast": 99, "rad_glcm_homogeneity": -1})
    lo = baseline_vote({"rad_mean": -9, "rad_std": -9, "rad_entropy": -99, "rad_glcm_contrast": -99, "rad_glcm_homogeneity": 99})
    assert hi.of == lo.of == 5
    if hi.label != "unavailable":  # features_v2.csv present
        assert hi.votes_pcos == 5 and lo.votes_pcos == 0


def test_gcn_is_deterministic_and_flagged_untrained():
    torch = pytest.importorskip("torch")
    from ovanet.gnn import gcn_embed, is_trained, GCN_WEIGHTS_PATH
    g = build_graph(_follicles(8), np.ones((256, 256), np.uint8))
    assert np.allclose(gcn_embed(g), gcn_embed(g))
    assert is_trained() == GCN_WEIGHTS_PATH.exists()


def test_fusion_attribution_is_not_a_constant():
    pytest.importorskip("torch")
    import pathlib
    if not pathlib.Path("weights/fusion_v1.joblib").exists():
        pytest.skip("fusion weights not present")
    from ovanet.fusion.model import predict
    a = predict({"rad_mean": 0.1, "rad_entropy": 3.0, "graph_n_follicles": 5.0})
    b = predict({"rad_mean": 0.5, "rad_entropy": 5.5, "graph_n_follicles": 30.0})
    assert set(a.family_attribution) == set(FEATURE_FAMILIES)
    assert a.family_attribution != b.family_attribution
    assert all(t.name != "placeholder" for t in a.top_features + b.top_features)
    assert a.family_attribution["cnn"] == 0.0  # the model has no CNN input
