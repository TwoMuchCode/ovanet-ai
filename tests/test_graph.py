"""
Tests for ovanet.graph (Contract C4).
"""

import numpy as np
import pytest

from ovanet.contracts import (
    GRAPH_FEATURE_DIM,
    GRAPH_STAT_NAMES,
    Follicle,
    GraphResult,
    validate_graph,
)
from ovanet.graph import build_graph
from ovanet.gnn import gcn_embed


def _make_dummy_follicles(n: int) -> list[Follicle]:
    follicles = []
    rng = np.random.RandomState(42)
    for i in range(1, n + 1):
        cx = float(rng.uniform(60, 190))
        cy = float(rng.uniform(60, 190))
        r = float(rng.uniform(4, 12))
        follicles.append(
            Follicle(
                id=i,
                cx=cx,
                cy=cy,
                area_px=float(np.pi * r * r),
                eq_radius_px=r,
                eccentricity=float(rng.uniform(0.1, 0.6)),
                solidity=float(rng.uniform(0.7, 0.95)),
                mean_intensity=float(rng.uniform(0.1, 0.4)),
            )
        )
    return follicles


def test_build_graph_normal():
    follicles = _make_dummy_follicles(6)
    g = build_graph(follicles)
    validate_graph(g)

    assert g.n == 6
    assert g.x.shape == (6, GRAPH_FEATURE_DIM)
    assert g.adj.shape == (6, 6)
    # Adjacency should be symmetric
    assert np.allclose(g.adj, g.adj.T)
    # Diagonal should be zero (no self-loops)
    assert np.allclose(np.diag(g.adj), 0.0)

    # Weights between 0 and 1
    assert 0.0 <= float(g.adj.min())
    assert float(g.adj.max()) <= 1.0

    # Check stats
    for name in GRAPH_STAT_NAMES:
        assert name in g.stats
        assert not np.isnan(g.stats[name])
    assert g.stats["n_follicles"] == 6.0


def test_build_graph_edge_cases_small_n():
    """n < 3 should return zeros for stats (except n_follicles) and empty/zero graph."""
    for n in [0, 1, 2]:
        follicles = _make_dummy_follicles(n)
        g = build_graph(follicles)
        validate_graph(g)

        assert g.n == n
        assert g.x.shape == (n, GRAPH_FEATURE_DIM)
        assert g.adj.shape == (n, n)
        assert len(g.edges) == 0
        assert g.stats["n_follicles"] == float(n)
        for name in GRAPH_STAT_NAMES:
            if name != "n_follicles":
                assert g.stats[name] == 0.0


def test_graph_feeds_into_gnn():
    """Verify that build_graph output feeds seamlessly into Sahina's gcn_embed."""
    follicles = _make_dummy_follicles(8)
    g = build_graph(follicles)
    embed = gcn_embed(g)
    assert embed.shape == (16,)
    assert embed.dtype == np.float32
    assert not np.isnan(embed).any()
    # Should not be all zeros for n >= 3
    assert not np.allclose(embed, 0.0)


def test_small_n_feeds_into_gnn_safely():
    for n in [0, 1, 2]:
        follicles = _make_dummy_follicles(n)
        g = build_graph(follicles)
        embed = gcn_embed(g)
        assert embed.shape == (16,)
        assert np.allclose(embed, 0.0)
