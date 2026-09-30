"""
tests/test_gnn.py

Confirms: n<3 doesn't crash (returns zeros), output shape is always correct.
"""

import numpy as np
from ovanet.gnn import gcn_embed
from ovanet.contracts import GCN_EMBED_DIM, GRAPH_FEATURE_DIM
from ovanet.mocks import mock_graph


def test_gcn_embed_normal_graph():
    graph = mock_graph(n=6)
    embed = gcn_embed(graph)
    assert embed.shape == (GCN_EMBED_DIM,)
    assert embed.dtype == np.float32


def test_gcn_embed_n_less_than_3_returns_zeros():
    graph = mock_graph(n=2)
    embed = gcn_embed(graph)
    assert embed.shape == (GCN_EMBED_DIM,)
    assert np.allclose(embed, 0.0)


def test_gcn_embed_n_zero_returns_zeros():
    graph = mock_graph(n=0)
    embed = gcn_embed(graph)
    assert embed.shape == (GCN_EMBED_DIM,)
    assert np.allclose(embed, 0.0)


def test_gcn_embed_larger_graph():
    graph = mock_graph(n=20)
    embed = gcn_embed(graph)
    assert embed.shape == (GCN_EMBED_DIM,)
    assert not np.isnan(embed).any()