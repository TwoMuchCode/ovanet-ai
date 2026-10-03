"""
ovanet/gnn.py

Small GCN in plain PyTorch (no PyTorch Geometric -- avoids install pain on
Windows/Colab/Spaces). Dense adjacency, since graphs have <~50 nodes.

gcn_embed(graph) -> float32[16], matching contract C4's GNN output shape.
n < 3 returns zeros (contract rule).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ovanet.contracts import GCN_EMBED_DIM, GRAPH_FEATURE_DIM


class GCNLayer(nn.Module):
    """One graph-conv layer: aggregate neighbor features via normalized adjacency, then linear + ReLU."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj_norm):
        # x: [n, in_dim], adj_norm: [n, n] (already normalized)
        agg = torch.matmul(adj_norm, x)  # [n, in_dim]
        return torch.relu(self.linear(agg))


class GCN(nn.Module):
    """1-2 graph-conv layers + global mean/max pool -> 16-dim embedding."""

    def __init__(self, in_dim=GRAPH_FEATURE_DIM, hidden_dim=32, embed_dim=GCN_EMBED_DIM):
        super().__init__()
        self.conv1 = GCNLayer(in_dim, hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, hidden_dim)
        self.project = nn.Linear(hidden_dim * 2, embed_dim)  # *2 for mean+max pool concat

    def forward(self, x, adj):
        adj_norm = self._normalize_adj(adj)
        h = self.conv1(x, adj_norm)
        h = self.conv2(h, adj_norm)
        mean_pool = h.mean(dim=0)
        max_pool = h.max(dim=0).values
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        return self.project(pooled)

    @staticmethod
    def _normalize_adj(adj):
        """Symmetric normalization: D^-0.5 * A * D^-0.5, with self-loops added."""
        n = adj.shape[0]
        adj_self = adj + torch.eye(n)
        deg = adj_self.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        d_mat = torch.diag(deg_inv_sqrt)
        return d_mat @ adj_self @ d_mat


# Optional trained weights. No trained GCN is shipped yet: the fusion model was fit on
# out-of-fold embeddings from throw-away per-fold GCNs (ovanet/evaluation/oof_gnn.py).
GCN_WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "weights" / "gcn_v1.pt"


def _build_default_model():
    """Deterministic model; loads trained weights if they exist. Returns (model, is_trained)."""
    with torch.random.fork_rng():
        torch.manual_seed(0)  # never random between runs
        model = GCN()
    trained = False
    if GCN_WEIGHTS_PATH.exists():
        model.load_state_dict(torch.load(GCN_WEIGHTS_PATH, map_location="cpu", weights_only=True))
        trained = True
    model.eval()
    return model, trained


_default_model, _IS_TRAINED = _build_default_model()


def is_trained() -> bool:
    """True only if trained GCN weights were loaded. Untrained embeddings must not feed fusion."""
    return _IS_TRAINED


def gcn_embed(graph) -> np.ndarray:
    """
    graph: object/dict with .x (float32[n,5]) and .adj (float32[n,n]), matching C4.
    Returns float32[16]. n < 3 -> zeros (contract rule).
    """
    n = graph.n if hasattr(graph, "n") else graph["n"]
    if n < 3:
        return np.zeros(GCN_EMBED_DIM, dtype=np.float32)

    x = graph.x if hasattr(graph, "x") else graph["x"]
    adj = graph.adj if hasattr(graph, "adj") else graph["adj"]

    x_t = torch.tensor(x, dtype=torch.float32)
    adj_t = torch.tensor(adj, dtype=torch.float32)

    with torch.no_grad():
        embed = _default_model(x_t, adj_t)

    return embed.numpy().astype(np.float32)

# ---------------------------------------------------------------------------
# is_trained() -- used by pipeline.py/app to know whether gcn_embed() output
# is a real learned embedding or just an untrained/random-init forward pass.
# No final GCN has been trained yet (see docs/DECISIONS.md, issue E), so this
# currently always returns False until weights/gcn_v1.pt exists.
# ---------------------------------------------------------------------------

import os

def is_trained() -> bool:
    return os.path.exists("weights/gcn_v1.pt")