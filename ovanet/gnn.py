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


GCN_WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "gcn_v1.pt"
_model = None


def load_gcn(path=GCN_WEIGHTS) -> GCN:
    """Loads the TRAINED GCN (trained on train split only by scripts/final_run.py)."""
    global _model
    if _model is None:
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} missing. Run: python scripts/final_run.py")
        m = GCN()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        _model = m
    return _model


def gcn_embed(graph) -> np.ndarray:
    """graph with .x [n,5], .adj [n,n], .n  ->  float32[16]. n<3 -> zeros (contract)."""
    n = graph.n if hasattr(graph, "n") else graph["n"]
    if n < 3:
        return np.zeros(GCN_EMBED_DIM, dtype=np.float32)
    x = graph.x if hasattr(graph, "x") else graph["x"]
    adj = graph.adj if hasattr(graph, "adj") else graph["adj"]
    with torch.no_grad():
        out = load_gcn()(torch.tensor(x, dtype=torch.float32), torch.tensor(adj, dtype=torch.float32))
    return out.numpy().astype(np.float32)