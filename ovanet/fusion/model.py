"""
ovanet/fusion/model.py

Fusion MLP skeleton (Day 3). Small per-family projection layers so a future
512-dim CNN embedding doesn't swamp 5 radiomics numbers. Trained here only on
synthetic features as a smoke test (rule S1) — never reported as a result.
"""

import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    def __init__(self, radiomics_dim=5, follicle_graph_dim=8, cnn_dim=512, gnn_dim=16,
                 proj_dim=16, hidden_dim=32):
        super().__init__()
        # Per-family projections keep the 512-dim CNN embedding from swamping
        # the 5-dim radiomics vector once PCA(e_cnn) is added on Day 6.
        self.radiomics_proj = nn.Linear(radiomics_dim, proj_dim)
        self.graph_proj = nn.Linear(follicle_graph_dim, proj_dim)
        self.cnn_proj = nn.Linear(cnn_dim, proj_dim)
        self.gnn_proj = nn.Linear(gnn_dim, proj_dim)

        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, radiomics, graph_feats, cnn_emb, gnn_emb):
        r = torch.relu(self.radiomics_proj(radiomics))
        g = torch.relu(self.graph_proj(graph_feats))
        c = torch.relu(self.cnn_proj(cnn_emb))
        n = torch.relu(self.gnn_proj(gnn_emb))
        fused = torch.cat([r, g, c, n], dim=-1)
        return self.classifier(fused).squeeze(-1)


if __name__ == "__main__":
    # Smoke test on SYNTHETIC data only -- confirms the model trains, nothing more.
    torch.manual_seed(42)
    model = FusionMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    n_samples = 100
    radiomics = torch.randn(n_samples, 5)
    graph_feats = torch.randn(n_samples, 8)
    cnn_emb = torch.randn(n_samples, 512)
    gnn_emb = torch.randn(n_samples, 16)
    labels = torch.randint(0, 2, (n_samples,)).float()

    print("Fusion MLP smoke test (SYNTHETIC data, plumbing only):")
    for epoch in range(20):
        optimizer.zero_grad()
        preds = model(radiomics, graph_feats, cnn_emb, gnn_emb)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f"  epoch {epoch}: loss = {loss.item():.4f}")

    print("Smoke test complete -- loss should have decreased.")