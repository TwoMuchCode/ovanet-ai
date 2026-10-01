"""
ovanet/evaluation/oof_gnn.py

Produces out-of-fold (OOF) GCN embeddings for the TRAIN split, per rule L1:
no graph's embedding comes from a model trained on that same graph's label.

Trains a small supervised GCN classifier per fold (since gcn_embed() in gnn.py
is untrained/random-init), then extracts embeddings from the held-out fold.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from PIL import Image

from ovanet.preprocessing import preprocess
from ovanet.detector import detect_follicles
from ovanet.graph import build_graph
from ovanet.gnn import GCN
from ovanet.contracts import GCN_EMBED_DIM

DATA_ROOT = "data/raw/pcos_primary/data"


def build_real_graph_for_row(row):
    """Loads an image, runs preprocess -> detect -> build_graph. Returns (x, adj, n)."""
    path = row["path"].replace("test\\", "train\\").replace("test/", "train/")
    full_path = f"{DATA_ROOT}/{path}"
    img = np.array(Image.open(full_path).convert("L"))
    prep = preprocess(img)
    vision = detect_follicles(prep.raw)
    graph = build_graph(vision.follicles, vision.ovary_mask)
    return graph


class SupervisedGCN(nn.Module):
    """GCN backbone + a classification head, trained per-fold then discarded.
    The embedding (pre-head) is what gets saved as the OOF feature."""

    def __init__(self):
        super().__init__()
        self.backbone = GCN()
        self.head = nn.Linear(GCN_EMBED_DIM, 1)

    def forward(self, x, adj):
        embed = self.backbone(x, adj)
        logit = self.head(embed)
        return embed, logit


def train_fold(train_graphs, train_labels, epochs=30, lr=1e-3):
    model = SupervisedGCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for graph, label in zip(train_graphs, train_labels):
            if graph.n < 3:
                continue  # skip degenerate graphs during training
            x = torch.tensor(graph.x, dtype=torch.float32)
            adj = torch.tensor(graph.adj, dtype=torch.float32)
            y = torch.tensor([float(label)])

            optimizer.zero_grad()
            _, logit = model(x, adj)
            loss = loss_fn(logit.view(1), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return model


def get_embedding(model, graph):
    if graph.n < 3:
        return np.zeros(GCN_EMBED_DIM, dtype=np.float32)
    x = torch.tensor(graph.x, dtype=torch.float32)
    adj = torch.tensor(graph.adj, dtype=torch.float32)
    with torch.no_grad():
        embed, _ = model(x, adj)
    return embed.numpy().astype(np.float32)


def main():
    df = pd.read_csv("data/processed/features_v2.csv")
    train_df = df[df["split"] == "train"].reset_index(drop=True)

    print(f"Building real graphs for {len(train_df)} train images... (this may take a while)")
    graphs = [build_real_graph_for_row(row) for _, row in train_df.iterrows()]
    labels = (train_df["label"] == "infected").astype(int).values
    groups = train_df["group_id"].values

    gkf = GroupKFold(n_splits=5)
    oof_embeddings = np.zeros((len(train_df), GCN_EMBED_DIM), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, labels, groups)):
        print(f"Fold {fold+1}/5: training on {len(train_idx)}, embedding {len(val_idx)}")
        fold_graphs = [graphs[i] for i in train_idx]
        fold_labels = [labels[i] for i in train_idx]
        model = train_fold(fold_graphs, fold_labels)

        for i in val_idx:
            oof_embeddings[i] = get_embedding(model, graphs[i])

    emb_df = pd.DataFrame(oof_embeddings, columns=[f"e_gnn_{i}" for i in range(GCN_EMBED_DIM)])
    out_df = pd.concat([train_df[["path", "label", "group_id"]].reset_index(drop=True), emb_df], axis=1)
    out_df.to_csv("data/processed/e_gnn_oof_v1.csv", index=False)
    print(f"Saved {len(out_df)} OOF embeddings to data/processed/e_gnn_oof_v1.csv")


if __name__ == "__main__":
    main()