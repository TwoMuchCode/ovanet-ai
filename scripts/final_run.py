"""
scripts/final_run.py  --  python scripts/final_run.py

1. graphs (same detector inputs as features: enh= passed), cached
2. GCN: OOF embeddings for TRAIN (GroupKFold); final GCN trained on TRAIN only -> embeddings for VAL/TEST (L1/L2)
3. G3 on VAL: does E_gnn beat graph stats?
4. Deployed model: LogReg(scaled) fit on TRAIN, Platt calibration fit on VAL only
5. Final table on TEST (touched here only), group-bootstrap 95% CIs, Brier
"""
import datetime
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ovanet.contracts import GCN_EMBED_DIM
from ovanet.detector import detect_follicles
from ovanet.evaluation.metrics import bootstrap_ci_by_group, compute_metrics
from ovanet.gnn import GCN
from ovanet.graph import build_graph
from ovanet.preprocessing import preprocess

RAW_ROOT = Path("data/raw/pcos_primary/data")
V1 = Path("data/processed/features_v1.csv")
V3 = Path("data/processed/features_v3.csv")
SPLITS = Path("data/processed/splits.csv")
CACHE = Path("data/processed/graphs_v3.pkl")
EGNN_CSV = Path("data/processed/e_gnn_v2.csv")
OUT_MD = Path("experiments/results/final_results.md")
OUT_JSON = Path("experiments/results/final_results.json")
W = Path("weights")

RAD = ["rad_mean", "rad_std", "rad_entropy", "rad_glcm_contrast", "rad_glcm_homogeneity"]
GRAPH = ["graph_n_follicles", "graph_mean_nn_dist_rel", "graph_std_nn_dist_rel", "graph_mean_degree",
         "graph_clustering_coeff", "graph_peripheral_index", "graph_size_cv", "graph_follicle_area_fraction"]
ECNN = [f"ecnn_pca_{i}" for i in range(10)]
EGNN = [f"e_gnn_{i}" for i in range(GCN_EMBED_DIM)]
B2 = ["width", "height", "brightness_mean", "contrast_std"]
EPOCHS = 20


# ---------------- data ----------------
def load_table():
    v3 = pd.read_csv(V3)
    sp = pd.read_csv(SPLITS)
    drop = [c for c in ["label", "group_id", "split"] + B2 if c in v3.columns]
    df = v3.drop(columns=drop).merge(sp[["path", "label", "group_id", "split"] + B2], on="path", how="inner")
    df = df.reset_index(drop=True)
    for c in RAD + GRAPH + ECNN:
        assert c in df.columns, f"column {c} missing in features_v3"
    df["y"] = (df["label"] == "infected").astype(int)
    return df


def build_graphs(df):
    if CACHE.exists():
        g = pickle.load(open(CACHE, "rb"))
        if set(df["path"]) <= set(g):
            return g
    graphs = {}
    for p in tqdm(df["path"], desc="graphs"):
        img = np.array(Image.open(RAW_ROOT / p.replace("\\", "/")).convert("L"))
        prep = preprocess(img)
        vis = detect_follicles(prep.raw, enh=prep.enh)
        g = build_graph(vis.follicles, vis.ovary_mask)
        graphs[p] = {"x": np.asarray(g.x, np.float32), "adj": np.asarray(g.adj, np.float32), "n": int(g.n)}
    pickle.dump(graphs, open(CACHE, "wb"))
    return graphs


# ---------------- GCN ----------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = GCN()
        self.head = nn.Linear(GCN_EMBED_DIM, 1)

    def forward(self, x, adj):
        e = self.backbone(x, adj)
        return e, self.head(e)


def train_net(gs, ys, seed=42):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lf = nn.BCEWithLogitsLoss()
    idx = [i for i, g in enumerate(gs) if g["n"] >= 3]
    for _ in range(EPOCHS):
        rng.shuffle(idx)
        for i in idx:
            opt.zero_grad()
            _, logit = net(torch.tensor(gs[i]["x"]), torch.tensor(gs[i]["adj"]))
            lf(logit.view(1), torch.tensor([float(ys[i])])).backward()
            opt.step()
    net.eval()
    return net


def embed(net, g):
    if g["n"] < 3:
        return np.zeros(GCN_EMBED_DIM, np.float32)
    with torch.no_grad():
        e, _ = net(torch.tensor(g["x"]), torch.tensor(g["adj"]))
    return e.numpy().astype(np.float32)


def compute_egnn(df, graphs):
    tr = np.where(df["split"] == "train")[0]
    other = np.where(df["split"] != "train")[0]
    y, grp = df["y"].values, df["group_id"].values
    G = [graphs[p] for p in df["path"]]
    E = np.zeros((len(df), GCN_EMBED_DIM), np.float32)
    for k, (a, b) in enumerate(GroupKFold(5).split(tr, y[tr], grp[tr])):
        print(f"GCN OOF fold {k + 1}/5")
        net = train_net([G[tr[i]] for i in a], [y[tr[i]] for i in a])
        for i in b:
            E[tr[i]] = embed(net, G[tr[i]])          # train rows: out-of-fold (L1)
    print("GCN final (train only)")
    final = train_net([G[i] for i in tr], [y[i] for i in tr])
    for i in other:
        E[i] = embed(final, G[i])                     # val/test: model never saw them
    W.mkdir(exist_ok=True)
    torch.save(final.backbone.state_dict(), W / "gcn_v1.pt")
    out = pd.DataFrame(E, columns=EGNN)
    out.insert(0, "path", df["path"].values)
    out.to_csv(EGNN_CSV, index=False)
    return out


# ---------------- models ----------------
def fit_lr(df, cols, tr):
    X = np.nan_to_num(df[cols].values.astype(np.float64))
    sc = StandardScaler().fit(X[tr])
    lr = LogisticRegression(max_iter=3000, random_state=42).fit(sc.transform(X[tr]), df["y"].values[tr])
    return sc, lr, X


def fit_vote(d, y):
    mid, dr = {}, {}
    for c in RAD:
        m1, m0 = d.loc[y == 1, c].mean(), d.loc[y == 0, c].mean()
        mid[c], dr[c] = float((m1 + m0) / 2), (1.0 if m1 >= m0 else -1.0)
    return {"cols": RAD, "mid": mid, "dir": dr}


def vote_counts(v, d):
    return sum(((d[c] - v["mid"][c]) * v["dir"][c] > 0).astype(int) for c in v["cols"]).values


def row(rid, desc, y, prob, grp, nfeat, brier=True):
    pred = (prob >= 0.5).astype(int)
    m = compute_metrics(y, pred, prob)
    ci = bootstrap_ci_by_group(y, pred, prob, grp, n_boot=500)
    r = {"id": rid, "desc": desc, "n_features": nfeat, "n_test": int(len(y)), "brier_valid": brier}
    for k in ["auroc", "accuracy", "sensitivity", "specificity", "f1", "brier"]:
        r[k] = float(m[k])
        r[k + "_ci"] = [float(c) for c in ci.get(k, (float("nan"), float("nan")))]
    print(f"{rid}: AUROC={r['auroc']:.3f} acc={r['accuracy']:.3f}")
    return r


def main():
    df = load_table()
    graphs = build_graphs(df)
    eg = compute_egnn(df, graphs)
    df = df.merge(eg, on="path", how="left")

    tr = (df["split"] == "train").values
    va = (df["split"] == "val").values
    te = (df["split"] == "test").values
    y = df["y"].values
    grp_te = df["group_id"].values[te]
    print(f"train={tr.sum()} val={va.sum()} test={te.sum()}")

    # ---- G3 on VAL ----
    def val_auc(cols):
        sc, lr, X = fit_lr(df, cols, tr)
        return roc_auc_score(y[va], lr.decision_function(sc.transform(X[va])))
    auc_a3, auc_g = val_auc(RAD + GRAPH), val_auc(RAD + GRAPH + EGNN)
    include_gnn = auc_g > auc_a3 + 0.01
    print(f"G3 (VAL AUROC): rad+graph={auc_a3:.3f}  +E_gnn={auc_g:.3f}  -> include E_gnn: {include_gnn}")

    # ---- deployed model ----
    cols = RAD + GRAPH + (EGNN if include_gnn else [])
    sc, lr, X = fit_lr(df, cols, tr)
    dec_va = lr.decision_function(sc.transform(X[va]))
    cal = LogisticRegression().fit(dec_va.reshape(-1, 1), y[va])      # Platt, VAL only
    vote = fit_vote(df[tr], y[tr])
    version = "fusion-v2-radiomics+graph" + ("+gnn" if include_gnn else "")
    joblib.dump({"cols": cols, "scaler": sc, "lr": lr, "cal": cal, "vote": vote,
                 "include_gnn": bool(include_gnn), "version": version}, W / "fusion_v2.joblib")

    # ---- table on TEST ----
    rows = []
    v1 = pd.read_csv(V1).drop_duplicates("path")
    a0 = df[["path"]].merge(v1[["path"] + RAD] if all(c in v1.columns for c in RAD) else v1[["path"]], on="path", how="left")
    if all(c in a0.columns for c in RAD) and not a0[RAD].isna().any().any():
        v0 = fit_vote(a0[tr], y[tr])
        rows.append(row("A0", "Paper vote on full-image radiomics (features_v1)", y[te], vote_counts(v0, a0[te]) / 5.0, grp_te, 5, False))
    else:
        print("A0 skipped: features_v1 radiomics missing")
    rows.append(row("A1", "Paper vote on masked-ROI radiomics (features_v3)", y[te], vote_counts(vote, df[te]) / 5.0, grp_te, 5, False))

    def lr_row(rid, desc, cc):
        s, m, XX = fit_lr(df, cc, tr)
        return row(rid, desc, y[te], m.predict_proba(s.transform(XX[te]))[:, 1], grp_te, len(cc))

    rows.append(lr_row("A2", "LogReg on radiomics", RAD))
    rows.append(lr_row("A3", "A2 + follicle/graph stats", RAD + GRAPH))
    rows.append(lr_row("A4", "A3 + PCA(E_cnn) [ImageNet encoder]", RAD + GRAPH + ECNN))
    rows.append(lr_row("A5", "A4 + E_gnn (OOF train / final-GCN val,test)", RAD + GRAPH + ECNN + EGNN))
    rows.append(lr_row("B1", "PCA(E_cnn) only (no plain ResNet18 classifier trained)", ECNN))
    rows.append(lr_row("B2", "SHORTCUT: width/height/brightness/contrast", B2))
    dec_te = lr.decision_function(sc.transform(X[te]))
    rows.append(row("DEPLOYED", f"{version} + Platt (val)", y[te], cal.predict_proba(dec_te.reshape(-1, 1))[:, 1], grp_te, len(cols)))

    b2 = next(r for r in rows if r["id"] == "B2")
    shortcut = b2["auroc"] > 0.9
    fmt = lambda r, k: f"{r[k]:.3f} [{r[k + '_ci'][0]:.3f}, {r[k + '_ci'][1]:.3f}]"
    L = [f"# Final results ({datetime.date.today()})", "",
         f"Split: `splits.csv` (group-aware near-duplicate groups, seed 42). Train {tr.sum()} / Val {va.sum()} / Test {te.sum()}. "
         "Test touched only by this table. 95% CIs resample groups. LogReg everywhere; scaler/calibration fit on train/val only.", "",
         f"**G3 (VAL AUROC):** radiomics+graph = {auc_a3:.3f}, +E_gnn = {auc_g:.3f} -> E_gnn in deployed model: **{include_gnn}**.", "",
         "| ID | Model | AUROC [95% CI] | Accuracy | Sensitivity | Specificity | F1 | Brier |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        br = fmt(r, "brier") if r["brier_valid"] else "n/a (vote)"
        L.append(f"| {r['id']} | {r['desc']} | {fmt(r, 'auroc')} | {r['accuracy']:.3f} | {r['sensitivity']:.3f} | {r['specificity']:.3f} | {r['f1']:.3f} | {br} |")
    L += ["", "## Conclusion (read this first)", ""]
    if shortcut:
        L += [f"- **B2 (image size/brightness/contrast only) reaches AUROC {b2['auroc']:.3f}.** Acquisition format alone separates the two classes in this dataset, "
              "so no row above is evidence of PCOS detection. Rows near B2 (especially A4, A5, B1 using CNN embeddings) mostly measure image format.",
              "- The low rows are the more honest ones: they show texture/follicle-graph features add little beyond format on this data."]
    L += ["- Label meaning ('infected') is unverified; single source dataset; not clinically validated; not a diagnosis.",
          "- Follicle detector not validated: follicle Dice 0.05 and count MAE 14.6 on 15 gold-test images; the 'ovary mask' is close to the whole frame (Dice 0.873 vs 0.793 for a full-frame mask).",
          "- Gate G2 (classical vs U-Net) was NOT run validly: no trained U-Net weights exist; classical is used by default, untested against a trained U-Net.",
          "- E_cnn comes from an ImageNet-pretrained encoder, not a label-free trained U-Net; it is not used by the deployed model.",
          "- Not done: dev-15 annotation/detector tuning, flip/rotation near-duplicate check (possible residual leakage), external dataset, plain CNN classifier, PDF report."]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    OUT_JSON.write_text(json.dumps({"g3": {"a3": auc_a3, "gnn": auc_g, "include": bool(include_gnn)}, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()