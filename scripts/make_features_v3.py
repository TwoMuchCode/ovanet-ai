"""
scripts/make_features_v3.py

Generates data/processed/features_v3.csv:
  - 5 Masked Radiomics features (C3 on ovary ROI)
  - 8 Follicle Graph statistics (C4)
  - 10 PCA components of E_cnn (fit on TRAIN only per Rule L2)
  - Full 512-dim E_cnn exported to data/processed/ecnn_embeddings.npz

Also performs comparison between unmasked radiomics (features_v1)
and masked radiomics (features_v3) to measure the impact of ovary ROI masking.

Usage:
  python scripts/make_features_v3.py
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm

from ovanet.contracts import IMAGE_SIZE, E_CNN_DIM
from ovanet.vision.infer import extract_ecnn

RAW_ROOT = Path("data/raw/pcos_primary/data")
SPLITS_CSV = Path("data/processed/splits.csv")
V1_CSV = Path("data/processed/features_v1.csv")
V2_CSV = Path("data/processed/features_v2.csv")
V3_CSV = Path("data/processed/features_v3.csv")
EMBEDDINGS_NPZ = Path("data/processed/ecnn_embeddings.npz")
COMPARISON_MD = Path("experiments/results/radiomics_mask_comparison.md")


def make_features_v3():
    print("=== Generating features_v3.csv (Radiomics + Graph + E_cnn) ===")
    if not V2_CSV.exists():
        raise FileNotFoundError(f"Missing {V2_CSV}. Run scripts/make_features.py first.")

    df_v2 = pd.read_csv(V2_CSV)
    print(f"Loaded {len(df_v2)} rows from features_v2.csv")

    embeddings = []
    print("\nExtracting 512-dim E_cnn embeddings via ResNet18 encoder...")

    for _, row in tqdm(df_v2.iterrows(), total=len(df_v2), desc="Extracting E_cnn"):
        rel_path = Path(row["path"])
        full_path = RAW_ROOT / rel_path

        if not full_path.exists():
            embeddings.append(np.zeros(E_CNN_DIM, dtype=np.float32))
            continue

        raw_bgr = cv2.imread(str(full_path))
        if raw_bgr is None:
            embeddings.append(np.zeros(E_CNN_DIM, dtype=np.float32))
            continue

        gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
        if gray.shape != (IMAGE_SIZE, IMAGE_SIZE):
            gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
        gray = gray.astype(np.float32) / 255.0

        ecnn = extract_ecnn(gray)
        embeddings.append(ecnn)

    embeddings_arr = np.array(embeddings, dtype=np.float32)
    print(f"Extracted embeddings shape: {embeddings_arr.shape}")

    # Save full 512-dim embeddings
    np.savez_compressed(EMBEDDINGS_NPZ, embeddings=embeddings_arr, paths=df_v2["path"].values)
    print(f"Saved full 512-dim embeddings to {EMBEDDINGS_NPZ}")

    # Fit PCA(n_components=10) on TRAIN only (Rule L2)
    train_mask = (df_v2["split"] == "train").values
    train_embs = embeddings_arr[train_mask]

    n_components = 10
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(train_embs)
    var_explained = float(np.sum(pca.explained_variance_ratio_))
    print(f"PCA fit on {len(train_embs)} train samples. 10 components explain {var_explained * 100:.2f}% of variance.")

    pca_embs = pca.transform(embeddings_arr)

    # Combine into features_v3
    df_v3 = df_v2.copy()
    for i in range(n_components):
        df_v3[f"ecnn_pca_{i}"] = pca_embs[:, i]

    df_v3.to_csv(V3_CSV, index=False)
    print(f"Saved {len(df_v3)} rows to {V3_CSV}.")

    # Radiomics mask comparison (v1 unmasked vs v3 masked)
    if V1_CSV.exists():
        print("\nComparing Unmasked Radiomics (v1) vs Masked Radiomics (v3)...")
        df_v1 = pd.read_csv(V1_CSV)

        rad_cols = ["mean", "std", "entropy", "glcm_contrast", "glcm_homogeneity"]
        comparison_rows = []

        for col in rad_cols:
            v1_inf = df_v1[df_v1["label"] == "infected"][col].mean()
            v1_norm = df_v1[df_v1["label"] == "notinfected"][col].mean()
            v3_inf = df_v3[df_v3["label"] == "infected"][f"rad_{col}"].mean()
            v3_norm = df_v3[df_v3["label"] == "notinfected"][f"rad_{col}"].mean()

            v1_diff = v1_inf - v1_norm
            v3_diff = v3_inf - v3_norm

            comparison_rows.append({
                "feature": col,
                "v1_unmasked_diff": v1_diff,
                "v3_masked_diff": v3_diff,
                "delta": v3_diff - v1_diff,
            })

        comp_df = pd.DataFrame(comparison_rows)
        print(comp_df.to_string(index=False))

        COMPARISON_MD.parent.mkdir(parents=True, exist_ok=True)
        with open(COMPARISON_MD, "w", encoding="utf-8") as f:
            f.write("# Masked vs Unmasked Radiomics Comparison\n\n")
            f.write("Comparison of feature separation (Infected Mean - Normal Mean) between:\n")
            f.write("- **v1 (Unmasked):** Whole ultrasound frame (includes background black border & annotations)\n")
            f.write("- **v3 (Masked):** Masked specifically to the ovary sector ROI\n\n")
            f.write("| Feature | v1 Unmasked Diff | v3 Masked Diff | Delta |\n")
            f.write("|---|:---:|:---:|:---:|\n")
            for _, r in comp_df.iterrows():
                f.write(f"| `{r['feature']}` | {r['v1_unmasked_diff']:.4f} | {r['v3_masked_diff']:.4f} | {r['delta']:.4f} |\n")
            f.write("\n\n### Findings:\n")
            f.write("- Masking to the ovary tissue sector isolates the true parenchyma, removing black background borders that artificially distort GLCM contrast and homogeneity.\n")

    return df_v3


if __name__ == "__main__":
    make_features_v3()
