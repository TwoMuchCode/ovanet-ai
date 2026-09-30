"""
scripts/make_features.py

Extracts Contract C3 (radiomics on ovary ROI) and Contract C4 (follicle graph stats)
for all images in splits.csv and saves data/processed/features_v2.csv.

Feature vector per image:
  - 5 radiomics features: [mean, std, entropy, glcm_contrast, glcm_homogeneity]
  - 8 graph statistics: [n_follicles, mean_nn_dist_rel, std_nn_dist_rel,
                         mean_degree, clustering_coeff, peripheral_index,
                         size_cv, follicle_area_fraction]

Usage:
  python scripts/make_features.py [--output data/processed/features_v2.csv]
"""

import argparse
import sys
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ovanet.preprocessing import preprocess
from ovanet.detector import detect_follicles
from ovanet.radiomics import extract as extract_radiomics
from ovanet.graph import build_graph
from ovanet.contracts import (
    RADIOMICS_FEATURE_ORDER,
    GRAPH_STAT_NAMES,
)

RAW_ROOT = Path("data/raw/pcos_primary/data")
SPLITS_CSV = Path("data/processed/splits.csv")
DEFAULT_OUT = Path("data/processed/features_v2.csv")


def make_features_v2(output_csv: Path = DEFAULT_OUT, max_samples: int | None = None):
    print("=== Computing features_v2 (Masked Radiomics + Graph Stats) ===")
    if not SPLITS_CSV.exists():
        raise FileNotFoundError(f"Missing {SPLITS_CSV}")

    df = pd.read_csv(SPLITS_CSV)
    if max_samples:
        df = df.head(max_samples)

    print(f"Processing {len(df)} images...")
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features_v2"):
        rel_path = Path(row["path"])
        full_path = RAW_ROOT / rel_path
        if not full_path.exists():
            continue

        raw_bgr = cv2.imread(str(full_path))
        if raw_bgr is None:
            continue

        # 1. Preprocessing (C1)
        prep = preprocess(raw_bgr)

        # 2. Vision Detector (C2)
        vision = detect_follicles(prep.raw, enh=prep.enh)

        # 3. Masked Radiomics (C3 on ovary ROI)
        rad = extract_radiomics(prep.raw, mask=vision.ovary_mask)

        # 4. Follicle Graph & Stats (C4)
        g = build_graph(vision.follicles, vision.ovary_mask)

        rec = {
            "path": str(rel_path),
            "label": row["label"],
            "group_id": row["group_id"],
            "split": row["split"],
        }

        # Add 5 radiomics
        for name in RADIOMICS_FEATURE_ORDER:
            rec[f"rad_{name}"] = getattr(rad, name)

        # Add 8 graph stats
        for name in GRAPH_STAT_NAMES:
            rec[f"graph_{name}"] = g.stats[name]

        records.append(rec)

    out_df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(out_df)} rows to {output_csv}.")
    print("Sample feature values (mean by label):")
    print(out_df.groupby("label")[["rad_mean", "rad_entropy", "graph_n_follicles", "graph_peripheral_index"]].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    make_features_v2(Path(args.output), max_samples=args.max_samples)
