"""
scripts/generate_pseudo_masks.py

Generates weak-supervision pseudo-masks for the dataset using the
tuned Day 4 classical follicle detector (ovanet.detector.detect_follicles).

Uses unambiguous filenames: <folder_split>_<label>_<stem>.npz to prevent
stem collision across classes/folders.

Produces:
  1. data/pseudo_masks/<folder_split>_<label>_<stem>.npz
  2. data/pseudo_masks/manifest_<split>.csv
  3. data/pseudo_masks_train_val.zip (ready for Colab upload)

Usage:
  python scripts/generate_pseudo_masks.py [--split train|val|all]
"""

import argparse
import sys
import zipfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ovanet.preprocessing import preprocess
from ovanet.detector import detect_follicles

RAW_ROOT = Path("data/raw/pcos_primary/data")
SPLITS_CSV = Path("data/processed/splits.csv")
OUTPUT_DIR = Path("data/pseudo_masks")
ZIP_OUTPUT = Path("data/pseudo_masks_train_val.zip")


def generate_pseudo_masks(splits: list[str], max_samples: int | None = None):
    print(f"=== Generating Pseudo-Masks for splits={splits} ===")
    if not SPLITS_CSV.exists():
        raise FileNotFoundError(f"Missing {SPLITS_CSV}")

    df = pd.read_csv(SPLITS_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_saved_npz = []

    for split in splits:
        subset = df[df["split"] == split].copy()
        if max_samples:
            subset = subset.head(max_samples)

        print(f"\nProcessing {split} split ({len(subset)} images)...")
        records = []

        for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"Generating {split} masks"):
            rel_path = Path(row["path"])
            full_path = RAW_ROOT / rel_path
            if not full_path.exists():
                continue

            raw_bgr = cv2.imread(str(full_path))
            if raw_bgr is None:
                continue

            prep = preprocess(raw_bgr)
            result = detect_follicles(prep.raw, enh=prep.enh)

            ovary_mask = result.ovary_mask.astype(np.uint8)
            follicle_mask = (result.follicle_labels > 0).astype(np.uint8)

            folder_split = row.get("folder_split", "data")
            label = row["label"]
            stem = rel_path.stem
            unique_name = f"{folder_split}_{label}_{stem}.npz"

            npz_file = OUTPUT_DIR / unique_name
            np.savez_compressed(
                npz_file,
                ovary=ovary_mask,
                follicles=follicle_mask,
            )
            all_saved_npz.append(npz_file)

            records.append({
                "stem": stem,
                "mask_file": unique_name,
                "path": str(rel_path),
                "label": label,
                "group_id": row["group_id"],
                "split": split,
                "n_follicles": len(result.follicles),
                "ovary_area_px": int(ovary_mask.sum()),
                "follicle_area_px": int(follicle_mask.sum()),
            })

        manifest_df = pd.DataFrame(records)
        manifest_csv = OUTPUT_DIR / f"manifest_{split}.csv"
        manifest_df.to_csv(manifest_csv, index=False)
        print(f"Saved manifest to {manifest_csv} ({len(manifest_df)} records)")
        print(f"  Mean detected follicles: {manifest_df['n_follicles'].mean():.2f}")
        print(f"  Mean ovary area px: {manifest_df['ovary_area_px'].mean():.1f}")

    # Package both splits into zip
    print(f"\nPackaging {len(all_saved_npz)} unique masks into {ZIP_OUTPUT}...")
    with zipfile.ZipFile(ZIP_OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for split in splits:
            m_csv = OUTPUT_DIR / f"manifest_{split}.csv"
            if m_csv.exists():
                zf.write(m_csv, arcname=m_csv.name)
        for f in all_saved_npz:
            zf.write(f, arcname=f"masks/{f.name}")

    zip_mb = ZIP_OUTPUT.stat().st_size / (1024 * 1024)
    print(f"Created {ZIP_OUTPUT} ({zip_mb:.2f} MB) with {len(all_saved_npz)} mask files. No collisions!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="all", choices=["train", "val", "all"])
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    splits = ["train", "val"] if args.split == "all" else [args.split]
    generate_pseudo_masks(splits=splits, max_samples=args.max_samples)
