"""
Day 2: Group-aware stratified split.

Reads manifest.csv (from audit.py), keeps only unique readable images,
splits by near-duplicate group so no group leaks across train/val/test.

Uses greedy bin-packing to balance IMAGE counts (not just group counts),
because group sizes vary widely (one group has 491 images).

Output: data/processed/splits.csv

Usage:
    python -m ovanet.data.make_splits [MANIFEST] [OUTPUT]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make_splits(
    manifest_path: str = "data/processed/manifest.csv",
    output_path: str = "data/processed/splits.csv",
    seed: int = 42,
    ratios: tuple = (0.70, 0.15, 0.15),
) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)

    # Keep only unique, readable images
    unique = df[(df["readable"]) & (~df["is_duplicate"])].copy()
    print(f"Unique readable images: {len(unique)}")

    # Image count per group
    group_sizes = unique.groupby("group_id").size()

    # Resolve each group's label (majority vote; warn if mixed)
    group_info = (
        unique.groupby("group_id")
        .agg(
            label=("label", lambda x: x.mode().iloc[0]),
            count=("label", "count"),
            n_labels=("label", "nunique"),
        )
        .reset_index()
    )
    mixed = group_info[group_info["n_labels"] > 1]
    if len(mixed) > 0:
        print(f"WARNING: {len(mixed)} groups have mixed labels — using majority.")
        for _, row in mixed.iterrows():
            imgs = unique[unique["group_id"] == row["group_id"]]
            print(f"  group {row['group_id']}: {dict(imgs['label'].value_counts())}")

    # Report large groups
    large = group_sizes[group_sizes > 20].sort_values(ascending=False)
    if len(large) > 0:
        print(f"\nLarge groups (>20 images): {len(large)}")
        for gid, sz in large.items():
            print(f"  group {gid}: {sz} images")

    rng = np.random.RandomState(seed)
    split_names = ["train", "val", "test"]
    group_to_split = {}

    for label in sorted(group_info["label"].unique()):
        gids = group_info.loc[group_info["label"] == label, "group_id"].values.copy()

        # Shuffle for randomness (seed-controlled)
        rng.shuffle(gids)

        # Sort by group size descending (largest first → greedy packing works best).
        # sorted() is stable, so same-size groups keep their shuffled order.
        gids_sorted = sorted(gids, key=lambda g: -int(group_sizes.get(g, 0)))

        # Target IMAGE counts per split for this label
        n_label_images = sum(int(group_sizes.get(g, 0)) for g in gids_sorted)
        targets = {s: r * n_label_images for s, r in zip(split_names, ratios)}
        current = {s: 0 for s in split_names}

        # Greedy: assign each group to the split that's most below its target
        for gid in gids_sorted:
            sz = int(group_sizes.get(gid, 0))
            best = min(split_names, key=lambda s: current[s] - targets[s])
            group_to_split[gid] = best
            current[best] += sz

    unique["split"] = unique["group_id"].map(group_to_split)

    # ── Verification ────────────────────────────────────────────────────
    # 1. No group leaks across splits
    for gid in unique["group_id"].unique():
        splits_in_group = unique.loc[unique["group_id"] == gid, "split"].unique()
        assert len(splits_in_group) == 1, (
            f"LEAKAGE: group {gid} in splits {splits_in_group}"
        )

    # 2. Both classes present in every split
    for sp in split_names:
        labels = unique.loc[unique["split"] == sp, "label"].unique()
        assert len(labels) >= 2, f"Split '{sp}' missing a class: {labels}"

    # ── Save ────────────────────────────────────────────────────────────
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    unique.to_csv(out, index=False, encoding="utf-8")

    # ── Summary ─────────────────────────────────────────────────────────
    n = len(unique)
    print(f"\nSplit summary  (seed={seed}, ratios={ratios}):")
    print(f"{'split':<8} {'images':>7} {'%':>6} {'infected':>9} {'notinfected':>12} {'groups':>7}")
    for sp in split_names:
        sub = unique[unique["split"] == sp]
        inf = int((sub["label"] == "infected").sum())
        notinf = int((sub["label"] == "notinfected").sum())
        grps = int(sub["group_id"].nunique())
        pct = f"{len(sub)/n*100:.1f}%"
        print(f"{sp:<8} {len(sub):>7} {pct:>6} {inf:>9} {notinf:>12} {grps:>7}")

    print(f"{'TOTAL':<8} {n:>7} {'100%':>6} "
          f"{int((unique['label']=='infected').sum()):>9} "
          f"{int((unique['label']=='notinfected').sum()):>12} "
          f"{int(unique['group_id'].nunique()):>7}")

    print(f"\nSaved → {out}")
    print("No-leakage assertion: PASSED")
    return unique


if __name__ == "__main__":
    manifest = sys.argv[1] if len(sys.argv) > 1 else "data/processed/manifest.csv"
    output = sys.argv[2] if len(sys.argv) > 2 else "data/processed/splits.csv"
    make_splits(manifest, output)
