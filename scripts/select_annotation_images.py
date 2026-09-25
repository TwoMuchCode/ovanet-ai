"""
Select images for annotation.

Picks 15 dev images (from VAL split, for Madhumita) and
15 gold-test images (from TEST split, for Sahina).

Stratified by class, one per group (for diversity).

Usage:
    python scripts/select_annotation_images.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np


def select_images(df: pd.DataFrame, n: int = 15, seed: int = 42) -> pd.DataFrame:
    """Pick n images, stratified by label, one per group for diversity."""
    rng = np.random.RandomState(seed)

    n_infected = n // 2 + (n % 2)  # 8 if n=15
    n_notinfected = n // 2          # 7 if n=15

    selected = []
    for label, count in [("infected", n_infected), ("notinfected", n_notinfected)]:
        subset = df[df["label"] == label].copy()
        # One image per group (keep first, shuffle for randomness)
        shuffled = subset.sample(frac=1, random_state=rng)
        deduped = shuffled.drop_duplicates(subset="group_id", keep="first")
        if len(deduped) >= count:
            chosen = deduped.head(count)
        else:
            chosen = deduped
        selected.append(chosen)

    return pd.concat(selected, ignore_index=True)


def main():
    splits = pd.read_csv("data/processed/splits.csv")

    print("=" * 50)
    print("  ANNOTATION IMAGE SELECTION")
    print("=" * 50)

    # Dev-15: from VAL split (Madhumita annotates, tunes classical detector)
    val = splits[splits["split"] == "val"]
    dev15 = select_images(val, n=15, seed=42)
    print(f"\nDev-15 (VAL split → Madhumita annotates):")
    print(f"  infected: {(dev15['label'] == 'infected').sum()}")
    print(f"  notinfected: {(dev15['label'] == 'notinfected').sum()}")
    print(f"  groups: {dev15['group_id'].nunique()}")
    print("\nPaths:")
    for _, row in dev15.iterrows():
        print(f"  [{row['label'][:3]}] {row['path']}")

    # Gold-test-15: from TEST split (Sahina annotates, used for final Dice)
    test = splits[splits["split"] == "test"]
    gold15 = select_images(test, n=15, seed=99)  # different seed for different selection
    print(f"\nGold-test-15 (TEST split → Sahina annotates):")
    print(f"  infected: {(gold15['label'] == 'infected').sum()}")
    print(f"  notinfected: {(gold15['label'] == 'notinfected').sum()}")
    print(f"  groups: {gold15['group_id'].nunique()}")
    print("\nPaths:")
    for _, row in gold15.iterrows():
        print(f"  [{row['label'][:3]}] {row['path']}")

    # Save lists
    out = Path("data/annotations")
    out.mkdir(parents=True, exist_ok=True)
    dev15[["path", "label", "group_id"]].to_csv(out / "dev15_list.csv", index=False)
    gold15[["path", "label", "group_id"]].to_csv(out / "gold_test15_list.csv", index=False)
    print(f"\nSaved → {out / 'dev15_list.csv'}")
    print(f"Saved → {out / 'gold_test15_list.csv'}")
    print("\nSend gold_test15_list.csv to Sahina for her annotations.")


if __name__ == "__main__":
    main()
