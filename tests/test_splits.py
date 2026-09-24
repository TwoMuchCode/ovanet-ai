"""Tests for data splits — no group leaks across train/val/test."""

import pandas as pd
import pytest
from pathlib import Path

SPLITS_PATH = Path("data/processed/splits.csv")


@pytest.fixture
def splits():
    if not SPLITS_PATH.exists():
        pytest.skip("splits.csv not yet generated — run make_splits first")
    return pd.read_csv(SPLITS_PATH)


class TestSplitIntegrity:

    def test_no_group_leakage(self, splits):
        """Every group_id appears in exactly one split."""
        for gid in splits["group_id"].unique():
            group_splits = splits.loc[splits["group_id"] == gid, "split"].unique()
            assert len(group_splits) == 1, (
                f"LEAKAGE: group {gid} in {list(group_splits)}"
            )

    def test_both_classes_in_each_split(self, splits):
        """Train, val, and test each contain both infected and notinfected."""
        for sp in ["train", "val", "test"]:
            labels = set(splits.loc[splits["split"] == sp, "label"])
            assert "infected" in labels, f"'{sp}' has no infected"
            assert "notinfected" in labels, f"'{sp}' has no notinfected"

    def test_three_splits_exist(self, splits):
        assert set(splits["split"].unique()) == {"train", "val", "test"}

    def test_no_duplicates_in_splits(self, splits):
        """Each image path appears exactly once."""
        assert splits["path"].is_unique, "Duplicate paths in splits.csv"

    def test_only_readable_non_duplicate(self, splits):
        """splits.csv should only contain readable, non-duplicate images."""
        assert splits["readable"].all(), "Non-readable images in splits"
        assert not splits["is_duplicate"].any(), "Duplicate images in splits"

    def test_approximate_ratios(self, splits):
        """70/15/15 ± 8% tolerance.

        Group-aware splits can't be exact because entire groups (up to
        hundreds of images) must go to one split.  8% is realistic.
        """
        n = len(splits)
        for sp, expected in [("train", 0.70), ("val", 0.15), ("test", 0.15)]:
            actual = len(splits[splits["split"] == sp]) / n
            assert abs(actual - expected) < 0.08, (
                f"{sp}: expected ~{expected:.0%}, got {actual:.1%}"
            )
