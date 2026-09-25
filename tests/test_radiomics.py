"""Tests for ovanet.radiomics (contract C3)."""

import numpy as np
import pytest

from ovanet.contracts import RADIOMICS_FEATURE_ORDER, validate_radiomics
from ovanet.radiomics import extract


class TestRadiomicsContract:
    """Output matches the frozen C3 contract."""

    def test_output_keys(self):
        img = np.random.rand(256, 256).astype(np.float32)
        r = extract(img)
        vec = r.as_vector()
        assert vec.shape == (5,)
        assert list(RADIOMICS_FEATURE_ORDER) == [
            "mean", "std", "entropy", "glcm_contrast", "glcm_homogeneity"
        ]

    def test_validates(self):
        img = np.random.rand(256, 256).astype(np.float32)
        r = extract(img)
        validate_radiomics(r)  # must not raise


class TestKnownValues:
    """Sanity checks on synthetic images with known properties."""

    def test_constant_image(self):
        """Constant image: entropy 0, contrast 0, homogeneity 1."""
        img = np.full((256, 256), 0.5, dtype=np.float32)
        r = extract(img)
        assert abs(r.entropy) < 1e-6, f"entropy should be 0, got {r.entropy}"
        assert abs(r.glcm_contrast) < 1e-6, f"contrast should be 0, got {r.glcm_contrast}"
        assert abs(r.glcm_homogeneity - 1.0) < 1e-6, f"homogeneity should be 1, got {r.glcm_homogeneity}"
        assert abs(r.std) < 1e-6

    def test_noise_image_high_entropy(self):
        """Random noise: high entropy, moderate contrast."""
        rng = np.random.RandomState(42)
        img = rng.rand(256, 256).astype(np.float32)
        r = extract(img)
        # 32 grey levels → max entropy = log2(32) = 5.0
        assert r.entropy > 3.0, f"entropy too low for noise: {r.entropy}"
        assert r.glcm_contrast > 0, "contrast should be > 0 for noise"

    def test_mean_value(self):
        """Mean of a uniform 0.5 image is 0.5."""
        img = np.full((256, 256), 0.5, dtype=np.float32)
        r = extract(img)
        assert abs(r.mean - 0.5) < 0.01

    def test_dark_image(self):
        """Dark image has low mean."""
        img = np.full((256, 256), 0.05, dtype=np.float32)
        r = extract(img)
        assert r.mean < 0.1

    def test_bright_image(self):
        """Bright image has high mean."""
        img = np.full((256, 256), 0.95, dtype=np.float32)
        r = extract(img)
        assert r.mean > 0.9


class TestMaskedExtraction:
    """With a mask, only ROI pixels are used."""

    def test_mask_changes_mean(self):
        """A mask that selects only the bright half changes the mean."""
        img = np.zeros((256, 256), dtype=np.float32)
        img[:, 128:] = 0.8  # right half is bright

        mask_full = np.ones((256, 256), dtype=np.uint8)
        mask_right = np.zeros((256, 256), dtype=np.uint8)
        mask_right[:, 128:] = 1

        r_full = extract(img, mask=mask_full)
        r_right = extract(img, mask=mask_right)

        assert abs(r_full.mean - 0.4) < 0.05, f"full mean {r_full.mean}"
        assert abs(r_right.mean - 0.8) < 0.05, f"right mean {r_right.mean}"

    def test_empty_mask_returns_zeros(self):
        """All-zero mask → degenerate result (zeros)."""
        img = np.random.rand(256, 256).astype(np.float32)
        mask = np.zeros((256, 256), dtype=np.uint8)
        r = extract(img, mask=mask)
        assert r.mean == 0.0
        assert r.entropy == 0.0


class TestNoNaN:
    """No feature should ever be NaN."""

    def test_random_images(self):
        rng = np.random.RandomState(0)
        for _ in range(5):
            img = rng.rand(256, 256).astype(np.float32)
            r = extract(img)
            vec = r.as_vector()
            assert not np.isnan(vec).any(), f"NaN in {vec}"
