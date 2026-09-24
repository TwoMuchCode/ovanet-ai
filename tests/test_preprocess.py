"""Tests for ovanet.preprocessing (contract C1)."""

import numpy as np
import pytest

from ovanet.contracts import IMAGE_SIZE, validate_preprocess
from ovanet.preprocessing import preprocess


class TestPreprocessContract:
    """Every test checks the output against the frozen C1 contract."""

    def test_grayscale_input(self):
        img = np.random.randint(30, 220, (300, 400), dtype=np.uint8)
        result = preprocess(img)
        validate_preprocess(result)

    def test_rgb_input(self):
        img = np.random.randint(30, 220, (200, 300, 3), dtype=np.uint8)
        result = preprocess(img)
        validate_preprocess(result)

    def test_rgba_input(self):
        img = np.random.randint(30, 220, (200, 300, 4), dtype=np.uint8)
        result = preprocess(img)
        validate_preprocess(result)

    def test_square_input(self):
        img = np.random.randint(30, 220, (256, 256), dtype=np.uint8)
        result = preprocess(img)
        validate_preprocess(result)

    def test_small_input(self):
        img = np.random.randint(30, 220, (32, 32), dtype=np.uint8)
        result = preprocess(img)
        validate_preprocess(result)

    def test_output_shape(self):
        img = np.random.randint(0, 256, (500, 300), dtype=np.uint8)
        result = preprocess(img)
        assert result.raw.shape == (IMAGE_SIZE, IMAGE_SIZE)
        assert result.enh.shape == (IMAGE_SIZE, IMAGE_SIZE)

    def test_output_dtype(self):
        img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        result = preprocess(img)
        assert result.raw.dtype == np.float32
        assert result.enh.dtype == np.float32

    def test_output_range(self):
        img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        result = preprocess(img)
        assert 0.0 <= float(result.raw.min())
        assert float(result.raw.max()) <= 1.0
        assert 0.0 <= float(result.enh.min())
        assert float(result.enh.max()) <= 1.0


class TestQualityCheck:
    """Quality gate must flag bad images but not reject normal ones."""

    def test_blank_image_rejected(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        result = preprocess(img)
        assert not result.quality.ok
        assert any("blank" in r or "dark" in r for r in result.quality.reasons)

    def test_dark_image_rejected(self):
        img = np.full((100, 100), 2, dtype=np.uint8)
        result = preprocess(img)
        assert not result.quality.ok

    def test_normal_image_ok(self):
        rng = np.random.RandomState(0)
        img = rng.randint(50, 200, (256, 256), dtype=np.uint8)
        result = preprocess(img)
        assert result.quality.ok
        assert result.quality.blur_var > 10.0

    def test_blur_var_is_float(self):
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = preprocess(img)
        assert isinstance(result.quality.blur_var, float)


class TestRawEnhDiffer:
    """ENH applies CLAHE + denoise, so it must not equal RAW."""

    def test_raw_and_enh_differ(self):
        rng = np.random.RandomState(1)
        img = rng.randint(20, 200, (256, 256), dtype=np.uint8)
        result = preprocess(img)
        assert not np.array_equal(result.raw, result.enh)
