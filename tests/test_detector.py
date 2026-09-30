"""
Tests for ovanet.detector (Contract C2).
"""

import numpy as np
import pytest

from ovanet.contracts import (
    IMAGE_SIZE,
    E_CNN_DIM,
    VisionResult,
    validate_vision,
)
from ovanet.detector import detect_follicles, estimate_ovary_mask


def test_estimate_ovary_mask():
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    # Bright sector
    img[40:220, 40:220] = 0.5
    mask = estimate_ovary_mask(img)
    assert mask.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert mask.dtype == np.uint8
    assert mask.max() == 1
    assert mask.min() == 0


def test_detect_follicles_synthetic_dots():
    """Create a synthetic ovary with 5 dark circular follicles."""
    raw = np.full((IMAGE_SIZE, IMAGE_SIZE), 0.5, dtype=np.float32)
    # Add dark circles (follicles)
    import cv2
    centers = [(100, 100), (150, 110), (120, 160), (90, 150), (160, 150)]
    for cx, cy in centers:
        cv2.circle(raw, (cx, cy), 8, 0.1, -1)

    result = detect_follicles(raw)
    validate_vision(result)

    assert result.seg_method == "classical"
    assert result.e_cnn.shape == (E_CNN_DIM,)
    assert result.ovary_mask.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert result.follicle_labels.shape == (IMAGE_SIZE, IMAGE_SIZE)
    # At least some follicles should be detected
    assert len(result.follicles) > 0
    for f in result.follicles:
        assert f.area_px > 0
        assert f.eq_radius_px > 0
        assert 0.0 <= f.solidity <= 1.0


def test_detect_follicles_blank_image():
    """Blank image should return 0 follicles and warning without crashing."""
    raw = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    result = detect_follicles(raw)
    validate_vision(result)
    assert len(result.follicles) == 0
    assert len(result.warnings) > 0


def test_detect_follicles_random_noise():
    """Random noise shouldn't crash or create NaN."""
    rng = np.random.RandomState(42)
    raw = rng.uniform(0.1, 0.9, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    result = detect_follicles(raw)
    validate_vision(result)
    assert not np.isnan(result.e_cnn).any()


def test_detect_follicles_custom_mask():
    raw = np.full((IMAGE_SIZE, IMAGE_SIZE), 0.5, dtype=np.float32)
    custom_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    custom_mask[80:180, 80:180] = 1

    result = detect_follicles(raw, ovary_mask=custom_mask)
    validate_vision(result)
    assert np.array_equal(result.ovary_mask, custom_mask)
