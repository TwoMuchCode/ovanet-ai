"""
tests/test_vision.py

Unit tests for Contract C2 vision inference (ovanet.vision.infer).
"""

import numpy as np
import pytest

from ovanet.contracts import IMAGE_SIZE, E_CNN_DIM, VisionResult, PreprocessResult, QualityCheck
from ovanet.preprocessing import preprocess
from ovanet.vision.infer import analyze_vision, extract_ecnn


def test_analyze_vision_conforms_to_contract_c2():
    """Verify analyze_vision output shapes, types, and values match Contract C2."""
    dummy_bgr = (np.random.rand(IMAGE_SIZE, IMAGE_SIZE, 3) * 255).astype(np.uint8)
    prep = preprocess(dummy_bgr)

    res = analyze_vision(prep)

    assert isinstance(res, VisionResult)
    assert res.ovary_mask.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert res.ovary_mask.dtype == np.uint8
    assert res.follicle_labels.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert res.follicle_labels.dtype in (np.int32, np.int64)
    assert isinstance(res.follicles, list)
    assert res.e_cnn.shape == (E_CNN_DIM,)
    assert res.e_cnn.dtype == np.float32
    assert res.seg_method == "classical"
    assert isinstance(res.warnings, list)


def test_extract_ecnn_dimensions():
    """Verify E_cnn extracted from raw image is strictly 512-dim float32."""
    raw = np.random.uniform(0.0, 1.0, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    e_cnn = extract_ecnn(raw)

    assert e_cnn.shape == (E_CNN_DIM,)
    assert e_cnn.dtype == np.float32
    assert not np.isnan(e_cnn).any()


def test_analyze_vision_blank_image():
    """Verify graceful handling of blank/dark inputs."""
    prep = PreprocessResult(
        raw=np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32),
        enh=np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32),
        quality=QualityCheck(ok=False, reasons=["blank_image"], blur_var=0.0),
    )

    res = analyze_vision(prep)
    assert isinstance(res, VisionResult)
    assert res.ovary_mask.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert res.e_cnn.shape == (E_CNN_DIM,)
