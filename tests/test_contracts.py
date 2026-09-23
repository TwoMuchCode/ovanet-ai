"""
tests/test_contracts.py

Validates every mock object against its contract shape. Also checks that a
deliberately broken shape is correctly rejected -- this is the safety net that
catches a contract violation the moment any real module returns the wrong shape.
"""

import numpy as np
import pytest

from ovanet.contracts import (
    validate_preprocess, validate_vision, validate_radiomics,
    validate_graph, validate_fusion, validate_analysis,
    IMAGE_SIZE, PreprocessResult, QualityCheck,
)
from ovanet.mocks import (
    mock_preprocess, mock_vision, mock_radiomics, mock_graph, mock_fusion, mock_analyze,
)


def test_mock_preprocess_valid():
    validate_preprocess(mock_preprocess())


def test_mock_vision_valid():
    validate_vision(mock_vision())


def test_mock_radiomics_valid():
    validate_radiomics(mock_radiomics())


def test_mock_graph_valid():
    validate_graph(mock_graph(n=6))


def test_mock_graph_handles_small_n():
    validate_graph(mock_graph(n=2))  # n<3 must still be a valid shape


def test_mock_fusion_valid():
    validate_fusion(mock_fusion())


def test_mock_analyze_end_to_end():
    result = mock_analyze()
    validate_analysis(result)
    assert result.is_mock is True


def test_broken_preprocess_shape_is_rejected():
    bad = PreprocessResult(
        raw=np.zeros((100, 100), dtype=np.float32),  # wrong size on purpose
        enh=np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32),
        quality=QualityCheck(ok=True, reasons=[], blur_var=1.0),
    )
    with pytest.raises(AssertionError):
        validate_preprocess(bad)


def test_broken_vision_embedding_dim_is_rejected():
    v = mock_vision()
    v.e_cnn = np.zeros(10, dtype=np.float32)  # wrong dim on purpose
    with pytest.raises(AssertionError):
        validate_vision(v)