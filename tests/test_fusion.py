# tests/test_fusion.py
import numpy as np
from ovanet.fusion.model import predict

def test_predict_returns_valid_fusion_result():
    fake_features = {f"feature_{i}": 0.0 for i in range(29)}
    try:
        result = predict(fake_features)
        assert 0.0 <= result.risk_score <= 100.0
        assert result.band in ("LOW", "MODERATE", "HIGH")
    except FileNotFoundError:
        import pytest
        pytest.skip("Run ovanet.fusion.train first to generate weights/fusion_v1.joblib")