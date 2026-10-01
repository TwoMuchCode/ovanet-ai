# tests/test_pipeline.py
import numpy as np
from PIL import Image
from ovanet.pipeline import analyze
import os

def test_pipeline_on_real_sample():
    sample_path = "data/raw/pcos_primary/data/train/infected/img_0_9172.jpg"
    if not os.path.exists(sample_path):
        import pytest
        pytest.skip("Sample image not found locally")
    img = np.array(Image.open(sample_path).convert("L"))
    result = analyze(img)
    assert result.is_mock is False
    assert 0.0 <= result.fusion.risk_score <= 100.0