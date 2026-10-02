# tests/test_pipeline.py
import numpy as np
from ovanet.pipeline import analyze


def test_pipeline_on_synthetic_image():
    rng = np.random.RandomState(0)
    img = (rng.rand(300, 300) * 60 + 80).astype(np.uint8)
    yy, xx = np.ogrid[:300, :300]
    img[((yy - 150) ** 2 / 110 ** 2 + (xx - 150) ** 2 / 80 ** 2) < 1] += 40
    r = analyze(img)
    assert r.is_mock is False
    assert 0.0 <= r.fusion.risk_score <= 100.0
    assert any("shortcut" in w.lower() for w in r.warnings)