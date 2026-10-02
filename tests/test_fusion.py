# tests/test_fusion.py
from ovanet.fusion.model import load_bundle, predict


def test_predict_real_bundle():
    b = load_bundle()
    r = predict({c: 0.0 for c in b["cols"]})
    assert 0.0 <= r.risk_score <= 100.0
    assert r.band in ("LOW", "MODERATE", "HIGH")
    assert abs(sum(r.family_attribution.values()) - 1.0) < 1e-6
    assert r.top_features and r.top_features[0].name != "placeholder"