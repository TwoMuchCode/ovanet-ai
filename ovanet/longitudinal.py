"""
ovanet/longitudinal.py

Real C7 implementation: rule-based comparison between two analysis results.
Rule S2: if the two images aren't genuinely the same patient, caller must set is_demo_pair=True.
"""

from ovanet.contracts import LongitudinalResult

TREND_MARGIN = 5.0  # risk-score points; change smaller than this = STABLE


def compare(visit_a_result, visit_b_result, is_demo_pair: bool = True) -> LongitudinalResult:
    score_a = visit_a_result.fusion.risk_score
    score_b = visit_b_result.fusion.risk_score
    delta_score = score_b - score_a

    follicles_a = len(visit_a_result.vision.follicles)
    follicles_b = len(visit_b_result.vision.follicles)
    delta_follicles = follicles_b - follicles_a

    if abs(delta_score) < TREND_MARGIN:
        trend = "STABLE"
    elif delta_score > 0:
        trend = "WORSENING"
    else:
        trend = "IMPROVING"

    pair_note = ("DEMO PAIR -- these are two unrelated sample images, not a real patient's visits. "
                 if is_demo_pair else "")

    report_text = (
        f"{pair_note}Risk score changed from {score_a:.0f} to {score_b:.0f} "
        f"({delta_score:+.0f} points). Follicle count changed from {follicles_a} to {follicles_b} "
        f"({delta_follicles:+d}). Trend classified as {trend} based on a {TREND_MARGIN}-point margin. "
        f"This is a heuristic comparison, not a validated longitudinal clinical assessment."
    )

    return LongitudinalResult(
        trend=trend,
        deltas={"risk_score": delta_score, "follicle_count": delta_follicles},
        report_text=report_text,
        is_demo_pair=is_demo_pair,
    )