"""
app/streamlit_app.py

OvaNet AI -- Streamlit app, v0 (Day 2). Runs entirely on mock data
(ovanet.mocks.mock_analyze) until real modules land. Swap in
ovanet.pipeline.analyze() as real pieces arrive (Day 3+).
"""

import streamlit as st
from PIL import Image
import numpy as np
from ovanet.preprocessing import preprocess as real_preprocess
from ovanet.mocks import mock_analyze, mock_longitudinal
from ovanet.contracts import GRAPH_STAT_NAMES

st.set_page_config(page_title="OvaNet AI", layout="wide")

st.warning(
    "⚕️ **Decision-support research prototype. Not a diagnosis.** "
    "OvaNet estimates PCOS risk from ultrasound for clinician review only. "
    "It is not clinically validated and is not a medical device."
)

st.title("OvaNet AI -- PCOS Risk Assessment")

uploaded = st.file_uploader("Upload an ovarian ultrasound image", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    pil_img = Image.open(uploaded).convert("L")
    img_array = np.array(pil_img)

    real_prep = real_preprocess(img_array)

    st.success("✅ Real preprocessing (C1) is now active.")
    st.image(real_prep.raw, caption="Real preprocessed image (raw)", clamp=True)
    if not real_prep.quality.ok:
        st.warning(f"Quality check failed: {real_prep.quality.reasons}")

    result = mock_analyze(image=uploaded)  # rest of the pipeline still mock

    if result.is_mock:
        st.error("🧪 MOCK DATA below this point — real radiomics/vision/fusion not wired in yet.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Segmentation overlay")
        st.image(result.vision.ovary_mask * 255, caption="Ovary mask (mock)", clamp=True)
        st.caption(f"Seg method: {result.vision.seg_method} | "
                   f"{len(result.vision.follicles)} follicles detected")
        for w in result.vision.warnings:
            st.caption(f"⚠️ {w}")

    with col2:
        st.subheader("Follicle graph")
        st.write(f"n = {result.graph.n} nodes")
        stats_display = {k: round(result.graph.stats.get(k, float("nan")), 3) for k in GRAPH_STAT_NAMES}
        st.json(stats_display)

    st.subheader("Radiomics (F_k)")
    st.json({
        "mean": round(result.radiomics.mean, 4),
        "std": round(result.radiomics.std, 4),
        "entropy": round(result.radiomics.entropy, 4),
        "glcm_contrast": round(result.radiomics.glcm_contrast, 4),
        "glcm_homogeneity": round(result.radiomics.glcm_homogeneity, 4),
    })

    st.subheader("Risk assessment")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Risk score", f"{result.fusion.risk_score:.1f} / 100")
        st.metric("Band", result.fusion.band)
        st.caption(f"Baseline vote (paper method): "
                   f"{result.fusion.baseline_vote.votes_pcos}/{result.fusion.baseline_vote.of}")
    with c2:
        st.write("Feature-family attribution")
        st.bar_chart(result.fusion.family_attribution)

    st.subheader("Two-visit comparison (demo)")
    longi = mock_longitudinal()
    if longi.is_demo_pair:
        st.info("🧪 DEMO PAIR -- two unrelated sample images, not a real patient.")
    st.write(f"Trend: **{longi.trend}**")
    st.caption(longi.report_text)

else:
    st.info("Upload an image to see the (currently mock) analysis flow.")