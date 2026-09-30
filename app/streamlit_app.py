"""
app/streamlit_app.py

OvaNet AI -- Streamlit app.
Day 3: real C1 (preprocess) and C3 (radiomics) wired in.
Vision (C2), graph (C4), and fusion (C6) are still mock until those land.
"""

from PIL import Image
import numpy as np
import streamlit as st

from ovanet.preprocessing import preprocess as real_preprocess
from ovanet.radiomics import extract as real_extract
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
    # ---- REAL C1: preprocessing ----
    pil_img = Image.open(uploaded).convert("L")
    img_array = np.array(pil_img)
    real_prep = real_preprocess(img_array)

    st.success("✅ Real preprocessing (C1) active.")
    st.image(real_prep.raw, caption="Real preprocessed image (raw)", clamp=True)
    if not real_prep.quality.ok:
        st.warning(f"Quality check failed: {real_prep.quality.reasons}")

    # ---- REAL C3: radiomics ----
    real_radiomics = real_extract(real_prep.raw)
    st.success("✅ Real radiomics (C3) active.")
    st.subheader("Radiomics (F_k) -- REAL")
    st.json({
        "mean": round(real_radiomics.mean, 4),
        "std": round(real_radiomics.std, 4),
        "entropy": round(real_radiomics.entropy, 4),
        "glcm_contrast": round(real_radiomics.glcm_contrast, 4),
        "glcm_homogeneity": round(real_radiomics.glcm_homogeneity, 4),
    })

    # ---- MOCK: everything else (vision, graph, fusion) until those land ----
    result = mock_analyze(image=uploaded)

    if result.is_mock:
        st.error("🧪 MOCK DATA below this line -- real vision/graph/fusion not wired in yet.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Segmentation overlay (MOCK)")
        st.image(result.vision.ovary_mask * 255, caption="Ovary mask (mock)", clamp=True)
        st.caption(f"Seg method: {result.vision.seg_method} | "
                   f"{len(result.vision.follicles)} follicles detected (mock)")
        for w in result.vision.warnings:
            st.caption(f"⚠️ {w}")

    with col2:
        st.subheader("Follicle graph (MOCK)")
        st.write(f"n = {result.graph.n} nodes")
        stats_display = {k: round(result.graph.stats.get(k, float("nan")), 3) for k in GRAPH_STAT_NAMES}
        st.json(stats_display)

    st.subheader("Risk assessment (MOCK)")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Risk score", f"{result.fusion.risk_score:.1f} / 100")
        st.metric("Band", result.fusion.band)
        st.caption(f"Baseline vote (paper method): "
                   f"{result.fusion.baseline_vote.votes_pcos}/{result.fusion.baseline_vote.of}")
    with c2:
        st.write("Feature-family attribution (mock)")
        st.bar_chart(result.fusion.family_attribution)

    st.subheader("Two-visit comparison (demo)")
    longi = mock_longitudinal()
    if longi.is_demo_pair:
        st.info("🧪 DEMO PAIR -- two unrelated sample images, not a real patient.")
    st.write(f"Trend: **{longi.trend}**")
    st.caption(longi.report_text)

else:
    st.info("Upload an image to see the analysis flow (C1 + C3 real, rest mock).")