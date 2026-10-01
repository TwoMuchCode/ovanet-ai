"""
app/streamlit_app.py

OvaNet AI -- Streamlit app.
Day 5: real C1, C2, C3, C4 wired in. Fusion (C6) still mock until Day 6.
"""

from PIL import Image
import numpy as np
import streamlit as st

from ovanet.preprocessing import preprocess as real_preprocess
from ovanet.detector import detect_follicles as real_detect
from ovanet.radiomics import extract as real_extract
from ovanet.graph import build_graph as real_build_graph
from ovanet.gnn import gcn_embed as real_gcn_embed
from ovanet.mocks import mock_fusion, mock_longitudinal
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

    # ---- REAL C1 ----
    real_prep = real_preprocess(img_array)

    # ---- REAL C2 ----
    real_vision = real_detect(real_prep.raw)

    # ---- REAL C3 ----
    real_radiomics = real_extract(real_prep.raw)

    # ---- REAL C4 ----
    real_graph = real_build_graph(real_vision.follicles, real_vision.ovary_mask)
    real_e_gnn = real_gcn_embed(real_graph)

    st.success("✅ Real pipeline active: preprocessing, follicle detection, radiomics, graph, GCN embedding.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Segmentation overlay -- REAL")
        st.image(real_vision.ovary_mask * 255, caption="Ovary ROI mask", clamp=True)
        st.caption(f"Seg method: {real_vision.seg_method} | {len(real_vision.follicles)} follicles detected")
        for w in real_vision.warnings:
            st.caption(f"⚠️ {w}")

    with col2:
        st.subheader("Follicle graph -- REAL")
        st.write(f"n = {real_graph.n} nodes")
        stats_display = {k: round(real_graph.stats.get(k, float("nan")), 3) for k in GRAPH_STAT_NAMES}
        st.json(stats_display)

    st.subheader("Radiomics (F_k) -- REAL")
    st.json({
        "mean": round(real_radiomics.mean, 4),
        "std": round(real_radiomics.std, 4),
        "entropy": round(real_radiomics.entropy, 4),
        "glcm_contrast": round(real_radiomics.glcm_contrast, 4),
        "glcm_homogeneity": round(real_radiomics.glcm_homogeneity, 4),
    })

    with st.expander("GCN embedding (E_gnn) -- REAL, 16-dim"):
        st.write(real_e_gnn)

    # ---- MOCK: fusion only (Day 6 work) ----
    fusion = mock_fusion()
    st.error("🧪 MOCK fusion/risk score below -- real fusion MLP not wired in yet (Day 6).")

    st.subheader("Risk assessment (MOCK)")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Risk score", f"{fusion.risk_score:.1f} / 100")
        st.metric("Band", fusion.band)
    with c2:
        st.bar_chart(fusion.family_attribution)

    st.subheader("Two-visit comparison (demo)")
    longi = mock_longitudinal()
    if longi.is_demo_pair:
        st.info("🧪 DEMO PAIR -- two unrelated sample images, not a real patient.")
    st.write(f"Trend: **{longi.trend}**")

else:
    st.info("Upload an image to see the pipeline (C1-C4 real, fusion mock until Day 6).")