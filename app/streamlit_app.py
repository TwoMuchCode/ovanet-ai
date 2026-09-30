"""
app/streamlit_app.py

OvaNet AI -- Streamlit app with sidebar navigation and landing page.
"""

from PIL import Image
import numpy as np
import streamlit as st

from ovanet.pipeline import analyze
from ovanet.mocks import mock_longitudinal
from ovanet.contracts import GRAPH_STAT_NAMES

st.set_page_config(page_title="OvaNet AI", layout="wide", page_icon="🩺")

st.markdown("""
<style>
    .ovanet-header {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #7c3aed, #ec4899);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .ovanet-subtitle { color: #9ca3af; font-size: 1rem; margin-top: -5px; margin-bottom: 1.5rem; }
    .hero-card {
        background: linear-gradient(135deg, #1a1d29, #161822);
        border-radius: 18px; padding: 40px; border: 1px solid #2d2f3d; text-align: center;
    }
    .metric-card { background: #1a1d29; border-radius: 14px; padding: 20px; border: 1px solid #2d2f3d; text-align: center; }
    .risk-badge { display: inline-block; padding: 6px 18px; border-radius: 20px; font-weight: 700; font-size: 1.1rem; margin-top: 8px; }
    .risk-low { background: #064e3b; color: #6ee7b7; }
    .risk-moderate { background: #78350f; color: #fcd34d; }
    .risk-high { background: #7f1d1d; color: #fca5a5; }
    .section-card { background: #161822; border-radius: 14px; padding: 22px; border: 1px solid #272a38; margin-bottom: 16px; }
    .feature-pill { display: inline-block; background: #1f2937; color: #d1d5db; padding: 4px 12px; border-radius: 999px; margin: 3px; font-size: 0.85rem; }
    .step-badge { background:#312e81; color:#c7d2fe; border-radius:999px; padding:3px 10px; font-size:0.8rem; margin-right:6px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🩺 OvaNet AI")
    page = st.radio("Navigate", ["🔬 Analyze", "📖 About", "🧪 Methodology"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Decision-support research prototype.")
    st.caption("Not a diagnosis. Not clinically validated.")
    st.markdown("---")
    st.caption("TwoMuchCode · HackSpire '26")

RISK_CLASS = {"LOW": "risk-low", "MODERATE": "risk-moderate", "HIGH": "risk-high"}
RISK_EMOJI = {"LOW": "🟢", "MODERATE": "🟡", "HIGH": "🔴"}

# ---------------------------------------------------------------------------
# PAGE: About
# ---------------------------------------------------------------------------
if page == "📖 About":
    st.markdown('<div class="ovanet-header">About OvaNet AI</div>', unsafe_allow_html=True)
    st.markdown("""
    OvaNet AI extends a validated statistical radiomics baseline with automatic follicle
    detection, spatial-graph modelling, and fusion -- producing an explainable risk
    estimate for clinician review.

    **What it does:**
    - Detects follicles and the ovarian region from an ultrasound frame
    - Computes texture (radiomics) features from a published baseline method
    - Builds a spatial graph of follicle positions and relationships
    - Fuses all signals into a calibrated risk score, with feature-family attribution

    **What it does NOT do:**
    - It does not diagnose PCOS
    - It is not clinically validated or regulatory-approved
    - It does not replace a clinician's judgement
    """)

# ---------------------------------------------------------------------------
# PAGE: Methodology
# ---------------------------------------------------------------------------
elif page == "🧪 Methodology":
    st.markdown('<div class="ovanet-header">Methodology</div>', unsafe_allow_html=True)
    st.markdown("""
    <span class="step-badge">1</span> **Preprocess** -- grayscale, resize, quality check<br><br>
    <span class="step-badge">2</span> **Vision** -- classical CV detects ovary ROI + follicle instances<br><br>
    <span class="step-badge">3</span> **Radiomics** -- mean, std, entropy, GLCM contrast/homogeneity (paper baseline)<br><br>
    <span class="step-badge">4</span> **Graph** -- follicles as nodes, kNN spatial graph, 8 statistics incl. peripheral distribution<br><br>
    <span class="step-badge">5</span> **GCN** -- learned graph embedding (out-of-fold, label-safe)<br><br>
    <span class="step-badge">6</span> **Fusion** -- calibrated risk score + SHAP feature-family attribution
    """, unsafe_allow_html=True)
    st.warning("⚠️ Known limitation: dataset has a confirmed size/brightness shortcut (B2 baseline = 100% on that alone). "
               "Current risk scores are not yet proven free of this contamination -- final honest evaluation is pending.")

# ---------------------------------------------------------------------------
# PAGE: Analyze
# ---------------------------------------------------------------------------
else:
    st.markdown('<div class="ovanet-header">🩺 OvaNet AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="ovanet-subtitle">Explainable PCOS risk assessment from ovarian ultrasound</div>', unsafe_allow_html=True)
    st.warning("⚕️ **Decision-support research prototype. Not a diagnosis.** Not clinically validated. Not a medical device.")

    uploaded = st.file_uploader("📤 Upload an ovarian ultrasound image", type=["png", "jpg", "jpeg"])

    if uploaded is None:
        st.markdown('<div class="hero-card">', unsafe_allow_html=True)
        st.markdown("### Upload an image to begin")
        st.markdown("OvaNet will detect follicles, build a spatial graph, compute radiomics, "
                     "and produce a calibrated risk estimate with full explainability.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        pil_img = Image.open(uploaded).convert("L")
        img_array = np.array(pil_img)

        with st.spinner("Running analysis pipeline..."):
            result = analyze(img_array)

        st.success(f"✅ Analysis complete in {result.timings['total_s']:.2f}s")
        for w in result.warnings:
            st.caption(f"⚠️ {w}")

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        rcol1, rcol2 = st.columns([1, 2])
        with rcol1:
            band = result.fusion.band
            st.markdown(f"""
                <div class="metric-card">
                    <div style="color:#9ca3af; font-size:0.9rem;">RISK SCORE</div>
                    <div style="font-size:2.8rem; font-weight:800; color:white;">{result.fusion.risk_score:.0f}<span style="font-size:1.2rem; color:#6b7280;">/100</span></div>
                    <div class="risk-badge {RISK_CLASS.get(band, '')}">{RISK_EMOJI.get(band,'')} {band}</div>
                </div>
            """, unsafe_allow_html=True)
        with rcol2:
            st.markdown("**Feature-family attribution**")
            st.bar_chart(result.fusion.family_attribution)
        st.markdown('</div>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["🖼️ Vision & Graph", "📊 Radiomics", "📅 Two-visit demo"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Segmentation overlay")
                st.image(result.vision.ovary_mask * 255, caption="Ovary ROI mask", clamp=True, use_container_width=True)
                st.caption(f"Method: `{result.vision.seg_method}` · {len(result.vision.follicles)} follicles")
            with col2:
                st.markdown("#### Follicle graph")
                st.markdown(f"**n = {result.graph.n}** nodes")
                pills = "".join(
                    f'<span class="feature-pill">{k}: {round(result.graph.stats.get(k, float("nan")), 3)}</span>'
                    for k in GRAPH_STAT_NAMES
                )
                st.markdown(pills, unsafe_allow_html=True)

        with tab2:
            rad_cols = st.columns(5)
            rad_items = [
                ("Mean", result.radiomics.mean), ("Std", result.radiomics.std),
                ("Entropy", result.radiomics.entropy), ("Contrast", result.radiomics.glcm_contrast),
                ("Homogeneity", result.radiomics.glcm_homogeneity),
            ]
            for col, (label, val) in zip(rad_cols, rad_items):
                with col:
                    st.metric(label, f"{val:.3f}")

        with tab3:
            longi = mock_longitudinal()
            st.info("🧪 DEMO PAIR -- two unrelated sample images, not a real patient.")
            st.write(f"Trend: **{longi.trend}**")