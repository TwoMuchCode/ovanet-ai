"""
app/streamlit_app.py

OvaNet AI -- Clinical Decision Support & Spatial Graph Modeling
Ultrasound Image Analysis for PCOS Risk Assessment.
"""

from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import streamlit as st

from ovanet.pipeline import analyze
from ovanet.report import generate_pdf_report
from ovanet.visualization import render_follicle_overlay
from ovanet.contracts import GRAPH_STAT_NAMES, IMAGE_SIZE

st.set_page_config(
    page_title="OvaNet AI — Clinical Decision Support",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .ovanet-header {
        font-size: 2.3rem; font-weight: 800;
        background: linear-gradient(90deg, #7c3aed, #ec4899);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .ovanet-subtitle { color: #9ca3af; font-size: 1rem; margin-top: -4px; margin-bottom: 1.2rem; }
    .hero-card {
        background: linear-gradient(135deg, #1a1d29, #161822);
        border-radius: 16px; padding: 36px; border: 1px solid #2d2f3d; text-align: center;
    }
    .metric-card {
        background: #1a1d29; border-radius: 14px; padding: 18px; border: 1px solid #2d2f3d; text-align: center;
    }
    .risk-badge {
        display: inline-block; padding: 6px 18px; border-radius: 20px; font-weight: 700; font-size: 1.05rem; margin-top: 8px;
    }
    .risk-low { background: #064e3b; color: #6ee7b7; border: 1px solid #059669; }
    .risk-moderate { background: #78350f; color: #fcd34d; border: 1px solid #d97706; }
    .risk-high { background: #7f1d1d; color: #fca5a5; border: 1px solid #dc2626; }
    .trend-improving { background: #064e3b; color: #6ee7b7; padding: 4px 14px; border-radius: 12px; font-weight: 700; }
    .trend-stable { background: #1e293b; color: #94a3b8; padding: 4px 14px; border-radius: 12px; font-weight: 700; }
    .trend-worsening { background: #7f1d1d; color: #fca5a5; padding: 4px 14px; border-radius: 12px; font-weight: 700; }
    .section-card {
        background: #161822; border-radius: 14px; padding: 20px; border: 1px solid #272a38; margin-bottom: 16px;
    }
    .feature-pill {
        display: inline-block; background: #1f2937; color: #d1d5db; padding: 4px 12px;
        border-radius: 999px; margin: 3px; font-size: 0.85rem; border: 1px solid #374151;
    }
    .step-badge {
        background:#312e81; color:#c7d2fe; border-radius:999px; padding:3px 10px; font-size:0.8rem; margin-right:6px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🩺 OvaNet AI")
    st.caption("AI Decision-Support System for Ovarian Ultrasound")
    page = st.radio("Navigate", ["🔬 Analyze Scan", "📅 Two-Visit Monitoring", "🧪 Methodology", "📖 About"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 📋 Sample Demo Scans")
    sample_choice = st.selectbox(
        "Load test scan:",
        ["None (Upload custom)", "PCOS Sample (img_0_1195)", "PCOS Sample (img_0_1903)", "Normal Sample (img_0_2739)"]
    )
    st.markdown("---")
    st.caption("⚠️ **Research Prototype**")
    st.caption("Decision-support research tool. Not approved as a primary diagnostic device.")
    st.caption("TwoMuchCode · HackSpire '26")

RISK_CLASS = {"LOW": "risk-low", "MODERATE": "risk-moderate", "HIGH": "risk-high"}
RISK_EMOJI = {"LOW": "🟢", "MODERATE": "🟡", "HIGH": "🔴"}

SAMPLE_MAP = {
    "PCOS Sample (img_0_1195)": "data/gold_test_annotations/images/img_0_1195.jpg",
    "PCOS Sample (img_0_1903)": "data/gold_test_annotations/images/img_0_1903.jpg",
    "Normal Sample (img_0_2739)": "data/gold_test_annotations/images/img_0_2739.jpg",
}

# ---------------------------------------------------------------------------
# PAGE: About
# ---------------------------------------------------------------------------
if page == "📖 About":
    st.markdown('<div class="ovanet-header">About OvaNet AI</div>', unsafe_allow_html=True)
    st.markdown("""
    **OvaNet AI** extends classical radiomics and ultrasound image processing by integrating
    **instance follicle detection**, **spatial graph network modeling (k-NN)**, and **multi-modal fusion**.

    ### Core Innovations
    1. **Spatial Follicle Network (Graph Modeling)**: Captures the pathognomonic peripheral
       *"string-of-pearls"* distribution of follicles characteristic of polycystic ovaries.
    2. **Multi-Modal Feature Fusion**: Fuses ultrasound texture radiomics, spatial topological
       metrics, and learned neural representations.
    3. **Explainable AI (XAI)**: Visualizes follicle boundaries, spatial connections, and
       feature-family attributions for clinician transparency.
    4. **Longitudinal Visit Monitoring**: Tracks morphometric and risk changes across clinical follow-ups.

    ### Clinical Boundary
    - OvaNet AI is an investigational decision-support tool.
    - It assists clinicians in quantitative measurements; it does not replace physician judgment.
    """)

# ---------------------------------------------------------------------------
# PAGE: Methodology
# ---------------------------------------------------------------------------
elif page == "🧪 Methodology":
    st.markdown('<div class="ovanet-header">Methodology & Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <span class="step-badge">1</span> **Preprocess (C1)** &mdash; Grayscale conversion, standardized 256&times;256 resolution, Bilateral filter noise reduction, and CLAHE acoustic enhancement.<br><br>
    <span class="step-badge">2</span> **Vision ROI (C2)** &mdash; Classical morphological Black Top-Hat transform isolates hypoechoic fluid follicle structures and estimates the ovary sector boundary.<br><br>
    <span class="step-badge">3</span> **Radiomics (C3)** &mdash; Acoustic texture extraction (Mean, Std, Shannon Entropy, GLCM Contrast, GLCM Homogeneity) restricted to the segmented ovarian tissue.<br><br>
    <span class="step-badge">4</span> **Spatial Graph (C4)** &mdash; Follicles mapped as nodes; spatial $k$-NN graph ($k=3$) with Gaussian weighted distance edges. Calculates Peripheral Index, Clustering Coefficient, and size CV.<br><br>
    <span class="step-badge">5</span> **GCN Embedding** &mdash; Graph Convolutional Network computes topological graph representation.<br><br>
    <span class="step-badge">6</span> **Fusion & Calibration (C6)** &mdash; Calibrated risk estimate (0&ndash;100) mapped to LOW, MODERATE, or HIGH risk bands with SHAP feature-family attribution.
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PAGE: Two-Visit Monitoring
# ---------------------------------------------------------------------------
elif page == "📅 Two-Visit Monitoring":
    st.markdown('<div class="ovanet-header">📅 Two-Visit Longitudinal Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="ovanet-subtitle">Compare ultrasound scans from baseline and follow-up clinical visits</div>', unsafe_allow_html=True)

    vcol1, vcol2 = st.columns(2)
    with vcol1:
        st.markdown("#### Visit 1: Baseline Scan")
        v1_file = st.file_uploader("Upload Baseline Ultrasound", type=["png", "jpg", "jpeg"], key="v1_upload")
    with vcol2:
        st.markdown("#### Visit 2: Follow-up Scan")
        v2_file = st.file_uploader("Upload Follow-up Ultrasound", type=["png", "jpg", "jpeg"], key="v2_upload")

    use_demo_pair = st.checkbox("Or load pre-configured demo patient pair (Baseline vs Post-treatment)", value=(v1_file is None and v2_file is None))

    img1, img2 = None, None
    if v1_file and v2_file:
        img1 = np.array(Image.open(v1_file).convert("L"))
        img2 = np.array(Image.open(v2_file).convert("L"))
    elif use_demo_pair:
        p1 = "data/gold_test_annotations/images/img_0_1195.jpg"
        p2 = "data/gold_test_annotations/images/img_0_2739.jpg"
        img1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)

    if img1 is not None and img2 is not None:
        with st.spinner("Analyzing longitudinal visit scans..."):
            r1 = analyze(img1)
            r2 = analyze(img2)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Longitudinal Assessment Summary")

        delta_risk = r2.fusion.risk_score - r1.fusion.risk_score
        delta_follicles = len(r2.vision.follicles) - len(r1.vision.follicles)
        delta_pi = r2.graph.stats.get("peripheral_index", 0.0) - r1.graph.stats.get("peripheral_index", 0.0)

        if delta_risk <= -10.0:
            trend_label = "IMPROVING (Decreased Risk & Follicle Count)"
            trend_class = "trend-improving"
        elif delta_risk >= 10.0:
            trend_label = "PROGRESSING (Elevated Follicle Activity)"
            trend_class = "trend-worsening"
        else:
            trend_label = "STABLE (No Significant Morphological Change)"
            trend_class = "trend-stable"

        st.markdown(f'<span class="{trend_class}">Overall Trend: {trend_label}</span>', unsafe_allow_html=True)
        st.write("")

        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        with dcol1:
            st.metric("Baseline Risk", f"{r1.fusion.risk_score:.0f}/100", f"{r1.fusion.band}")
        with dcol2:
            st.metric("Follow-up Risk", f"{r2.fusion.risk_score:.0f}/100", f"{delta_risk:+.1f} pts", delta_color="inverse")
        with dcol3:
            st.metric("Follicle Count", f"{len(r2.vision.follicles)}", f"{delta_follicles:+d} follicles", delta_color="inverse")
        with dcol4:
            st.metric("Peripheral Index", f"{r2.graph.stats.get('peripheral_index', 0.0):.3f}", f"{delta_pi:+.3f}", delta_color="inverse")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Visual Comparison")
        comp1, comp2 = st.columns(2)
        with comp1:
            st.markdown(f"**Visit 1 (Baseline)** &mdash; {len(r1.vision.follicles)} follicles, Risk {r1.fusion.risk_score:.0f}")
            ov1 = render_follicle_overlay(r1.vision.ovary_mask, r1.vision, r1.graph)
            st.image(ov1, caption="Visit 1 Clinical Overlay", use_container_width=True)
        with comp2:
            st.markdown(f"**Visit 2 (Follow-up)** &mdash; {len(r2.vision.follicles)} follicles, Risk {r2.fusion.risk_score:.0f}")
            ov2 = render_follicle_overlay(r2.vision.ovary_mask, r2.vision, r2.graph)
            st.image(ov2, caption="Visit 2 Clinical Overlay", use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE: Analyze Scan (Main)
# ---------------------------------------------------------------------------
else:
    st.markdown('<div class="ovanet-header">🩺 OvaNet AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="ovanet-subtitle">Explainable PCOS risk assessment from ovarian ultrasound</div>', unsafe_allow_html=True)
    st.warning("⚕️ **Decision-support research prototype. Not a diagnosis.** Not clinically validated. Not a medical device.")

    uploaded = st.file_uploader("📤 Upload an ovarian ultrasound image", type=["png", "jpg", "jpeg"])

    img_array = None
    scan_display_name = "custom_upload"

    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("L")
        img_array = np.array(pil_img)
        scan_display_name = uploaded.name
    elif sample_choice != "None (Upload custom)":
        sample_path = SAMPLE_MAP[sample_choice]
        img_array = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
        scan_display_name = sample_choice.split()[0]
        st.info(f"Loaded demo image: `{sample_choice}`")

    if img_array is None:
        st.markdown('<div class="hero-card">', unsafe_allow_html=True)
        st.markdown("### Upload an ultrasound image or select a sample scan in the sidebar")
        st.markdown("OvaNet will detect follicles, construct a spatial follicle graph, compute acoustic radiomics, "
                    "and produce a calibrated risk estimate with feature attribution.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Executing OvaNet multi-modal analysis pipeline..."):
            result = analyze(img_array)

        st.success(f"✅ Analysis complete in {result.timings['total_s']:.2f}s")
        for w in result.warnings:
            st.caption(f"⚠️ {w}")

        # Top Risk & Attribution Section
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        rcol1, rcol2 = st.columns([1, 2])
        with rcol1:
            band = result.fusion.band
            st.markdown(f"""
                <div class="metric-card">
                    <div style="color:#9ca3af; font-size:0.9rem;">CALIBRATED RISK SCORE</div>
                    <div style="font-size:2.8rem; font-weight:800; color:white;">{result.fusion.risk_score:.0f}<span style="font-size:1.2rem; color:#6b7280;">/100</span></div>
                    <div class="risk-badge {RISK_CLASS.get(band, '')}">{RISK_EMOJI.get(band,'')} {band} RISK</div>
                </div>
            """, unsafe_allow_html=True)

            # Generate PDF Report in memory
            pdf_bytes = generate_pdf_report(result, patient_id="CLINICAL-DEMO", scan_id=scan_display_name)
            st.write("")
            st.download_button(
                label="📄 Download Clinical PDF Report",
                data=pdf_bytes,
                file_name=f"ovanet_report_{scan_display_name.replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        with rcol2:
            st.markdown("**Explainability & Feature-Family Attribution**")
            st.bar_chart(result.fusion.family_attribution)
            st.caption("Attribution weights from radiomics, follicle morphology, spatial graph topology, and neural embeddings.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Tabbed Findings View
        tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Clinical Overlay & Spatial Graph", "📊 Radiomics Biomarkers", "🕸️ Graph Network Topology", "📋 Clinical Context"])

        with tab1:
            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown("#### Clinical Overlay")
                # Render full color overlay
                raw_resized = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
                overlay_img = render_follicle_overlay(raw_resized, result.vision, result.graph)

                view_option = st.radio(
                    "Display Layer:",
                    ["Full Clinical Overlay (Ovary + Follicles + Graph)", "Ovary ROI Sector Mask", "Raw Ultrasound Scan"],
                    horizontal=True
                )

                if view_option == "Full Clinical Overlay (Ovary + Follicles + Graph)":
                    st.image(overlay_img, caption="Green: Ovary Sector ROI | Pink: Follicle Instances | Cyan: Spatial Connections", use_container_width=True)
                elif view_option == "Ovary ROI Sector Mask":
                    st.image(result.vision.ovary_mask * 255, caption="Estimated Ovarian Tissue Sector", clamp=True, use_container_width=True)
                else:
                    st.image(raw_resized, caption="Raw Preprocessed Ultrasound Frame", use_container_width=True)

            with col2:
                st.markdown("#### Morphometric Summary")
                n_f = len(result.vision.follicles)
                st.metric("Detected Follicles", f"{n_f} follicles", help="Threshold for polycystic appearance often >= 12-20 follicles per ovary")
                pi = result.graph.stats.get("peripheral_index", 0.0)
                st.metric("Peripheral Index", f"{pi:.3f}", help="Measures tendency of follicles to cluster around the outer ovarian cortex (string-of-pearls)")
                clustering = result.graph.stats.get("clustering_coeff", 0.0)
                st.metric("Clustering Coefficient", f"{clustering:.3f}", help="Average local connectivity between neighboring follicles")

                st.markdown("##### Segmentation Protocol")
                st.caption(f"Method: `{result.vision.seg_method.upper()}` (Selected via Gate G2)")
                st.caption(f"Quality Check: `{'PASS' if result.quality.ok else 'FLAGGED'}` (Blur Var: {result.quality.blur_var:.1f})")

        with tab2:
            st.markdown("#### Acoustic Texture Radiomics (Restricted to Ovary ROI)")
            rad_cols = st.columns(5)
            rad_items = [
                ("Mean Echogenicity", result.radiomics.mean, "Average grayscale reflection"),
                ("Intensity Std", result.radiomics.std, "Acoustic variance"),
                ("Shannon Entropy", result.radiomics.entropy, "Texture disorder (higher in PCOS stroma)"),
                ("GLCM Contrast", result.radiomics.glcm_contrast, "Local texture intensity differences"),
                ("GLCM Homogeneity", result.radiomics.glcm_homogeneity, "Uniformity of stromal pattern"),
            ]
            for col, (label, val, desc) in zip(rad_cols, rad_items):
                with col:
                    st.metric(label, f"{val:.3f}")
                    st.caption(desc)

        with tab3:
            st.markdown("#### Spatial Follicle Graph Statistics (Contract C4)")
            pills = "".join(
                f'<span class="feature-pill"><b>{k}</b>: {round(result.graph.stats.get(k, float("nan")), 3)}</span>'
                for k in GRAPH_STAT_NAMES
            )
            st.markdown(pills, unsafe_allow_html=True)
            st.write("")
            st.markdown(f"**Nodes**: {result.graph.n} follicle centroids | **Edges**: {len(result.graph.edges)} spatial $k$-NN connections ($k=3$, Gaussian distance weighting)")

        with tab4:
            st.markdown("#### Clinical Rotterdam Diagnostic Alignment")
            st.markdown("""
            Under the **Rotterdam Consensus (2003)**, PCOS is diagnosed when at least **2 of 3** criteria are met:
            1. **Oligo- or anovulation** (irregular or absent menstrual cycles).
            2. **Clinical and/or biochemical signs of hyperandrogenism** (hirsutism, acne, elevated androgens).
            3. **Polycystic ovarian morphology (PCOM)** on ultrasound:
               - &ge; 20 follicles (2&ndash;9 mm) per ovary, OR
               - Ovarian volume &ge; 10 mL.
            """)
            st.info("OvaNet AI addresses **Criterion 3 (PCOM)** by quantifying follicle count, peripheral displacement, and stromal texture.")