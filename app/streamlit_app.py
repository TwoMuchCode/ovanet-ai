"""
app/streamlit_app.py

OvaNet AI -- complete app: full pipeline display (graph, GCN, radiomics, SHAP)
+ clinical checklist + two-visit comparison + PDF export.
"""

from PIL import Image
import numpy as np
import streamlit as st
from ovanet.preprocessing import preprocess as real_preprocess
from ovanet.visualization import draw_follicle_overlay, draw_graph_overlay
from ovanet.pipeline import analyze
from ovanet.checklist import evaluate_checklist
from ovanet.longitudinal import compare as real_compare
from ovanet.report import generate_pdf_report
from ovanet.contracts import GRAPH_STAT_NAMES, ClinicalInput

st.set_page_config(page_title="OvaNet AI", layout="wide", page_icon="🩺")

st.markdown("""<style>
.hero-title{font-size:2.4rem;font-weight:800;background:linear-gradient(90deg,#a78bfa,#f472b6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.glass-card{background:rgba(255,255,255,0.04);border-radius:16px;padding:22px;border:1px solid rgba(255,255,255,0.08);margin-bottom:16px;}
.risk-badge{display:inline-block;padding:6px 18px;border-radius:999px;font-weight:700;margin-top:8px;}
.risk-low{background:#064e3b;color:#6ee7b7;}
.risk-moderate{background:#78350f;color:#fcd34d;}
.risk-high{background:#7f1d1d;color:#fca5a5;}
.stat-pill{display:inline-block;background:#1e2030;color:#d1d5db;padding:6px 14px;border-radius:10px;margin:4px;font-size:0.85rem;border:1px solid #2d2f42;}
.stat-pill b{color:#c4b5fd;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">🩺 OvaNet AI</div>', unsafe_allow_html=True)
st.warning("⚕️ Decision-support research prototype. Not a diagnosis. Not clinically validated.")

RISK_CLASS = {"LOW": "risk-low", "MODERATE": "risk-moderate", "HIGH": "risk-high"}

tab_analyze, tab_checklist, tab_twovisit = st.tabs(["🔬 Single Analysis", "📋 Clinical Checklist", "📅 Two-Visit Comparison"])

# ===========================================================================
# TAB 1: Single Analysis -- FULL pipeline display
# ===========================================================================
with tab_analyze:
    uploaded = st.file_uploader("Upload an ovarian ultrasound image", type=["png", "jpg", "jpeg"], key="single")

    if uploaded:
        img = np.array(Image.open(uploaded).convert("L"))

        with st.spinner("Running full pipeline: preprocess -> detect -> radiomics -> graph -> GCN -> fusion..."):
            prep = real_preprocess(img)          # need raw/enh directly for the step-by-step views
            result = analyze(img)                # full pipeline result (reuses the same steps internally)

        st.success(f"✅ Pipeline complete in {result.timings['total_s']:.2f}s")
        for w in result.warnings:
            st.caption(f"⚠️ {w}")

        st.markdown("---")

        # ===================== STEP 1 — Raw uploaded image =====================
        st.markdown("### Step 1 — Raw USG image (as uploaded)")
        st.image(img, caption="Original upload", clamp=True, width=350)

        # ===================== STEP 2 — Preprocessed =====================
        st.markdown("### Step 2 — Preprocessed (resized to 256×256, normalized)")
        st.image(prep.raw, caption="Preprocessed (raw channel)", clamp=True, width=350)
        if not prep.quality.ok:
            st.warning(f"Quality flags: {prep.quality.reasons}")

        # ===================== STEP 3 — Follicle detection =====================
        st.markdown("### Step 3 — Follicle detection (ROI + individual follicles marked)")
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(result.vision.ovary_mask * 255, caption="Ovary ROI mask", clamp=True, width=330)
        with col_b:
            follicle_overlay = draw_follicle_overlay(prep.raw, result.vision.follicles)
            st.image(follicle_overlay, caption=f"{len(result.vision.follicles)} follicles detected (circled + numbered)", width=330)

        # ===================== STEP 4 — kNN follicle graph =====================
        st.markdown("### Step 4 — kNN Follicle Graph (spatial relationships between follicles)")
        graph_overlay = draw_graph_overlay(prep.raw, result.vision.follicles, result.graph)
        st.image(graph_overlay, caption=f"n = {result.graph.n} nodes, kNN edges (brighter/thicker = stronger connection)", width=400)

        pills = "".join(
            f'<span class="stat-pill"><b>{k.replace("_"," ")}</b>: {round(result.graph.stats.get(k, float("nan")), 3)}</span>'
            for k in GRAPH_STAT_NAMES
        )
        st.markdown(pills, unsafe_allow_html=True)

        # ===================== STEP 5 — GCN embedding =====================
        st.markdown("### Step 5 — GCN Embedding (E_gnn, 16-dim, from the graph above)")
        from ovanet.gnn import gcn_embed
        e_gnn = gcn_embed(result.graph)
        gnn_cols = st.columns(8)
        for i, col in enumerate(gnn_cols):
            with col:
                st.metric(f"d{i}", f"{e_gnn[i]:.3f}")
        gnn_cols2 = st.columns(8)
        for i, col in enumerate(gnn_cols2):
            with col:
                st.metric(f"d{i+8}", f"{e_gnn[i+8]:.3f}")

        # ===================== STEP 6 — Radiomics =====================
        st.markdown("### Step 6 — Radiomics (texture features from the ROI)")
        rad_cols = st.columns(5)
        rad_items = [
            ("Mean", result.radiomics.mean), ("Std", result.radiomics.std),
            ("Entropy", result.radiomics.entropy), ("Contrast", result.radiomics.glcm_contrast),
            ("Homogeneity", result.radiomics.glcm_homogeneity),
        ]
        for col, (label, val) in zip(rad_cols, rad_items):
            with col:
                st.metric(label, f"{val:.3f}")

        # ===================== STEP 7 — Fusion / risk score =====================
        st.markdown("### Step 7 — Fusion model: combined risk score")
        rcol1, rcol2 = st.columns([1, 2])
        with rcol1:
            band = result.fusion.band
            st.markdown(f"""<div style="text-align:center;">
                <div style="color:#9ca3af; font-size:0.9rem;">RISK SCORE</div>
                <div style="font-size:2.8rem; font-weight:800;">{result.fusion.risk_score:.0f}<span style="font-size:1.2rem;color:#6b7280;">/100</span></div>
                <div class="risk-badge {RISK_CLASS.get(band,'')}">{band}</div>
            </div>""", unsafe_allow_html=True)
        with rcol2:
            st.markdown("**Feature-family attribution (SHAP-based)**")
            st.bar_chart(result.fusion.family_attribution)

        # ===================== PDF export =====================
        pdf_bytes = generate_pdf_report(result, patient_label=uploaded.name)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"ovanet_report_{uploaded.name}.pdf",
            mime="application/pdf",
        )
# ===========================================================================
# TAB 2: Clinical Checklist
# ===========================================================================
with tab_checklist:
    st.markdown("### Rotterdam / 2023 Guideline Checklist (rule-based, NOT fed to the AI model)")
    col1, col2 = st.columns(2)
    with col1:
        cycle_irregular = st.checkbox("Irregular menstrual cycles")
        hyperandrogenism = st.checkbox("Clinical/biochemical hyperandrogenism (acne, hirsutism, elevated androgens)")
        years_since_menarche = st.number_input("Years since menarche", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
    with col2:
        lh_fsh = st.number_input("LH/FSH ratio (optional, 0 = not entered)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
        amh = st.number_input("AMH level ng/mL (optional, 0 = not entered)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)

    if st.button("Evaluate checklist"):
        clinical = ClinicalInput(
            cycle_irregular=cycle_irregular, hyperandrogenism=hyperandrogenism,
            lh_fsh=lh_fsh if lh_fsh > 0 else None, amh=amh if amh > 0 else None,
            years_since_menarche=years_since_menarche, source="user_entered",
        )
        result = evaluate_checklist(clinical)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"**{result['summary']}**")
        if result["criteria_met"]:
            st.write("✅ Criteria met:", ", ".join(result["criteria_met"]))
        if result["criteria_unknown"]:
            st.write("❓ Not entered:", ", ".join(result["criteria_unknown"]))
        if result["adolescent_caution"]:
            st.warning(result["adolescent_caution"])
        st.caption(result["disclaimer"])
        st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
# TAB 3: Two-Visit Comparison
# ===========================================================================
with tab_twovisit:
    st.markdown("### Two-Visit Comparison (demo pair unless stated otherwise)")
    col1, col2 = st.columns(2)
    with col1:
        img1_file = st.file_uploader("Visit 1 image", type=["png", "jpg", "jpeg"], key="v1")
    with col2:
        img2_file = st.file_uploader("Visit 2 image", type=["png", "jpg", "jpeg"], key="v2")

    if img1_file and img2_file:
        img1 = np.array(Image.open(img1_file).convert("L"))
        img2 = np.array(Image.open(img2_file).convert("L"))
        r1 = analyze(img1)
        r2 = analyze(img2)
        comparison = real_compare(r1, r2, is_demo_pair=True)

        st.info("🧪 DEMO PAIR — unless these are verified to be the same patient's real scans.")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"**Trend: {comparison.trend}**")
        st.write(comparison.report_text)
        st.json(comparison.deltas)
        st.markdown('</div>', unsafe_allow_html=True)