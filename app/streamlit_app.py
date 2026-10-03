"""
app/streamlit_app.py

OvaNet AI -- redesigned frontend. PRESENTATION ONLY: calls the same pipeline functions
as before (preprocess, analyze, evaluate_checklist, compare, generate_pdf_report) and
renders their real outputs. No model, contract or pipeline logic lives here.

Run:  streamlit run app/streamlit_app.py
"""

from PIL import Image
import numpy as np
import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ovanet.preprocessing import preprocess as real_preprocess
from ovanet.visualization import draw_follicle_overlay, draw_graph_overlay
from ovanet.pipeline import analyze
from ovanet.checklist import evaluate_checklist
from ovanet.longitudinal import compare as real_compare
from ovanet.report import generate_pdf_report
from ovanet.contracts import GRAPH_STAT_NAMES, ClinicalInput

from ovanet_ui.theme import CSS
from ovanet_ui import components as ui

st.set_page_config(page_title="OvaNet AI", layout="wide", page_icon="🩺", initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------- helpers
def _file_key(f):
    return (f.name, f.size)


def _region_view(raw01, mask):
    """Tint the existing ovary mask over the preprocessed image (display only)."""
    g = np.clip(raw01 * 255.0, 0, 255).astype(np.float32)
    rgb = np.stack([g, g, g], axis=-1)
    m = np.asarray(mask) > 0
    tint = np.array([139, 92, 246], dtype=np.float32)
    rgb[m] = rgb[m] * 0.62 + tint * 0.38
    return rgb.astype(np.uint8)


def _run(f, tl_slot=None):
    """Run the existing pipeline once per uploaded file and cache it for this session."""
    cache = st.session_state.setdefault("_cache", {})
    k = _file_key(f)
    if k not in cache:
        if tl_slot is not None:
            with tl_slot.container():
                ui.timeline("running")
        img = np.array(Image.open(f).convert("L"))
        prep = real_preprocess(img)
        result = analyze(img)
        cache[k] = (img, prep, result)
    return cache[k]


ui.nav()
ui.hero()
ui.how_it_works()

# =========================================================================
# ANALYSIS WORKSPACE
# =========================================================================
ui.section("Analysis workspace", "Ultrasound analysis workspace",
           "Upload an ovarian ultrasound, review the evidence at every stage, then see how the risk estimate was formed.", anchor="analysis")

with st.container(key="glass_workspace"):
    ui.workflow_strip()
    uploaded = st.file_uploader("Upload ovarian ultrasound", type=["png", "jpg", "jpeg"], key="single",
                                help="Supported image formats: PNG, JPG, JPEG")
    st.caption("Supported image formats: PNG, JPG, JPEG")
    tl_slot = st.empty()

    data = None
    if uploaded:
        try:
            data = _run(uploaded, tl_slot)
        except Exception as exc:  # surface failures without breaking the page
            tl_slot.empty()
            st.error(f"The image could not be analyzed: {exc}")
    if data:
        img, prep, result = data
        with tl_slot.container():
            ui.timeline("done", result.timings.get("total_s"))
    else:
        with tl_slot.container():
            ui.timeline("idle")

if data:
    img, prep, result = data
    for w in result.warnings:
        st.caption(f"⚠️ {w}")

    # ---- image viewer ----
    with st.container(key="viewer_main"):
        tabs = st.tabs(["Original", "Enhanced", "Ovary Region", "Follicles", "Spatial Graph"])
        with tabs[0]:
            st.image(img, clamp=True, use_container_width=True)
            ui.html('<span class="badge">Original upload</span>')
        with tabs[1]:
            st.image(prep.enh, clamp=True, use_container_width=True)
            ui.html('<span class="badge">Enhanced · 256×256 normalized</span>')
        with tabs[2]:
            st.image(_region_view(prep.raw, result.vision.ovary_mask), use_container_width=True)
            ui.html(f'<span class="badge ok">Ovary mask (violet tint)</span><span class="badge">Method: {ui.e(result.vision.seg_method)}</span>')
        with tabs[3]:
            st.image(draw_follicle_overlay(prep.raw, result.vision.follicles), use_container_width=True)
            ui.html(f'<span class="badge ok">{len(result.vision.follicles)} follicles</span><span class="badge">Circled &amp; numbered</span>')
        with tabs[4]:
            st.image(draw_graph_overlay(prep.raw, result.vision.follicles, result.graph), use_container_width=True)
            ui.html('<span class="badge ok">kNN graph</span><span class="badge">Brighter / thicker edge = stronger connection</span>')
    ui.status_strip(result.quality.ok, result.quality.reasons, bool(np.asarray(result.vision.ovary_mask).any()),
                    len(result.vision.follicles), result.graph.n, len(result.graph.edges))

    # ---- risk ----
    ui.section("Result", "Risk Assessment")
    ui.risk_card(result.fusion.risk_score, result.fusion.band, getattr(result.fusion, "model_version", ""))
    bv = getattr(result.fusion, "baseline_vote", None)
    if bv is not None:
        ui.html(f'<div class="note">Baseline models: {ui.e(bv.votes_pcos)} of {ui.e(bv.of)} voted PCOS-pattern ({ui.e(bv.label)}).</div>')
    try:
        pdf_bytes = generate_pdf_report(result, patient_label=uploaded.name)
        st.download_button("Download PDF report", data=pdf_bytes, file_name=f"ovanet_report_{uploaded.name}.pdf", mime="application/pdf")
    except Exception as exc:
        st.caption(f"PDF report unavailable: {exc}")

    # ---- insights ----
    ui.section("Insights", "Why this assessment?", "Evidence behind the estimate, from feature families to individual follicles.", anchor="insights")
    ui.attribution(result.fusion.family_attribution, result.fusion.top_features)

    ui.section("Radiomics", "Texture & intensity features")
    ui.radiomics_cards(result.radiomics)

    ui.section("Follicles", "Follicle-Level Analysis")
    ui.follicle_section(result.vision.follicles, result.graph.stats)

    ui.section("Graph", "Spatial Relationships")
    from ovanet.gnn import gcn_embed, is_trained
    e_gnn = None
    if is_trained():  # never show an untrained (random) embedding as if it were a result
        try:
            e_gnn = gcn_embed(result.graph)
        except Exception:
            e_gnn = None
    ui.graph_section(result.graph, GRAPH_STAT_NAMES, e_gnn)
else:
    ui.section("Result", "Risk Assessment")
    ui.empty("Upload an ultrasound above to generate a risk estimate. Nothing is shown until real model output exists.")
    ui.section("Insights", "Why this assessment?", anchor="insights")
    ui.empty("Feature-family attribution, radiomics, follicle and graph evidence will appear here after analysis.")

# =========================================================================
# CLINICAL CHECKLIST
# =========================================================================
ui.section("Clinical review", "Clinical Review Checklist",
           "Rule-based and transparent. Not used by the AI model. For clinician/researcher review — not a diagnostic conclusion.", anchor="checklist")
with st.container(key="glass_checklist"):
    with st.form("checklist_form"):
        c1, c2 = st.columns(2)
        with c1:
            cycle_irregular = st.checkbox("Irregular menstrual cycles")
            hyperandrogenism = st.checkbox("Clinical/biochemical hyperandrogenism (acne, hirsutism, elevated androgens)")
            years_since_menarche = st.number_input("Years since menarche", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
        with c2:
            lh_fsh = st.number_input("LH/FSH ratio (optional, 0 = not entered)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
            amh = st.number_input("AMH level ng/mL (optional, 0 = not entered)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
        submitted = st.form_submit_button("Evaluate checklist")
    if submitted:
        clinical = ClinicalInput(cycle_irregular=cycle_irregular, hyperandrogenism=hyperandrogenism,
                                 lh_fsh=lh_fsh if lh_fsh > 0 else None, amh=amh if amh > 0 else None,
                                 years_since_menarche=years_since_menarche, source="user_entered")
        st.session_state["_checklist"] = evaluate_checklist(clinical)
if st.session_state.get("_checklist"):
    ui.checklist_cards(st.session_state["_checklist"])

# =========================================================================
# LONGITUDINAL COMPARISON
# =========================================================================
ui.section("Longitudinal", "Track Changes Over Time",
           "Compare two scans. Treated as a demo pair unless verified to be the same patient’s real visits.", anchor="comparison")
with st.container(key="glass_compare"):
    col1, col2 = st.columns(2)
    with col1:
        ui.visit_header(1, "Previous Scan")
        img1_file = st.file_uploader("Visit 1 image", type=["png", "jpg", "jpeg"], key="v1")
    with col2:
        ui.visit_header(2, "Current Scan")
        img2_file = st.file_uploader("Visit 2 image", type=["png", "jpg", "jpeg"], key="v2")

    if img1_file and img2_file:
        try:
            with st.spinner("Analyzing both visits…"):
                _, p1, r1 = _run(img1_file)
                _, p2, r2 = _run(img2_file)
                comparison = real_compare(r1, r2, is_demo_pair=True)
            v1c, v2c = st.columns(2)
            with v1c:
                st.image(draw_follicle_overlay(p1.raw, r1.vision.follicles), use_container_width=True)
                ui.visit_summary(r1)
            with v2c:
                st.image(draw_follicle_overlay(p2.raw, r2.vision.follicles), use_container_width=True)
                ui.visit_summary(r2)
            ui.comparison_result(comparison, r1, r2)
        except Exception as exc:
            st.error(f"The comparison could not be completed: {exc}")
    else:
        ui.empty("Upload both visits to compare risk score, follicle count and trend.")

# =========================================================================
# PCOS PRECAUTIONS (general health guidance, not tool-specific)
# =========================================================================
ui.section("Precautions", "General Precautions & Lifestyle Guidance",
           "General, widely-recommended guidance for people managing or at risk of PCOS. Not personalized medical advice — always consult a qualified clinician.", anchor="precautions")
with st.container(key="glass_precautions"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
#### ✅ Generally recommended
- Maintain a balanced diet with controlled refined-sugar and processed-carb intake
- Regular moderate exercise (150+ min/week is a commonly cited general target)
- Prioritize consistent sleep (7–9 hours); poor sleep is linked to worse insulin resistance
- Track menstrual cycles to identify irregularity patterns early
- Manage stress — chronic stress affects hormonal balance
- Regular check-ups with a gynecologist or endocrinologist, especially if symptoms are present
- Discuss family history of PCOS, diabetes, or metabolic conditions with your doctor
""")
    with c2:
        st.markdown("""
#### ⚠️ Generally discouraged
- Self-diagnosing or self-medicating without professional input
- Ignoring persistent irregular cycles, excess hair growth, or acne without medical review
- Crash dieting or extreme calorie restriction without professional guidance
- Delaying care due to symptom embarrassment — PCOS is common and manageable
- Relying on unverified online "cures" or unregulated supplements
""")
    st.markdown("""
---
**When to see a doctor promptly:** significantly irregular or absent periods,
sudden or severe acne/hair growth changes, difficulty conceiving, or any
symptom that concerns you.
""")

ui.transparency()
ui.footer()