"""
app/streamlit_app.py

OvaNet AI -- Streamlit app with sidebar navigation and landing page.
"""

from PIL import Image
import numpy as np
import streamlit as st

from ovanet.pipeline import analyze
import cv2
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
def make_overlay(img_array, result, draw_graph=False):
    base = cv2.resize(img_array, (256, 256))
    rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    om = (result.vision.ovary_mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(om, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, cnts, -1, (0, 200, 255), 1)
    lab = result.vision.follicle_labels
    for k in np.unique(lab):
        if k == 0:
            continue
        c, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, c, -1, (255, 60, 60), 1)
    if draw_graph:
        pos = {n.id: (int(n.x), int(n.y)) for n in result.graph.nodes}
        for e in result.graph.edges:
            s, t = (e.source, e.target) if hasattr(e, "source") else (e["source"], e["target"])
            if s in pos and t in pos:
                cv2.line(rgb, pos[s], pos[t], (250, 220, 40), 1)
        for p in pos.values():
            cv2.circle(rgb, p, 3, (80, 255, 120), -1)
    return cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_NEAREST)


def trend_from_delta(delta, margin=10.0):
    if delta > margin:
        return "WORSENING"
    if delta < -margin:
        return "IMPROVING"
    return "STABLE"

# ---------------------------------------------------------------------------
# PAGE: About
# ---------------------------------------------------------------------------
if page == "📖 About":
    st.markdown('<div class="ovanet-header">About OvaNet AI</div>', unsafe_allow_html=True)
    st.markdown("""
    OvaNet AI extends a published statistical radiomics baseline with automatic follicle
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
    st.error("Dataset shortcut: image size/brightness alone separates the classes (B2 AUROC = 1.0). "
            "Scores are NOT evidence of PCOS. See experiments/results/final_results.md.")
# ---------------------------------------------------------------------------
# PAGE: Analyze
# ---------------------------------------------------------------------------
else:
    st.markdown('<div class="ovanet-header">🩺 OvaNet AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="ovanet-subtitle">Experimental ovarian ultrasound analysis (research prototype)</div>', unsafe_allow_html=True)
    st.warning("⚕️ **Decision-support research prototype. Not a diagnosis.** Not clinically validated. Not a medical device.")

    uploaded = st.file_uploader("📤 Upload an ovarian ultrasound image", type=["png", "jpg", "jpeg"])

    if uploaded is None:
        st.markdown('<div class="hero-card"><h3>Upload an image to begin</h3>'
                    'OvaNet detects candidate follicles, builds a spatial graph and computes texture features.</div>',
                    unsafe_allow_html=True)
    else:
        img_array = np.array(Image.open(uploaded).convert("L"))
        with st.spinner("Running analysis pipeline..."):
            result = analyze(img_array)

        st.success(f"Analysis complete in {result.timings['total_s']:.2f}s")

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        rcol1, rcol2 = st.columns([1, 2])
        with rcol1:
            band = result.fusion.band
            st.markdown(f"""
                <div class="metric-card">
                    <div style="color:#9ca3af; font-size:0.9rem;">MODEL SCORE (experimental)</div>
                    <div style="font-size:2.8rem; font-weight:800; color:white;">{result.fusion.risk_score:.0f}<span style="font-size:1.2rem; color:#6b7280;">/100</span></div>
                    <div class="risk-badge {RISK_CLASS.get(band, '')}">{RISK_EMOJI.get(band,'')} {band}</div>
                </div>""", unsafe_allow_html=True)
            st.error(result.warnings[0])
        with rcol2:
            st.markdown("**Feature-family attribution** (linear contribution to the model's log-odds)")
            st.bar_chart(result.fusion.family_attribution)
            bv = result.fusion.baseline_vote
            st.caption(f"Paper-style threshold vote: {bv.votes_pcos}/{bv.of} texture features on the dataset-positive side ({bv.label}).")
        st.markdown('</div>', unsafe_allow_html=True)
        st.warning(result.warnings[1])
        for w in result.warnings[2:]:
            st.caption(f"⚠️ {w}")

        tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Vision & Graph", "📊 Radiomics & drivers", "📋 Clinical checklist", "📅 Two-visit demo"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.image(make_overlay(img_array, result), caption="Ovary region (cyan) + candidate follicles (red)", use_container_width=True)
            with c2:
                st.image(make_overlay(img_array, result, draw_graph=True), caption="Follicle graph (kNN, k=3)", use_container_width=True)
            st.caption(f"Method: `{result.vision.seg_method}` · {len(result.vision.follicles)} candidates · sizes in pixels, no mm calibration · single 2D frame.")
            st.markdown("".join(f'<span class="feature-pill">{k}: {round(result.graph.stats.get(k, float("nan")), 3)}</span>'
                                for k in GRAPH_STAT_NAMES), unsafe_allow_html=True)

        with tab2:
            cols5 = st.columns(5)
            for col, (lbl, val) in zip(cols5, [("Mean", result.radiomics.mean), ("Std", result.radiomics.std),
                                               ("Entropy", result.radiomics.entropy), ("Contrast", result.radiomics.glcm_contrast),
                                               ("Homogeneity", result.radiomics.glcm_homogeneity)]):
                col.metric(lbl, f"{val:.3f}")
            st.markdown("**Top contributing features**")
            st.table([{"feature": t.name, "family": t.family, "contribution": round(t.contribution, 3)} for t in result.fusion.top_features])

        with tab3:
            st.caption("Rule-based checklist from clinician-entered inputs only. The image model is NOT used here. "
                       "Verify criteria against the 2023 International PCOS Guideline.")
            a = st.checkbox("Oligo/anovulation (irregular cycles)")
            b = st.checkbox("Clinical or biochemical hyperandrogenism")
            c = st.checkbox("Polycystic ovarian morphology on clinician's own ultrasound read")
            d = st.checkbox("Elevated AMH (adult)")
            rotterdam = sum([a, b, c]) >= 2
            g2023 = a and (b or c or d)
            st.write(f"Rotterdam-style (≥2 of 3 criteria): **{'criteria met' if rotterdam else 'not met'}**")
            st.write(f"2023-guideline-style (irregular cycles + hyperandrogenism, or + PCOM/AMH): **{'criteria met' if g2023 else 'not met'}**")
            st.caption("Exclusion of other causes is required. This is a checklist aid, not a diagnosis.")

        with tab4:
            st.info("🧪 DEMO PAIR -- two images you upload are treated as visit 1 and visit 2. "
                    "If they are unrelated images this is NOT a real patient. Heuristic only (±10 score margin), not validated.")
            up2 = st.file_uploader("Upload visit 2 image", type=["png", "jpg", "jpeg"], key="visit2")
            if up2 is not None:
                r2 = analyze(np.array(Image.open(up2).convert("L")))
                delta = r2.fusion.risk_score - result.fusion.risk_score
                st.write(f"Visit 1 score: **{result.fusion.risk_score:.0f}** · Visit 2 score: **{r2.fusion.risk_score:.0f}** · Δ = **{delta:+.0f}**")
                st.write(f"Trend: **{trend_from_delta(delta)}**")
                st.caption("Scores are experimental and dataset-shortcut-contaminated; trend is illustrative only.")