"""
app/ovanet_ui/components.py

Reusable, data-driven HTML builders for the OvaNet AI interface.
Presentation only: every function receives values that already exist in the
pipeline output and renders them. Missing data renders an empty state, never a guess.
"""

from html import escape
import math
import numpy as np
import streamlit as st

FAMILY_INFO = {
    "radiomics": ("Radiomics", ["Texture", "Intensity", "Homogeneity", "Contrast"]),
    "follicle": ("Follicle", ["Count", "Distribution", "Morphology"]),
    "graph": ("Graph", ["Spatial relationships", "Connectivity", "Distribution"]),
    "cnn": ("CNN", ["Learned image embedding"]),
    "gnn": ("GNN", ["Learned graph embedding"]),
}
FAMILY_ORDER = ["radiomics", "follicle", "graph", "cnn", "gnn"]
TREND_LABEL = {"IMPROVING": "Improving", "STABLE": "Stable", "WORSENING": "Worsening",
               "INSUFFICIENT_DATA": "Insufficient data"}
STAGES = ["Preprocessing", "Vision Analysis", "Radiomics", "Graph Analysis", "Fusion", "Risk Assessment"]


def html(s: str) -> None:
    """Render raw HTML. Lines are flattened so Markdown never treats indentation as a code block."""
    flat = " ".join(line.strip() for line in s.splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)


def e(x) -> str:
    return escape(str(x))


def fmt(v, nd=3) -> str:
    try:
        return f"{float(v):.{nd}f}" if v is not None and not math.isnan(float(v)) else "n/a"
    except (TypeError, ValueError):
        return "n/a"


# ----------------------------------------------------------------- navigation
LOGO = ('<svg width="30" height="30" viewBox="0 0 32 32" aria-hidden="true"><defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#4f7cff"/><stop offset=".5" stop-color="#22d3ee"/><stop offset="1" stop-color="#8b5cf6"/></linearGradient></defs>'
        '<ellipse cx="16" cy="16" rx="13" ry="10" fill="none" stroke="url(#lg)" stroke-width="1.8"/>'
        '<path d="M9 16 L16 10 L23 17 L15 22 Z" fill="none" stroke="url(#lg)" stroke-width="1" opacity=".7"/>'
        '<circle cx="9" cy="16" r="2.2" fill="#4f7cff"/><circle cx="16" cy="10" r="2.2" fill="#22d3ee"/>'
        '<circle cx="23" cy="17" r="2.2" fill="#8b5cf6"/><circle cx="15" cy="22" r="1.8" fill="#c4b5fd"/></svg>')

NAV_ITEMS = [("Overview", "overview"), ("Analysis", "analysis"), ("Insights", "insights"), ("Comparison", "comparison"), ("Safety", "safety")]

def nav() -> None:
    links = "".join(f'<a href="#{i}" target="_self">{t}</a>' for t, i in NAV_ITEMS)
    html(f"""<nav class="ov-nav" aria-label="Primary">
      <a class="ov-brand" href="#overview" target="_self">{LOGO}<span>Ova<b>Net</b> AI</span></a>
      <div class="ov-links">{links}</div>
      <div class="ov-right"><span class="ov-proto">Research Prototype</span>
        <a class="ov-cta" href="#analysis" target="_self">Start Analysis</a>
        <details class="ov-menu"><summary aria-label="Menu">&#9776;</summary><div>{links}</div></details>
      </div></nav>""")


# ----------------------------------------------------------------------- hero
def _card(x, y, text, cls):
    return (f'<g class="fl {cls}"><rect x="{x}" y="{y}" width="132" height="34" rx="11" fill="#0d1538" stroke="#3b4f9a" stroke-opacity=".7"/>'
            f'<circle cx="{x+14}" cy="{y+17}" r="4" fill="#22d3ee" class="nd"/>'
            f'<text x="{x+26}" y="{y+21}" font-size="10.5" fill="#dfe6ff">{text}</text></g>')


def hero_visual() -> str:
    stages = ["Ultrasound", "Ovary Region", "Follicles", "Spatial Graph", "AI Risk"]
    pipe = "".join(
        f'<circle cx="{60+i*100}" cy="440" r="7" fill="#0d1538" stroke="url(#hg)" stroke-width="2"/>'
        f'<circle cx="{60+i*100}" cy="440" r="2.5" fill="#22d3ee" class="nd"/>'
        f'<text x="{60+i*100}" y="466" font-size="10" text-anchor="middle" fill="#93a0c8">{s}</text>'
        for i, s in enumerate(stages))
    pipe_line = '<line x1="60" y1="440" x2="460" y2="440" stroke="url(#hg)" stroke-width="1.5" class="flow"/>'
    return f"""<div class="ov-herovis"><svg viewBox="0 0 520 490" role="img" aria-label="Illustration of the OvaNet analysis pipeline">
      <defs>
        <linearGradient id="hg" x1="0" x2="1"><stop offset="0" stop-color="#4f7cff"/><stop offset=".5" stop-color="#22d3ee"/><stop offset="1" stop-color="#8b5cf6"/></linearGradient>
        <radialGradient id="us" cx=".5" cy=".25" r=".9"><stop offset="0" stop-color="#27408f"/><stop offset=".6" stop-color="#101a45"/><stop offset="1" stop-color="#070b1f"/></radialGradient>
        <filter id="gl"><feGaussianBlur stdDeviation="6"/></filter>
      </defs>
      <ellipse cx="260" cy="215" rx="150" ry="140" fill="#4f7cff" opacity=".18" filter="url(#gl)"/>
      <path d="M95 128 L155 118 M95 205 L160 205 M95 282 L158 275 M425 160 L368 160 M425 255 L366 250" stroke="url(#hg)" stroke-width="1.2" class="flow" fill="none"/>
      <rect x="150" y="70" width="220" height="290" rx="24" fill="#080d2b" stroke="url(#hg)" stroke-opacity=".8"/>
      <path d="M260 88 L356 330 Q260 360 164 330 Z" fill="url(#us)" stroke="#3b4f9a" stroke-opacity=".6"/>
      <g fill="#9fb4ff" opacity=".35"><circle cx="230" cy="180" r="1.4"/><circle cx="285" cy="150" r="1.2"/><circle cx="300" cy="230" r="1.6"/>
        <circle cx="250" cy="260" r="1.3"/><circle cx="215" cy="230" r="1.1"/><circle cx="270" cy="205" r="1.3"/><circle cx="320" cy="285" r="1.2"/><circle cx="205" cy="290" r="1.4"/></g>
      <ellipse cx="262" cy="232" rx="68" ry="52" fill="none" stroke="#8b5cf6" stroke-width="1.4" stroke-dasharray="3 4" class="flow"/>
      <g stroke="#22d3ee" stroke-opacity=".55" stroke-width="1" fill="none">
        <path d="M232 214 L268 200 L296 232 L262 258 L226 244 Z"/><path d="M268 200 L262 258"/><path d="M232 214 L296 232"/></g>
      <g fill="#0b1236" stroke="#22d3ee" stroke-width="1.4"><circle cx="232" cy="214" r="7" class="nd"/><circle cx="268" cy="200" r="9" class="nd"/>
        <circle cx="296" cy="232" r="6" class="nd"/><circle cx="262" cy="258" r="8" class="nd"/><circle cx="226" cy="244" r="5" class="nd"/></g>
      <text x="260" y="346" font-size="9" fill="#6b78a3" text-anchor="middle" letter-spacing="1.5">ILLUSTRATIVE VISUALISATION</text>
      {_card(-4, 111, "Ovary region", "fl1")}{_card(-4, 188, "Follicles", "fl2")}{_card(-4, 265, "Radiomics", "fl3")}
      {_card(392, 143, "Graph", "fl4")}{_card(392, 235, "Risk assessment", "fl5")}
      {pipe_line}{pipe}
    </svg></div>"""


def hero() -> None:
    chips = "".join(f'<span class="ov-chip">{c}</span>' for c in
                    ["Image analysis", "Radiomics", "Follicle-level features", "Spatial relationships", "Graph-based analysis", "Explainable risk"])
    html(f"""<div id="overview" class="ov-anchor"></div>
    <section class="ov-hero"><div>
      <div class="ov-eyebrow">AI-assisted ovarian ultrasound analysis</div>
      <h1 class="ov-h1">From Pixels<br>to <span>Patterns.</span></h1>
      <div class="ov-sub">Explainable AI-assisted risk assessment from ovarian ultrasound imagery.</div>
      <p class="ov-desc">OvaNet combines image analysis, radiomics, follicle-level features and graph-based spatial
      relationships into one explainable risk estimate, built for research and decision support.</p>
      <div class="ov-chips">{chips}</div>
      <div class="ov-btns"><a class="ov-btn p" href="#analysis" target="_self">Start Analysis</a>
        <a class="ov-btn s" href="#how" target="_self">View How It Works</a></div>
      <div class="ov-resp">Research prototype &bull; Decision support only &bull; Not a diagnosis</div>
    </div>{hero_visual()}</section>""")


def how_it_works() -> None:
    items = [("Ultrasound", "Upload and quality-check the image."), ("Vision", "Ovary region and follicles are delineated."),
             ("Radiomics", "Texture and intensity features from the region."), ("Spatial graph", "Follicles become nodes linked by proximity."),
             ("Explainable risk", "Fusion model yields a score with family attribution."), ("Clinical review", "Rule-based checklist for the reviewer."),
             ("Tracking", "Compare two visits over time.")]
    cards = "".join(f'<div class="glass metric" style="animation-delay:{i*70}ms"><div class="l">0{i+1}</div><div class="v" style="font-size:1.1rem">{t}</div><div class="h">{d}</div></div>'
                    for i, (t, d) in enumerate(items))
    html(f'<div id="how" class="ov-anchor"></div><div class="ov-sec" style="margin-top:20px"><div class="k">How it works</div><h2>Ultrasound to explainable risk</h2></div>'
         f'<div class="grid g4">{cards}</div>')


# ------------------------------------------------------------ section header
def section(kicker: str, title: str, sub: str = "", anchor: str = "") -> None:
    a = f'<div id="{anchor}" class="ov-anchor"></div>' if anchor else ""
    s = f"<p>{sub}</p>" if sub else ""
    html(f'{a}<div class="ov-sec"><div class="k">{kicker}</div><h2>{title}</h2>{s}</div>')


def empty(msg: str) -> None:
    html(f'<div class="empty">{e(msg)}</div>')


# ------------------------------------------------------------------- generic
def metric_cards(items, cols=4) -> None:
    """items: (label, value, hint, optional bar 0..1)"""
    out = ""
    for i, it in enumerate(items):
        label, val, hint = it[0], it[1], it[2]
        bar = f'<div class="bar"><i style="width:{max(0,min(1,it[3]))*100:.0f}%"></i></div>' if len(it) > 3 and it[3] is not None else ""
        out += (f'<div class="glass metric" style="animation-delay:{i*60}ms"><div class="l">{e(label)}</div>'
                f'<div class="v">{e(val)}</div><div class="h">{e(hint)}</div>{bar}</div>')
    html(f'<div class="grid g{cols}">{out}</div>')


# ------------------------------------------------------------------ pipeline
def timeline(state: str = "idle", total_s=None) -> None:
    cls = {"idle": "", "running": "run", "done": "done"}[state]
    cells = "".join(f'<div class="s {cls}" style="animation-delay:{i*120}ms"><b>0{i+1}</b>{s}</div>' for i, s in enumerate(STAGES))
    cap = {"idle": "Awaiting an ultrasound image.", "running": "Running the OvaNet pipeline…",
           "done": f"Pipeline complete in {total_s:.2f}s." if total_s is not None else "Pipeline complete."}[state]
    html(f'<div class="tl" role="list">{cells}</div><div class="note">{cap}</div>')


def workflow_strip() -> None:
    html('<div class="steps"><span><b>1</b>Upload ultrasound</span><span><b>2</b>Analyze</span>'
         '<span><b>3</b>Review evidence</span><span><b>4</b>Understand risk</span></div>')


def status_strip(quality_ok: bool, reasons, has_mask: bool, n_fol: int, n_nodes: int, n_edges: int) -> None:
    q = "✓ Passed" if quality_ok else "⚠ Flagged"
    html(f"""<div class="strip">
      <div><small>Image quality</small><b>{q}</b></div>
      <div><small>Ovary region</small><b>{'✓ Segmented' if has_mask else '○ Not found'}</b></div>
      <div><small>Follicles</small><b>{'✓ ' + str(n_fol) if n_fol else '○ None detected'}</b></div>
      <div><small>Graph</small><b>{'✓ ' + str(n_nodes) + ' nodes · ' + str(n_edges) + ' edges' if n_nodes else '○ Not built'}</b></div></div>""")
    if not quality_ok and reasons:
        html(f'<span class="badge warn">Quality flags: {e(reasons)}</span>')


# ---------------------------------------------------------------------- risk
def risk_card(score: float, band: str, model_version: str = "") -> None:
    r, c = 88, 2 * math.pi * 88
    off = c * (1 - max(0, min(100, score)) / 100)
    col = {"LOW": "#2dd4bf", "MODERATE": "#fbbf24", "HIGH": "#fb7185"}.get(band, "#8b5cf6")
    ver = f'<span class="badge">Model {e(model_version)}</span>' if model_version else ""
    html(f"""<div class="glass risk"><svg viewBox="0 0 220 220" role="img" aria-label="Risk score {score:.0f} out of 100">
      <defs><linearGradient id="rg" x1="0" x2="1"><stop offset="0" stop-color="#4f7cff"/><stop offset="1" stop-color="{col}"/></linearGradient></defs>
      <circle cx="110" cy="110" r="{r}" fill="none" stroke="rgba(255,255,255,.08)" stroke-width="14"/>
      <circle cx="110" cy="110" r="{r}" fill="none" stroke="url(#rg)" stroke-width="14" stroke-linecap="round"
        stroke-dasharray="{c:.1f}" stroke-dashoffset="{off:.1f}" transform="rotate(-90 110 110)" style="filter:drop-shadow(0 0 8px {col}55)"/>
      <text x="110" y="118" text-anchor="middle" font-size="52" font-weight="800" fill="#e6eaff">{score:.0f}</text>
      <text x="110" y="144" text-anchor="middle" font-size="14" fill="#6b78a3">/ 100</text></svg>
      <div><div class="rs">RISK SCORE</div><div class="big">{score:.0f}<small> / 100</small></div>
      <span class="band {e(band)}">{e(band)}</span>{ver}
      <div class="disc">AI-generated risk estimate for research and decision-support purposes.<br>
      <b>This result is not a medical diagnosis and has not been clinically validated.</b></div></div></div>""")


def attribution(family_attr: dict, top_features) -> None:
    if not family_attr:
        empty("Feature-family attribution is not available for this analysis.")
        return
    mx = max((abs(v) for v in family_attr.values()), default=0) or 1.0
    rows = ""
    for fam in FAMILY_ORDER + [k for k in family_attr if k not in FAMILY_ORDER]:
        if fam not in family_attr:
            continue
        v = float(family_attr[fam])
        name, tags = FAMILY_INFO.get(fam, (fam.title(), []))
        chips = "".join(f'<span class="tag">{t}</span>' for t in tags)
        d = "raises estimate" if v > 0 else "lowers estimate" if v < 0 else "no contribution"
        rows += (f'<div class="attr"><div><div class="n">{e(name)}</div>{chips}</div>'
                 f'<div><div class="bar" style="height:10px"><i style="width:{abs(v)/mx*100:.0f}%"></i></div></div>'
                 f'<div class="d">{v:+.3f} · {d}</div></div>')
    html(f'<div class="glass"><b>Why this assessment?</b><div class="note" style="margin:2px 0 8px">Contribution of each feature family to the model’s log-odds (linear model: coefficient × standardised value). Right of centre raises the estimate, left lowers it. Families the model does not use show 0.</div>{rows}</div>')
    if top_features:
        tags = "".join(f'<span class="tag">{e(str(getattr(t, "name", t)).replace("rad_", "radiomics ").replace("graph_", "graph ").replace("e_gnn_", "GNN dim ").replace("_", " "))}' +
                       (f' · {float(t.contribution):+.3f}' if hasattr(t, "contribution") else "") + '</span>' for t in top_features)
        html(f'<div class="note">Top contributing features</div>{tags}')


def radiomics_cards(rad) -> None:
    items = [("Mean", rad.mean, "Average intensity", True), ("Std. deviation", rad.std, "Intensity spread", True),
             ("Entropy", rad.entropy, "Texture randomness", False), ("GLCM contrast", rad.glcm_contrast, "Local intensity variation", False),
             ("GLCM homogeneity", rad.glcm_homogeneity, "Texture uniformity", True)]
    cards = []
    for l, v, h, bounded in items:
        bar = min(1, max(0, float(v))) if bounded else None  # bar only where the value is naturally bounded to 0-1
        cards.append((l, fmt(v), h, bar))
    html('<div class="note" style="margin:0 0 8px">Texture &amp; intensity features</div>')
    metric_cards(cards, cols=5)


def follicle_section(follicles, stats: dict) -> None:
    if not follicles:
        empty("No follicles were detected in this image, so follicle-level statistics are not available.")
        return
    areas = np.array([f.area_px for f in follicles], dtype=float)
    radii = np.array([f.eq_radius_px for f in follicles], dtype=float)
    ecc = np.array([f.eccentricity for f in follicles], dtype=float)
    sol = np.array([f.solidity for f in follicles], dtype=float)
    metric_cards([("Follicle count", str(len(follicles)), "Detected in the ovary region"),
                  ("Mean radius", f"{radii.mean():.1f} px", "Equivalent radius (pixels)"),
                  ("Mean eccentricity", fmt(ecc.mean(), 2), "0 = circular"),
                  ("Mean solidity", fmt(sol.mean(), 2), "1 = convex")], 4)
    counts, edges = np.histogram(areas, bins=min(8, max(2, len(areas))))
    bars = "".join(f'<i title="{edges[i]:.0f}–{edges[i+1]:.0f} px²: {c}" style="height:{c/counts.max()*100:.0f}%"></i>' for i, c in enumerate(counts))
    sp = [("Mean NN distance (rel.)", stats.get("mean_nn_dist_rel")), ("NN distance SD (rel.)", stats.get("std_nn_dist_rel")),
          ("Peripheral index", stats.get("peripheral_index")), ("Size CV", stats.get("size_cv")),
          ("Area fraction", stats.get("follicle_area_fraction"))]
    pills = "".join(f'<span class="tag">{k}: {fmt(v)}</span>' for k, v in sp)
    html(f'<div class="glass" style="margin-top:14px"><b>Size distribution</b><div class="note" style="margin:2px 0 0">Follicle area histogram, px² (hover bars for ranges). Measurements are in pixels, not calibrated units.</div>'
         f'<div class="hist">{bars}</div><div style="margin-top:12px"><b>Spatial &amp; morphological statistics</b><br>{pills}</div></div>')


def graph_section(graph, stats_names, e_gnn=None) -> None:
    pills = "".join(f'<span class="tag">{k.replace("_", " ")}: {fmt(graph.stats.get(k))}</span>' for k in stats_names)
    html(f'<div class="glass"><b>Spatial relationships</b>'
         f'<p class="note" style="margin:6px 0 10px">OvaNet represents follicle relationships as a spatial graph to capture patterns that isolated image features may miss.</p>'
         f'<span class="badge ok">{graph.n} nodes</span><span class="badge ok">{len(graph.edges)} edges</span><div style="margin-top:8px">{pills}</div></div>')
    if e_gnn is not None and len(e_gnn):
        a = np.asarray(e_gnn, dtype=float)
        m = float(np.abs(a).max()) or 1.0
        cells = "".join(f'<i title="d{i}: {v:.3f}" style="background:rgba({"79,124,255" if v>=0 else "139,92,246"},{0.15+0.85*abs(v)/m:.2f})"></i>' for i, v in enumerate(a))
        html(f'<div class="glass" style="margin-top:14px"><b>GCN embedding</b><div class="note" style="margin:2px 0 0">{len(a)}-dimensional learned graph representation (hover cells for values).</div><div class="emb">{cells}</div></div>')


# ----------------------------------------------------------------- checklist
def checklist_cards(res: dict) -> None:
    cards = []
    for t in res["criteria_met"]:
        cards.append(("ok", "✓", t, "Criterion met"))
    for t in res["criteria_unknown"]:
        cards.append(("unk", "○", t, "Not entered"))
    if res.get("adolescent_caution"):
        cards.append(("warn", "⚠", "Adolescent caution", res["adolescent_caution"]))
    body = "".join(f'<div class="glass ck {c}" style="animation-delay:{i*70}ms"><div class="ic">{ic}</div><div><b>{e(t)}</b><small>{e(s)}</small></div></div>'
                   for i, (c, ic, t, s) in enumerate(cards))
    html(f'<div class="glass" style="margin:14px 0"><b>{e(res["summary"])}</b></div>'
         + (f'<div class="grid g2">{body}</div>' if body else "")
         + f'<div class="note">For clinician/researcher review — not a diagnostic conclusion. {e(res["disclaimer"])}</div>')


# ---------------------------------------------------------------- comparison
def visit_header(n: int, label: str) -> None:
    html(f'<div class="visit"><div class="t">VISIT 0{n}</div><div class="u">{label}</div></div>')


def visit_summary(res) -> None:
    metric_cards([("Risk score", f"{res.fusion.risk_score:.0f}", f"Band: {res.fusion.band}"),
                  ("Follicles", str(len(res.vision.follicles)), "Detected")], 2)


def comparison_result(cmp, r1, r2) -> None:
    d = cmp.deltas
    link = ('<svg viewBox="0 0 420 50" aria-hidden="true"><defs><linearGradient id="cg" x1="0" x2="1"><stop offset="0" stop-color="#4f7cff"/>'
            '<stop offset="1" stop-color="#8b5cf6"/></linearGradient></defs><line x1="30" y1="25" x2="390" y2="25" stroke="url(#cg)" stroke-width="2" class="flow"/>'
            f'<circle cx="30" cy="25" r="9" fill="#0b1236" stroke="#4f7cff" stroke-width="2"/><circle cx="390" cy="25" r="9" fill="#0b1236" stroke="#8b5cf6" stroke-width="2"/>'
            f'<text x="210" y="16" text-anchor="middle" font-size="12" fill="#93a0c8">{d.get("risk_score", 0):+.0f} risk points</text></svg>')
    rows = "".join(f'<span class="tag">{e(k.replace("_", " "))}: {v:+.2f}</span>' if isinstance(v, float) else f'<span class="tag">{e(k.replace("_", " "))}: {v:+d}</span>'
                   for k, v in d.items())
    demo = '<span class="badge warn">Demo pair — unless verified to be the same patient’s real scans</span>' if cmp.is_demo_pair else ""
    html(f'<div class="link">{link}</div><div class="glass"><span class="trend {e(cmp.trend)}">{TREND_LABEL.get(cmp.trend, cmp.trend)}</span> {demo}'
         f'<p style="color:#93a0c8;line-height:1.6;margin:12px 0 8px">{e(cmp.report_text)}</p><b>Changes</b><br>{rows}</div>')


# --------------------------------------------------------------- transparency
def transparency() -> None:
    pts = [("Research prototype", "Built to explore explainable imaging AI."), ("Not a diagnostic tool", "Outputs never constitute a diagnosis."),
           ("Not clinically validated", "No clinical validation is claimed."), ("Professional review", "AI output should be reviewed by qualified professionals."),
           ("Relative units", "Measurements are in pixels/relative units where calibration is unavailable.")]
    cards = "".join(f'<div class="glass" style="animation-delay:{i*70}ms"><b>{t}</b>{d}</div>' for i, (t, d) in enumerate(pts))
    section("Responsible AI", "Built for Explainability. Designed for Research.")
    html(f'<div class="tr">{cards}</div>')


def footer() -> None:
    html('<div class="foot"><span>OvaNet AI — research prototype. Decision support only.</span><span>Not a medical device.</span></div>')
