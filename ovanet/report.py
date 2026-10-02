"""
ovanet/report.py

Generates structured, professional clinical decision-support PDF reports
for ultrasound scans analyzed by the OvaNet AI pipeline.
Uses ReportLab to output in-memory bytes suitable for Streamlit downloads.
"""

import io
from datetime import datetime
import numpy as np
from PIL import Image

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
)

from ovanet.contracts import AnalysisResult
from ovanet.visualization import render_follicle_overlay


def generate_pdf_report(
    analysis: AnalysisResult,
    patient_id: str = "DEMO-PATIENT-001",
    scan_id: str = "US-SCAN-8842",
) -> bytes:
    """
    Builds a complete PDF report from an AnalysisResult object.

    Parameters
    ----------
    analysis : AnalysisResult
        Complete C8 pipeline analysis result.
    patient_id : str
        Subject identifier.
    scan_id : str
        Ultrasound scan study identifier.

    Returns
    -------
    bytes
        PDF document file content.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#4c1d95"),  # Deep Purple
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748b"),
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#94a3b8"),
    )

    elements = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>OvaNet AI</b> &bull; Clinical Decision Support", title_style),
            Paragraph(
                f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}<br/>"
                f"<b>Study ID:</b> {scan_id}<br/>"
                f"<b>Patient:</b> {patient_id}",
                subtitle_style,
            ),
        ]
    ]
    header_table = Table(header_data, colWidths=[340, 200])
    header_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    elements.append(header_table)

    # Divider bar
    divider = Table([[""]], colWidths=[540], rowHeights=[2])
    divider.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#7c3aed"))]))
    elements.append(divider)
    elements.append(Spacer(1, 10))

    # 2. Risk Assessment Card
    score = analysis.fusion.risk_score
    band = analysis.fusion.band

    if band == "HIGH":
        badge_bg = colors.HexColor("#fef2f2")
        badge_border = colors.HexColor("#ef4444")
        badge_text = colors.HexColor("#b91c1c")
    elif band == "MODERATE":
        badge_bg = colors.HexColor("#fffbeb")
        badge_border = colors.HexColor("#f59e0b")
        badge_text = colors.HexColor("#b45309")
    else:
        badge_bg = colors.HexColor("#ecfdf5")
        badge_border = colors.HexColor("#10b981")
        badge_text = colors.HexColor("#047857")

    risk_html = (
        f"<font size=11 color='#64748b'><b>CALIBRATED RISK SCORE:</b></font><br/>"
        f"<font size=28 color='#0f172a'><b>{score:.1f}</b></font> <font size=14 color='#94a3b8'>/ 100</font>"
    )
    band_html = (
        f"<font size=10 color='#64748b'><b>RISK BAND:</b></font><br/>"
        f"<font size=18 color='{badge_text.hexval()}'><b>{band} RISK</b></font><br/>"
        f"<font size=8 color='#64748b'>Model: {analysis.fusion.model_version}</font>"
    )

    risk_table = Table(
        [[Paragraph(risk_html, styles["Normal"]), Paragraph(band_html, styles["Normal"])]],
        colWidths=[270, 270],
    )
    risk_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), badge_bg),
            ("BOX", (0, 0), (-1, -1), 1.5, badge_border),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 16),
            ("RIGHTPADDING", (0, 0), (-1, -1), 16),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    elements.append(risk_table)
    elements.append(Spacer(1, 12))

    # 3. Visual Overlay Image
    # Render overlay from raw and vision/graph
    overlay_rgb = render_follicle_overlay(analysis.vision.ovary_mask, analysis.vision, analysis.graph)
    # Convert overlay to PNG in memory
    img_pil = Image.fromarray(overlay_rgb)
    img_buf = io.BytesIO()
    img_pil.save(img_buf, format="PNG")
    img_buf.seek(0)
    rl_img = RLImage(img_buf, width=190, height=190)

    # Quality and Key Findings summary next to image
    n_follicles = len(analysis.vision.follicles)
    periph_idx = analysis.graph.stats.get("peripheral_index", 0.0)
    clustering = analysis.graph.stats.get("clustering_coeff", 0.0)
    area_frac = analysis.graph.stats.get("follicle_area_fraction", 0.0)
    mean_dist = analysis.graph.stats.get("mean_nn_dist_rel", 0.0)

    findings_text = (
        f"<b>Quality Check:</b> {'PASS' if analysis.quality.ok else 'FLAGGED'}<br/>"
        f"<b>Blur Variance:</b> {analysis.quality.blur_var:.1f}<br/>"
        f"<b>Segmentation Method:</b> {analysis.vision.seg_method.upper()}<br/><br/>"
        f"<b>Follicles Detected:</b> {n_follicles} instances<br/>"
        f"<b>Peripheral Index:</b> {periph_idx:.3f} (elevated in PCOS)<br/>"
        f"<b>Clustering Coeff:</b> {clustering:.3f}<br/>"
        f"<b>Follicle Area Fraction:</b> {area_frac:.3f}<br/>"
        f"<b>Mean NN Distance (Rel):</b> {mean_dist:.3f}<br/>"
        f"<b>Latency:</b> {analysis.timings.get('total_s', 0.0):.2f}s"
    )

    vis_data = [
        [
            rl_img,
            Paragraph(
                f"<b><font size=11 color='#1e293b'>Image Analysis & Spatial Graph</font></b><br/><br/>"
                + findings_text,
                body_style,
            ),
        ]
    ]
    vis_table = Table(vis_data, colWidths=[200, 340])
    vis_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (1, 0), (1, -1), 14),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (0, -1), 8),
        ])
    )
    elements.append(vis_table)
    elements.append(Spacer(1, 12))

    # 4. Quantitative Biomarkers Table
    elements.append(Paragraph("<b>Quantitative Biomarkers & Features</b>", section_heading))

    rad = analysis.radiomics
    bio_data = [
        ["Biomarker Category", "Feature Name", "Extracted Value", "Reference Interpretation"],
        ["Morphological Graph", "Follicle Count (n)", f"{n_follicles}", "PCOS threshold often >= 12-20"],
        ["Morphological Graph", "Peripheral Index", f"{periph_idx:.3f}", "High values indicate subcapsular ring"],
        ["Morphological Graph", "Clustering Coefficient", f"{clustering:.3f}", "Local spatial follicle clustering"],
        ["Texture (Radiomics)", "Mean Grayscale Intensity", f"{rad.mean:.3f}", "Overall echogenicity level"],
        ["Texture (Radiomics)", "Shannon Entropy", f"{rad.entropy:.3f}", "PCOS stroma exhibits higher entropy"],
        ["Texture (Radiomics)", "GLCM Contrast", f"{rad.glcm_contrast:.3f}", "Local intensity variation"],
        ["Texture (Radiomics)", "GLCM Homogeneity", f"{rad.glcm_homogeneity:.3f}", "Smoothness of acoustic texture"],
    ]
    bio_table = Table(bio_data, colWidths=[130, 140, 90, 180])
    bio_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    elements.append(bio_table)
    elements.append(Spacer(1, 10))

    # 5. Feature Attribution Breakdown
    elements.append(Paragraph("<b>Explainability: Signal Family Attribution</b>", section_heading))
    attr = analysis.fusion.family_attribution
    attr_data = [["Feature Family", "Weight / Attribution", "Description"]]
    family_desc = {
        "radiomics": "Acoustic texture and entropy of ovarian tissue",
        "follicle": "Follicle density, size variance, and area fraction",
        "graph": "Spatial distribution (peripheral clustering vs uniform)",
        "cnn": "Deep representation from ResNet encoder backbone",
        "gnn": "Topological graph neural network embedding",
    }
    for fam, val in attr.items():
        attr_data.append([fam.capitalize(), f"{val * 100:.1f}%" if val <= 1.0 else f"{val:.1f}", family_desc.get(fam, "Learned signal")])

    attr_table = Table(attr_data, colWidths=[130, 130, 280])
    attr_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    elements.append(attr_table)
    elements.append(Spacer(1, 14))

    # 6. Medical Disclaimer & Sign-off
    disclaimer_text = (
        "<b>LEGAL & CLINICAL DISCLAIMER:</b> OvaNet AI is an experimental decision-support research prototype "
        "developed for HackSpire '26. It is not an FDA/CE cleared medical device, has not undergone prospective clinical trials, "
        "and is NOT intended to provide medical diagnosis, treatment, or clinical management recommendations. "
        "All quantitative parameters and automated risk indices must be independently evaluated and verified by a licensed radiologist or OB/GYN physician."
    )
    elements.append(
        KeepTogether([
            Paragraph(disclaimer_text, disclaimer_style),
            Spacer(1, 12),
            Table(
                [[
                    Paragraph("<b>Reviewing Clinician:</b> ___________________________", body_style),
                    Paragraph("<b>Signature:</b> ___________________________", body_style),
                    Paragraph("<b>Date:</b> ____________", body_style),
                ]],
                colWidths=[200, 220, 120],
            ),
        ])
    )

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
