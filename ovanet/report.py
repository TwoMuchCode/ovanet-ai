"""
ovanet/report.py

PDF export of a single analysis result: risk score, key features, SHAP-style
family attribution, and disclaimers. Uses ReportLab.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime
import io


def generate_pdf_report(result, patient_label: str = "Anonymous") -> bytes:
    """
    result: AnalysisResult from ovanet.pipeline.analyze()
    Returns PDF bytes (can be written to file or served directly in Streamlit).
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], textColor=colors.HexColor("#7c3aed"))
    disclaimer_style = ParagraphStyle("Disclaimer", parent=styles["Normal"], textColor=colors.red, fontSize=9)

    elements = []

    elements.append(Paragraph("OvaNet AI — PCOS Risk Assessment Report", title_style))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    elements.append(Paragraph(f"Reference: {patient_label}", styles["Normal"]))
    elements.append(Spacer(1, 14))

    elements.append(Paragraph(
        "⚠️ DECISION-SUPPORT RESEARCH PROTOTYPE. NOT A DIAGNOSIS. NOT CLINICALLY VALIDATED. "
        "NOT A MEDICAL DEVICE. This report must be reviewed by a qualified clinician.",
        disclaimer_style
    ))
    elements.append(Spacer(1, 16))

    elements.append(Paragraph("Risk Assessment", styles["Heading2"]))
    risk_data = [
        ["Risk Score", f"{result.fusion.risk_score:.0f} / 100"],
        ["Risk Band", result.fusion.band],
        ["Model Version", result.fusion.model_version],
    ]
    risk_table = Table(risk_data, colWidths=[2.2 * inch, 3 * inch])
    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f3f4f6")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(risk_table)
    elements.append(Spacer(1, 16))

    elements.append(Paragraph("Radiomics Features (F_k)", styles["Heading2"]))
    rad_data = [
        ["Feature", "Value"],
        ["Mean", f"{result.radiomics.mean:.4f}"],
        ["Std", f"{result.radiomics.std:.4f}"],
        ["Entropy", f"{result.radiomics.entropy:.4f}"],
        ["GLCM Contrast", f"{result.radiomics.glcm_contrast:.4f}"],
        ["GLCM Homogeneity", f"{result.radiomics.glcm_homogeneity:.4f}"],
    ]
    rad_table = Table(rad_data, colWidths=[2.5 * inch, 2 * inch])
    rad_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7c3aed")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    elements.append(rad_table)
    elements.append(Spacer(1, 16))

    elements.append(Paragraph("Follicle Graph Statistics", styles["Heading2"]))
    graph_data = [["Statistic", "Value"]]
    for k, v in result.graph.stats.items():
        graph_data.append([k.replace("_", " ").title(), f"{v:.3f}"])
    graph_table = Table(graph_data, colWidths=[2.8 * inch, 1.7 * inch])
    graph_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7c3aed")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    elements.append(graph_table)
    elements.append(Spacer(1, 16))

    elements.append(Paragraph("Feature-Family Attribution", styles["Heading2"]))
    attr_data = [["Family", "Contribution"]] + [
        [k.title(), f"{v:.2f}"] for k, v in result.fusion.family_attribution.items()
    ]
    attr_table = Table(attr_data, colWidths=[2.8 * inch, 1.7 * inch])
    attr_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7c3aed")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    elements.append(attr_table)
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Known Limitations", styles["Heading2"]))
    limitations = (
        "This dataset has a confirmed metadata-based shortcut (a classifier using only image "
        "size/brightness/contrast achieves AUROC=1.0), indicating possible confounding between "
        "image acquisition conditions and labels. Risk scores should be interpreted with caution "
        "pending further validation on external data. No mm-calibration is available; follicle "
        "sizes are relative, not absolute."
    )
    elements.append(Paragraph(limitations, styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer.read()