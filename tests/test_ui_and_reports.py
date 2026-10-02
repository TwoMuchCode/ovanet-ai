"""
tests/test_ui_and_reports.py

Unit tests verifying:
1. Clinical follicle overlay & side-by-side visualization.
2. In-memory PDF report generation using ReportLab.
"""

import numpy as np
from ovanet.contracts import IMAGE_SIZE
from ovanet.mocks import mock_analyze
from ovanet.visualization import render_follicle_overlay, render_side_by_side
from ovanet.report import generate_pdf_report


def test_render_follicle_overlay_shapes():
    analysis = mock_analyze()
    raw = np.random.rand(IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)

    overlay = render_follicle_overlay(raw, analysis.vision, analysis.graph)
    assert overlay.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert overlay.dtype == np.uint8


def test_render_side_by_side_shape():
    analysis = mock_analyze()
    raw = np.random.rand(IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)

    composite = render_side_by_side(raw, analysis.vision, analysis.graph)
    # 3 panels of width 256 + 2 border separators of width 3 = 256*3 + 6 = 774
    assert composite.shape == (IMAGE_SIZE, 774, 3)
    assert composite.dtype == np.uint8


def test_generate_pdf_report_valid_bytes():
    analysis = mock_analyze()
    pdf_bytes = generate_pdf_report(analysis, patient_id="TEST-PT-101", scan_id="SCAN-001")

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 5000
    # PDF specification magic header
    assert pdf_bytes.startswith(b"%PDF")
