"""
ovanet/visualization.py

Clinical ultrasound visualization utilities:
- Follicle & ovary boundary overlays on ultrasound scans.
- Follicle spatial network graph visualization (nodes & weighted k-NN edges).
- Side-by-side composite views for clinical decision support.
"""

import cv2
import numpy as np
from ovanet.contracts import VisionResult, GraphResult, IMAGE_SIZE


def render_follicle_overlay(
    raw: np.ndarray,
    vision: VisionResult,
    graph: GraphResult = None,
    draw_ovary_contour: bool = True,
    draw_follicles: bool = True,
    draw_edges: bool = True,
) -> np.ndarray:
    """
    Renders an RGB clinical overlay on top of the raw ultrasound image.

    Parameters
    ----------
    raw : np.ndarray
        Grayscale float32 in [0, 1] or uint8 in [0, 255], shape (IMAGE_SIZE, IMAGE_SIZE).
    vision : VisionResult
        Detected ovary mask, follicle instances, and labels.
    graph : GraphResult, optional
        Spatial graph nodes, edges, and topological stats.

    Returns
    -------
    np.ndarray
        RGB uint8 image of shape (IMAGE_SIZE, IMAGE_SIZE, 3).
    """
    # 1. Base grayscale to RGB uint8
    if raw.dtype != np.uint8:
        base_uint8 = np.clip(raw * 255.0, 0, 255).astype(np.uint8)
    else:
        base_uint8 = raw.copy()

    if base_uint8.ndim == 2:
        rgb = cv2.cvtColor(base_uint8, cv2.COLOR_GRAY2RGB)
    else:
        rgb = base_uint8.copy()

    overlay = rgb.copy()

    # 2. Draw ovary ROI contour (Clean Emerald Green)
    if draw_ovary_contour and vision.ovary_mask is not None:
        ovary_uint8 = (vision.ovary_mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(ovary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Soft transparent fill for ovary region
        fill_layer = overlay.copy()
        cv2.drawContours(fill_layer, contours, -1, (16, 185, 129), cv2.FILLED)
        cv2.addWeighted(fill_layer, 0.12, overlay, 0.88, 0, overlay)
        # Sharp boundary contour
        cv2.drawContours(overlay, contours, -1, (52, 211, 153), 2, cv2.LINE_AA)

    # 3. Draw follicle spatial graph edges (Cyan/Sky blue)
    if draw_edges and graph is not None and graph.edges:
        edge_layer = overlay.copy()
        for edge in graph.edges:
            if edge.source < len(graph.nodes) and edge.target < len(graph.nodes):
                n1 = graph.nodes[edge.source]
                n2 = graph.nodes[edge.target]
                pt1 = (int(round(n1.x)), int(round(n1.y)))
                pt2 = (int(round(n2.x)), int(round(n2.y)))
                # Thickness and intensity proportional to spatial weight
                weight = float(np.clip(edge.weight, 0.2, 1.0))
                color = (int(56 * weight), int(189 * weight), int(248 * weight))
                cv2.line(edge_layer, pt1, pt2, color, 1, cv2.LINE_AA)
        cv2.addWeighted(edge_layer, 0.75, overlay, 0.25, 0, overlay)

    # 4. Draw follicle boundaries & centroids (Vibrant Coral / Gold)
    if draw_follicles and vision.follicles:
        follicle_fill = overlay.copy()
        for f in vision.follicles:
            cx, cy = int(round(f.cx)), int(round(f.cy))
            r = int(max(round(f.eq_radius_px), 2))

            # Translucent follicle fill
            cv2.circle(follicle_fill, (cx, cy), r, (236, 72, 153), cv2.FILLED)
        cv2.addWeighted(follicle_fill, 0.25, overlay, 0.75, 0, overlay)

        for f in vision.follicles:
            cx, cy = int(round(f.cx)), int(round(f.cy))
            r = int(max(round(f.eq_radius_px), 2))
            # Follicle outer ring (Rose-Pink)
            cv2.circle(overlay, (cx, cy), r, (244, 63, 94), 1, cv2.LINE_AA)
            # Follicle centroid (Amber/Yellow dot)
            cv2.circle(overlay, (cx, cy), 2, (251, 191, 36), cv2.FILLED, cv2.LINE_AA)

    return overlay


def render_side_by_side(raw: np.ndarray, vision: VisionResult, graph: GraphResult) -> np.ndarray:
    """
    Renders a 3-panel horizontal comparison:
    [Original Ultrasound Scan | Ovary ROI Sector | Follicle Graph Network]
    """
    # 1. Raw grayscale as RGB
    if raw.dtype != np.uint8:
        raw_u8 = np.clip(raw * 255.0, 0, 255).astype(np.uint8)
    else:
        raw_u8 = raw.copy()
    raw_rgb = cv2.cvtColor(raw_u8, cv2.COLOR_GRAY2RGB)

    # 2. Mask panel: Dark background with green ROI
    mask_rgb = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    if vision.ovary_mask is not None:
        mask_rgb[vision.ovary_mask > 0] = (16, 185, 129)
    if vision.follicle_labels is not None:
        mask_rgb[vision.follicle_labels > 0] = (244, 63, 94)

    # 3. Full overlay with graph network
    overlay_rgb = render_follicle_overlay(raw, vision, graph)

    # Stack side-by-side with border separator
    border = np.zeros((IMAGE_SIZE, 3, 3), dtype=np.uint8) + 40
    composite = np.hstack([raw_rgb, border, mask_rgb, border, overlay_rgb])
    return composite
