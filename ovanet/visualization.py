"""
ovanet/visualization.py

Draws visible overlays for the step-by-step pipeline view:
- follicle markers (circles + id) on the preprocessed image
- kNN graph (nodes + edges) on the preprocessed image

Pure drawing helpers -- no model logic here.
"""

import numpy as np
from PIL import Image, ImageDraw


def _to_rgb(raw01: np.ndarray) -> Image.Image:
    """float32 [0,1] grayscale -> RGB PIL image we can draw colored markers on."""
    img_uint8 = np.clip(raw01 * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img_uint8).convert("RGB")


def draw_follicle_overlay(raw01: np.ndarray, follicles: list) -> Image.Image:
    """Draws a circle + id number at every detected follicle (cx, cy, eq_radius_px)."""
    img = _to_rgb(raw01)
    draw = ImageDraw.Draw(img)
    for f in follicles:
        x, y, r = f.cx, f.cy, max(f.eq_radius_px, 3.0)
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 70, 70), width=2)
        draw.text((x + r + 1, y - r), str(f.id), fill=(255, 220, 60))
    return img


def draw_graph_overlay(raw01: np.ndarray, follicles: list, graph) -> Image.Image:
    """Draws the kNN graph: edges as lines between follicle centers, nodes as dots."""
    img = _to_rgb(raw01)
    draw = ImageDraw.Draw(img)

    # position lookup: graph.nodes carries {id, x, y} in the same pixel space as follicles
    pos = {n.id: (n.x, n.y) for n in graph.nodes}

    # edges first, so nodes draw on top
    for e in graph.edges:
        if e.source in pos and e.target in pos:
            x1, y1 = pos[e.source]
            x2, y2 = pos[e.target]
            # thicker/brighter line = stronger kNN weight
            w = max(1, min(4, int(round(e.weight * 4))))
            draw.line([x1, y1, x2, y2], fill=(90, 200, 255), width=w)

    for node_id, (x, y) in pos.items():
        r = 4
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 220, 60), outline=(0, 0, 0))

    return img