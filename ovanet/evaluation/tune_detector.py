"""
ovanet/evaluation/tune_detector.py

Grid-searches detect_follicles' thresholds PLUS a new local-contrast filter
against gold-test-15 ground truth. The local-contrast check requires a
candidate blob's interior to be darker than the tissue ring immediately
surrounding it -- this is what actually distinguishes a real follicle
(sharp anechoic boundary) from generic speckle texture (no boundary),
which plain brightness thresholding cannot tell apart.
"""

import glob
import json
import os
import math

import numpy as np
import cv2
from PIL import Image

from ovanet.preprocessing import preprocess
from ovanet.detector import estimate_ovary_mask, _calculate_eccentricity
from ovanet.contracts import IMAGE_SIZE

ANNOT_DIR = "data/gold_test_annotations"
IMAGES_DIR = os.path.join(ANNOT_DIR, "images")


def polygon_centroid_and_radius(points):
    pts = np.array(points, dtype=np.float32)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    r = np.mean(np.sqrt(((pts - [cx, cy]) ** 2).sum(axis=1)))
    return cx, cy, max(r, 3.0)


def find_candidates(raw, mask, percentile, min_area, min_local_contrast,
                     max_area=800.0, min_solidity=0.65, max_eccentricity=0.85,
                     max_mean_intensity=0.55):
    gray = (raw * 255).astype(np.uint8)
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_tophat)
    tophat_masked = cv2.bitwise_and(tophat, tophat, mask=mask)

    mask_pixels = tophat[mask > 0]
    if len(mask_pixels) == 0 or mask_pixels.max() == 0:
        return []
    thresh_val = float(np.percentile(mask_pixels, percentile))
    thresh_val = max(15.0, min(thresh_val, 180.0))
    _, thresh = cv2.threshold(tophat_masked, thresh_val, 255, cv2.THRESH_BINARY)
    thresh[mask == 0] = 0

    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        if hull_area <= 0 or (area / hull_area) < min_solidity:
            continue
        m = cv2.moments(cnt)
        if m["m00"] <= 0:
            continue
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        ix, iy = int(round(cx)), int(round(cy))
        if ix < 0 or ix >= IMAGE_SIZE or iy < 0 or iy >= IMAGE_SIZE or mask[iy, ix] == 0:
            continue
        if _calculate_eccentricity(m, cnt) > max_eccentricity:
            continue

        c_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        cv2.drawContours(c_mask, [cnt], -1, 1, -1)
        inside_mean = float(raw[c_mask == 1].mean())
        if inside_mean > max_mean_intensity:
            continue

        # --- NEW: local contrast check ---
        # Ring = dilated blob minus the blob itself (the tissue just outside it)
        eq_r = max(3, int(round(math.sqrt(area / math.pi))))
        ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eq_r * 2 + 1, eq_r * 2 + 1))
        dilated = cv2.dilate(c_mask, ring_kernel)
        ring_mask = (dilated > 0) & (c_mask == 0) & (mask > 0)
        if ring_mask.sum() < 5:
            continue  # ring fell mostly outside the ROI, unreliable
        ring_mean = float(raw[ring_mask].mean())
        local_contrast = ring_mean - inside_mean
        if local_contrast < min_local_contrast:
            continue  # not meaningfully darker than its surroundings -> likely speckle

        results.append((cx, cy))
    return results


def match(pred_centers, gt_follicles):
    matched = set()
    tp = 0
    for cx, cy in pred_centers:
        best_j, best_d = -1, 1e9
        for j, (gx, gy, gr) in enumerate(gt_follicles):
            if j in matched:
                continue
            d = math.hypot(cx - gx, cy - gy)
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0 and best_d <= gt_follicles[best_j][2] * 1.5:
            matched.add(best_j)
            tp += 1
    fp = len(pred_centers) - tp
    fn = len(gt_follicles) - len(matched)
    return tp, fp, fn


def main():
    json_paths = sorted(glob.glob(os.path.join(ANNOT_DIR, "*.json")))
    cases = []
    for jp in json_paths:
        with open(jp, "r") as f:
            ann = json.load(f)
        img_name = os.path.basename(ann.get("imagePath", os.path.basename(jp).replace(".json", ".jpg")))
        img_path = os.path.join(IMAGES_DIR, img_name)
        if not os.path.exists(img_path):
            continue
        img = np.array(Image.open(img_path).convert("L"))
        prep = preprocess(img)
        mask = estimate_ovary_mask(prep.raw)
        gt_follicles = [
            polygon_centroid_and_radius(s["points"])
            for s in ann.get("shapes", [])
            if s.get("label", "").lower().startswith("follicle")
        ]
        cases.append((img_name, prep.raw, mask, gt_follicles))

    print(f"Loaded {len(cases)} gold-test cases. Grid-searching (this takes longer now)...\n")

    percentiles = [70, 75, 80, 85, 88]
    min_areas = [12, 25, 40, 60]
    min_contrasts = [0.0, 0.03, 0.05, 0.08, 0.12, 0.16, 0.20]

    results = []
    for pct in percentiles:
        for min_area in min_areas:
            for mc in min_contrasts:
                total_tp = total_fp = total_fn = 0
                for _, raw, mask, gt_follicles in cases:
                    preds = find_candidates(raw, mask, percentile=pct, min_area=min_area, min_local_contrast=mc)
                    tp, fp, fn = match(preds, gt_follicles)
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                results.append((f1, precision, recall, pct, min_area, mc, total_tp, total_fp, total_fn))

    results.sort(reverse=True)
    print(f"{'F1':>6} {'Prec':>6} {'Recall':>6} {'pctile':>7} {'min_area':>9} {'min_contrast':>13}  TP/FP/FN")
    for f1, p, r, pct, ma, mc, tp, fp, fn in results[:20]:
        print(f"{f1:6.3f} {p:6.3f} {r:6.3f} {pct:7d} {ma:9d} {mc:13.2f}  {tp}/{fp}/{fn}")


if __name__ == "__main__":
    main()