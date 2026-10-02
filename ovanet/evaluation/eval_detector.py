"""
ovanet/evaluation/eval_detector.py

Scores ovanet.detector.detect_follicles() against the gold-test-15 Labelme
annotations (data/gold_test_annotations/*.json, label "ovary" for the ROI
polygon, label "follicle" for each follicle polygon).

Ovary mask -> Dice score vs GT polygon.
Follicle detection -> precision/recall via nearest-centroid matching
(a predicted follicle counts as a match if its center falls inside, or
within one GT-follicle-radius of, a GT follicle polygon).
"""

import glob
import json
import os

import numpy as np
import cv2
from PIL import Image

from ovanet.preprocessing import preprocess
from ovanet.detector import detect_follicles
from ovanet.contracts import IMAGE_SIZE

ANNOT_DIR = "data/gold_test_annotations"
IMAGES_DIR = os.path.join(ANNOT_DIR, "images")


def polygon_to_mask(points, shape=(IMAGE_SIZE, IMAGE_SIZE)):
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def dice(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    total = mask_a.sum() + mask_b.sum()
    return 2.0 * inter / total if total > 0 else float("nan")


def polygon_centroid_and_radius(points):
    pts = np.array(points, dtype=np.float32)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    r = np.mean(np.sqrt(((pts - [cx, cy]) ** 2).sum(axis=1)))
    return cx, cy, max(r, 3.0)


def match_follicles(pred_follicles, gt_follicles):
    """Greedy nearest-centroid matching. Returns (tp, fp, fn)."""
    matched_gt = set()
    tp = 0
    for pf in pred_follicles:
        best_j, best_d = -1, 1e9
        for j, (gx, gy, gr) in enumerate(gt_follicles):
            if j in matched_gt:
                continue
            d = np.hypot(pf.cx - gx, pf.cy - gy)
            if d < best_d:
                best_d, best_j = d, j
        # count as a match if within the GT follicle's own radius (generous but fair)
        if best_j >= 0 and best_d <= gt_follicles[best_j][2] * 1.5:
            matched_gt.add(best_j)
            tp += 1
    fp = len(pred_follicles) - tp
    fn = len(gt_follicles) - len(matched_gt)
    return tp, fp, fn


def main():
    json_paths = sorted(glob.glob(os.path.join(ANNOT_DIR, "*.json")))
    if not json_paths:
        print(f"No annotation JSON files found in {ANNOT_DIR}")
        return

    dice_scores = []
    total_tp = total_fp = total_fn = 0
    rows = []

    for jp in json_paths:
        with open(jp, "r") as f:
            ann = json.load(f)

        img_name = os.path.basename(ann.get("imagePath", os.path.basename(jp).replace(".json", ".jpg")))
        img_path = os.path.join(IMAGES_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"SKIP (image not found): {img_path}")
            continue

        img = np.array(Image.open(img_path).convert("L"))
        prep = preprocess(img)
        vision = detect_follicles(prep.raw)

        gt_ovary_mask = None
        gt_follicles = []
        for shape in ann.get("shapes", []):
            label = shape.get("label", "").lower()
            points = shape.get("points", [])
            if label == "ovary":
                gt_ovary_mask = polygon_to_mask(points)
            elif label.startswith("follicle"):
                gt_follicles.append(polygon_centroid_and_radius(points))

        d = dice(vision.ovary_mask, gt_ovary_mask) if gt_ovary_mask is not None else float("nan")
        tp, fp, fn = match_follicles(vision.follicles, gt_follicles)

        dice_scores.append(d)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        rows.append((img_name, d, len(gt_follicles), len(vision.follicles), tp, fp, fn))
        print(f"{img_name:30s}  ovary_dice={d:.3f}  GT_follicles={len(gt_follicles):2d}  "
              f"pred={len(vision.follicles):2d}  TP={tp} FP={fp} FN={fn}")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float("nan")
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
    mean_dice = float(np.nanmean(dice_scores))

    print("\n" + "=" * 60)
    print(f"Mean ovary Dice:     {mean_dice:.4f}")
    print(f"Follicle precision:  {precision:.4f}")
    print(f"Follicle recall:     {recall:.4f}")
    print(f"Follicle F1:         {f1:.4f}")
    print(f"(TP={total_tp}, FP={total_fp}, FN={total_fn} across {len(rows)} images)")


if __name__ == "__main__":
    main()