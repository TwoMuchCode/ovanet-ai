"""
C2 — Follicle detection and ovary segmentation (classical approach).

detect_follicles(raw, enh=None, ovary_mask=None) -> VisionResult

Algorithm:
  1. If ovary_mask is None, estimate ultrasound sector / tissue ROI:
     - Clear the top 9% banner and bottom 5% margin to exclude machine text/metadata
     - Sector thresholding + morphological closing to get the scan fan/sector
     - Select largest connected component
  2. Morphological Black Top-Hat on enhanced/raw image with an elliptical
     kernel to isolate dark, anechoic circular fluid-filled follicles.
  3. Adaptive/percentile thresholding restricted to the ovary tissue ROI.
  4. Morphological opening to suppress speckle noise.
  5. Contour extraction and geometric filtering:
     - Area within [12, 800] pixels (matching follicle scale at 256x256)
     - Solidity >= 0.65 (compact rounded shape)
     - Eccentricity <= 0.85 (rejects linear border / edge artifacts)
     - Mean intensity <= 0.55 (ensures hypoechoic / dark fluid region)
  6. Outputs VisionResult (ovary_mask, follicle_labels, follicles, e_cnn, seg_method="classical")
"""

import math
import numpy as np
import cv2

from ovanet.contracts import (
    IMAGE_SIZE,
    E_CNN_DIM,
    Follicle,
    VisionResult,
)


def estimate_ovary_mask(raw: np.ndarray) -> np.ndarray:
    """
    Estimate ultrasound sector / ovary tissue ROI.
    Returns uint8 binary mask of shape (IMAGE_SIZE, IMAGE_SIZE) with values {0, 1}.
    """
    gray = (np.clip(raw, 0.0, 1.0) * 255).astype(np.uint8)
    h, w = gray.shape

    # Ultrasound machines place patient metadata/hospital text in the top 10%
    # and calibration bars in the bottom 5%. Suppress those to find the actual scan fan.
    scan_roi = gray.copy()
    scan_roi[: int(h * 0.09), :] = 0
    scan_roi[int(h * 0.95) :, :] = 0

    # Sector threshold (ultrasound tissue backscatter is > ~15/255)
    _, sector = cv2.threshold(scan_roi, 15, 255, cv2.THRESH_BINARY)

    # Morphological closing to join tissue and bridge dark internal follicles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    sector_closed = cv2.morphologyEx(sector, cv2.MORPH_CLOSE, kernel)

    # Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sector_closed)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8)
    else:
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        cv2.circle(mask, (IMAGE_SIZE // 2, IMAGE_SIZE // 2), int(IMAGE_SIZE * 0.38), 1, -1)

    # If mask is degenerate (< 5% of image), fall back to central region
    if mask.sum() < (IMAGE_SIZE * IMAGE_SIZE * 0.05):
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        cv2.circle(mask, (IMAGE_SIZE // 2, IMAGE_SIZE // 2), int(IMAGE_SIZE * 0.38), 1, -1)

    return mask


def _calculate_eccentricity(moments: dict, contour: np.ndarray) -> float:
    """Calculate eccentricity from fitted ellipse or second-order moments."""
    if len(contour) >= 5:
        try:
            (_, _), (ma, ma_maj), _ = cv2.fitEllipse(contour)
            minor = min(ma, ma_maj)
            major = max(ma, ma_maj)
            if major > 1e-4:
                ratio = minor / major
                return float(math.sqrt(max(0.0, 1.0 - ratio * ratio)))
        except Exception:
            pass

    # Moment-based approximation:
    mu20 = moments.get("mu20", 0.0)
    mu02 = moments.get("mu02", 0.0)
    mu11 = moments.get("mu11", 0.0)
    delta = math.sqrt(max(0.0, (mu20 - mu02) ** 2 + 4 * (mu11 ** 2)))
    lambda1 = (mu20 + mu02 + delta) / 2.0
    lambda2 = max(0.0, (mu20 + mu02 - delta) / 2.0)
    if lambda1 > 1e-4:
        return float(math.sqrt(max(0.0, 1.0 - lambda2 / lambda1)))
    return 0.0


def detect_follicles(
    raw: np.ndarray,
    enh: np.ndarray | None = None,
    ovary_mask: np.ndarray | None = None,
) -> VisionResult:
    """
    Contract C2 classical follicle detection.

    Parameters
    ----------
    raw : float32 [256, 256] in [0, 1]
    enh : float32 [256, 256] in [0, 1], optional (contrast-enhanced)
    ovary_mask : uint8 [256, 256] optional ({0, 1}). If None, estimated automatically.

    Returns
    -------
    VisionResult
    """
    warnings: list[str] = []

    # Ensure shape and types
    if raw.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raw = cv2.resize(raw, (IMAGE_SIZE, IMAGE_SIZE))
    raw = np.clip(raw, 0.0, 1.0).astype(np.float32)

    if enh is None:
        enh = raw
    elif enh.shape != (IMAGE_SIZE, IMAGE_SIZE):
        enh = cv2.resize(enh, (IMAGE_SIZE, IMAGE_SIZE))
    enh = np.clip(enh, 0.0, 1.0).astype(np.float32)

    # 1. Estimate or validate ovary mask
    if ovary_mask is None:
        mask = estimate_ovary_mask(raw)
    else:
        mask = (ovary_mask > 0).astype(np.uint8)
        if mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
            mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

    # 2. Black Top-Hat on enhanced image
    # Follicles are dark round regions, diameter typically 6 to 30 pixels (radius 3 to 15)
    gray = (enh * 255).astype(np.uint8)
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_tophat)

    # Restrict to ovary mask
    tophat_masked = cv2.bitwise_and(tophat, tophat, mask=mask)

    # 3. Dynamic percentile threshold within mask
    mask_pixels = tophat[mask > 0]
    if len(mask_pixels) > 0 and mask_pixels.max() > 0:
        thresh_val = float(np.percentile(mask_pixels, 88))
        thresh_val = max(15.0, min(thresh_val, 180.0))
        _, thresh = cv2.threshold(tophat_masked, thresh_val, 255, cv2.THRESH_BINARY)
    else:
        thresh = cv2.adaptiveThreshold(
            tophat_masked,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=21,
            C=-3,
        )
    thresh[mask == 0] = 0

    # 4. Morphological opening to clean single-pixel speckles
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)

    # 5. Extract contours and filter
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    follicles: list[Follicle] = []
    follicle_labels = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32)

    follicle_id = 1
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        # Follicle area bounds in 256x256 space: ~12 to 800 px
        if area < 12.0 or area > 800.0:
            continue

        # Solidity
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        if hull_area <= 0:
            continue
        solidity = area / hull_area
        if solidity < 0.65:
            continue

        # Centroid
        m = cv2.moments(cnt)
        if m["m00"] <= 0:
            continue
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])

        # Check inside ovary mask
        ix, iy = int(round(cx)), int(round(cy))
        if ix < 0 or ix >= IMAGE_SIZE or iy < 0 or iy >= IMAGE_SIZE or mask[iy, ix] == 0:
            continue

        # Eccentricity
        eccentricity = _calculate_eccentricity(m, cnt)
        if eccentricity > 0.85:
            continue

        # Mean intensity inside follicle in raw image
        c_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        cv2.drawContours(c_mask, [cnt], -1, 1, -1)
        mean_intensity = float(raw[c_mask == 1].mean())

        # Follicles are hypoechoic / dark fluid
        if mean_intensity > 0.55:
            continue

        eq_radius = float(math.sqrt(area / math.pi))

        follicle_obj = Follicle(
            id=follicle_id,
            cx=round(cx, 2),
            cy=round(cy, 2),
            area_px=round(area, 2),
            eq_radius_px=round(eq_radius, 2),
            eccentricity=round(eccentricity, 4),
            solidity=round(solidity, 4),
            mean_intensity=round(mean_intensity, 4),
        )
        follicles.append(follicle_obj)
        follicle_labels[c_mask == 1] = follicle_id
        follicle_id += 1

    if len(follicles) == 0:
        warnings.append("No follicles detected with classical morphology.")

    # 6. E_cnn: PLACEHOLDER — deterministic pixel-hash embedding.
    #    This is NOT a real neural network embedding. It fills the contract shape
    #    so that callers who only need follicle/ovary data can ignore it.
    #    The authoritative E_cnn is produced by analyze_vision() → extract_ecnn()
    #    which runs the ResNet-18 encoder (see ovanet/vision/infer.py).
    e_cnn = np.zeros(E_CNN_DIM, dtype=np.float32)
    block_8x8 = cv2.resize(raw, (8, 8), interpolation=cv2.INTER_AREA).ravel()
    e_cnn[:64] = block_8x8
    block_16x16 = cv2.resize(raw, (16, 16), interpolation=cv2.INTER_AREA).ravel()
    e_cnn[64 : 64 + 256] = block_16x16
    e_cnn[320] = float(len(follicles))
    if len(follicles) > 0:
        e_cnn[321] = float(np.mean([f.area_px for f in follicles]))
        e_cnn[322] = float(np.std([f.area_px for f in follicles]))

    return VisionResult(
        ovary_mask=mask,
        follicle_labels=follicle_labels,
        follicles=follicles,
        e_cnn=e_cnn,
        seg_method="classical",
        warnings=warnings,
    )
