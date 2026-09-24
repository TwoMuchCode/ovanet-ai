"""
C1 — Preprocessing contract implementation.

preprocess(image) -> PreprocessResult
  RAW = grayscale + resize 256x256 + normalise to [0,1] float32
  ENH = grayscale + resize + bilateral denoise + CLAHE + normalise
  quality = blur check + brightness check

Input:  numpy array, uint8, HxW (grayscale) or HxWx3 (RGB/BGR).
Output: PreprocessResult (see ovanet.contracts).
"""

import numpy as np
import cv2

from ovanet.contracts import IMAGE_SIZE, PreprocessResult, QualityCheck

# Quality thresholds (configurable; documented here, not magic numbers)
_BLUR_THRESHOLD = 10.0      # Laplacian variance below this → "blurry"
_DARK_THRESHOLD = 5.0       # mean intensity below this → "too dark"
_BLANK_THRESHOLD = 1.0      # pixel std below this → "nearly blank"

# CLAHE settings (frozen — change requires DECISIONS.md entry)
_CLAHE_CLIP = 2.0
_CLAHE_GRID = (8, 8)

# Bilateral filter settings
_BILATERAL_D = 9
_BILATERAL_SIGMA_COLOR = 75
_BILATERAL_SIGMA_SPACE = 75


def _to_gray_uint8(image: np.ndarray) -> np.ndarray:
    """Convert any input to single-channel uint8 grayscale."""
    if image.dtype != np.uint8:
        # Assume float [0,1] → uint8
        image = np.clip(image * 255, 0, 255).astype(np.uint8)

    if image.ndim == 3:
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return image[:, :, 0]
    return image


def _quality_check(gray: np.ndarray) -> QualityCheck:
    """Assess image quality on the original-resolution grayscale frame."""
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    reasons: list[str] = []

    if blur_var < _BLUR_THRESHOLD:
        reasons.append(f"blurry (Laplacian var {blur_var:.1f} < {_BLUR_THRESHOLD})")
    if float(gray.mean()) < _DARK_THRESHOLD:
        reasons.append(f"too dark (mean {gray.mean():.1f})")
    if float(gray.std()) < _BLANK_THRESHOLD:
        reasons.append(f"nearly blank (std {gray.std():.1f})")

    return QualityCheck(ok=len(reasons) == 0, reasons=reasons, blur_var=blur_var)


def preprocess(image: np.ndarray) -> PreprocessResult:
    """
    Contract C1.

    Parameters
    ----------
    image : np.ndarray
        uint8 HxW or HxWx3 (BGR convention from cv2.imread, or RGB from PIL).

    Returns
    -------
    PreprocessResult with .raw, .enh (float32 256x256 in [0,1]) and .quality.
    """
    gray = _to_gray_uint8(image)

    # Quality check on the original resolution
    quality = _quality_check(gray)

    # Resize to 256 x 256 (squash; aspect-ratio distortion documented in
    # DECISIONS.md — images range from 1:1 to ~2:1, all have black borders)
    resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    # ── RAW variant ─────────────────────────────────────────────────────
    raw = resized.astype(np.float32) / 255.0

    # ── ENH variant ─────────────────────────────────────────────────────
    denoised = cv2.bilateralFilter(
        resized, _BILATERAL_D, _BILATERAL_SIGMA_COLOR, _BILATERAL_SIGMA_SPACE
    )
    clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP, tileGridSize=_CLAHE_GRID)
    enhanced = clahe.apply(denoised)
    enh = enhanced.astype(np.float32) / 255.0

    return PreprocessResult(raw=raw, enh=enh, quality=quality)
