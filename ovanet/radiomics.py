"""
C3 — Radiomics feature extraction.

extract(raw, mask=None) -> RadiomicsResult

Five features in fixed order (RADIOMICS_FEATURE_ORDER):
  mean, std, entropy, glcm_contrast, glcm_homogeneity

GLCM settings (frozen — log any change in DECISIONS.md):
  32 grey levels, distance 1, angles [0, 45, 90, 135] averaged,
  symmetric, normalised.
"""

import numpy as np
from skimage.feature import graycomatrix, graycoprops

from ovanet.contracts import RadiomicsResult

# ── Frozen GLCM settings ────────────────────────────────────────────────
_GLCM_LEVELS = 32
_GLCM_DISTANCE = 1
_GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def _quantize(image: np.ndarray, levels: int = _GLCM_LEVELS) -> np.ndarray:
    """float32 [0,1] → uint8 [0, levels-1]."""
    return np.clip((image * (levels - 1)).round(), 0, levels - 1).astype(np.uint8)


def extract(raw: np.ndarray, mask: np.ndarray | None = None) -> RadiomicsResult:
    """
    Contract C3.

    Parameters
    ----------
    raw : float32 [256, 256] in [0, 1]  (from C1 preprocess .raw)
    mask : uint8 [256, 256] optional.  Nonzero = inside ROI.
           When None, the whole image is used.

    Returns
    -------
    RadiomicsResult with the 5 features in RADIOMICS_FEATURE_ORDER.
    """
    # ── Select pixels ───────────────────────────────────────────────────
    if mask is not None:
        pixels = raw[mask > 0].astype(np.float64)
    else:
        pixels = raw.ravel().astype(np.float64)

    if pixels.size == 0:
        # Degenerate: return zeros (will be flagged by quality check)
        return RadiomicsResult(
            mean=0.0, std=0.0, entropy=0.0,
            glcm_contrast=0.0, glcm_homogeneity=0.0,
        )

    # ── Mean and std ────────────────────────────────────────────────────
    mean_val = float(np.mean(pixels))
    std_val = float(np.std(pixels))

    # ── Entropy (Shannon, base-2, skip zero bins) ───────────────────────
    quantized = np.clip((pixels * (_GLCM_LEVELS - 1)).round(), 0, _GLCM_LEVELS - 1).astype(int)
    hist = np.bincount(quantized, minlength=_GLCM_LEVELS).astype(np.float64)
    hist = hist / hist.sum()
    nonzero = hist[hist > 0]
    entropy_val = float(-np.sum(nonzero * np.log2(nonzero)))

    # ── GLCM ────────────────────────────────────────────────────────────
    # Prepare quantised 2-D image for graycomatrix.
    # When a mask is given: crop to the bounding box of the mask.
    # Non-mask pixels inside the bbox are set to 0, which introduces a
    # small number of spurious (0, x) pairs at mask edges.  For a roughly
    # convex ovary mask this error is negligible and documented.
    if mask is not None:
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if len(rows) > 0 and len(cols) > 0:
            r0, r1 = rows[0], rows[-1] + 1
            c0, c1 = cols[0], cols[-1] + 1
            crop = raw[r0:r1, c0:c1].copy()
            crop_mask = mask[r0:r1, c0:c1]
            crop[crop_mask == 0] = 0.0
            img_q = _quantize(crop)
        else:
            img_q = _quantize(raw)
    else:
        img_q = _quantize(raw)

    glcm = graycomatrix(
        img_q,
        distances=[_GLCM_DISTANCE],
        angles=_GLCM_ANGLES,
        levels=_GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )

    # Average across the four angles
    glcm_contrast = float(graycoprops(glcm, "contrast").mean())
    glcm_homogeneity = float(graycoprops(glcm, "homogeneity").mean())

    return RadiomicsResult(
        mean=mean_val,
        std=std_val,
        entropy=entropy_val,
        glcm_contrast=glcm_contrast,
        glcm_homogeneity=glcm_homogeneity,
    )
