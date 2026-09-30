"""
ovanet/vision/infer.py

Contract C2 — Unified vision inference module.
Implements analyze_vision(prep) -> VisionResult.

Gate G2 Resolution:
  - Segmentation method selected: "classical" (achieved higher Gold-Test Dice 0.8727 vs U-Net 0.0650)
  - E_cnn: Extracted from label-free ResNet18 encoder via Global Average Pooling (512-dim),
    strictly conforming to contract C2 (E_CNN_DIM = 512).
"""

from pathlib import Path
import numpy as np
import torch
import cv2

from ovanet.contracts import (
    IMAGE_SIZE,
    E_CNN_DIM,
    PreprocessResult,
    VisionResult,
    Follicle,
)
from ovanet.detector import detect_follicles

# Global cached encoder model for CPU inference (avoids re-instantiating on every image)
_ENCODER_MODEL = None


def get_encoder_model():
    """Lazy load cached ResNet18 encoder for label-free 512-dim embedding extraction."""
    global _ENCODER_MODEL
    if _ENCODER_MODEL is None:
        from ovanet.vision.unet import build_unet
        weights_path = Path("weights/unet_resnet18_best.pth")
        if weights_path.exists():
            model = build_unet(encoder_name="resnet18", encoder_weights=None, in_channels=1, classes=2)
            model.load_checkpoint(weights_path, map_location="cpu")
        else:
            model = build_unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=1, classes=2)
        model.eval()
        _ENCODER_MODEL = model
    return _ENCODER_MODEL


def extract_ecnn(raw: np.ndarray) -> np.ndarray:
    """
    Extract 512-dim bottleneck embedding E_cnn from raw grayscale image.

    Parameters
    ----------
    raw : float32 [256, 256] in [0, 1]

    Returns
    -------
    e_cnn : float32 [512]
    """
    try:
        model = get_encoder_model()
        if raw.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raw = cv2.resize(raw, (IMAGE_SIZE, IMAGE_SIZE))
        tensor = torch.from_numpy(raw).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            embedding = model.extract_encoder_embedding(tensor)
        return embedding.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception:
        # Fallback: deterministic pixel-hash embedding when smp/torch unavailable.
        # Uses a local RandomState to avoid polluting the global np.random state.
        import warnings
        warnings.warn(
            "smp/torch unavailable: E_cnn is a deterministic pixel-hash, "
            "not a real neural embedding. Downstream model quality may degrade.",
            RuntimeWarning,
            stacklevel=2,
        )
        seed = int(np.sum(raw[:10, :10]) * 1000) % (2**31)
        rng = np.random.RandomState(seed)
        return rng.randn(E_CNN_DIM).astype(np.float32)


def analyze_vision(prep: PreprocessResult) -> VisionResult:
    """
    Contract C2 authoritative entrypoint.

    Parameters
    ----------
    prep : PreprocessResult
        Result from C1 preprocess(image), containing .raw and .enh.

    Returns
    -------
    VisionResult
        Conforms to Contract C2.
    """
    raw = prep.raw
    enh = prep.enh

    # 1. Classical follicle & ovary detection (Winner of Gate G2)
    det_res = detect_follicles(raw, enh=enh)

    # 2. Extract label-free 512-dim encoder embedding E_cnn
    e_cnn = extract_ecnn(raw)

    return VisionResult(
        ovary_mask=det_res.ovary_mask,
        follicle_labels=det_res.follicle_labels,
        follicles=det_res.follicles,
        e_cnn=e_cnn,
        seg_method="classical",
        warnings=det_res.warnings,
    )
