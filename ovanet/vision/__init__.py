"""Vision package for OvaNet: classical detection, U-Net segmentation, and embeddings."""
from ovanet.detector import detect_follicles, estimate_ovary_mask

__all__ = ["detect_follicles", "estimate_ovary_mask"]
