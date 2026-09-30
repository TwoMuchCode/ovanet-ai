"""
ovanet/vision/unet.py

U-Net architecture for automated ovary and follicle segmentation.
Uses segmentation_models_pytorch with a ResNet18 encoder to provide:
  1. 2-channel segmentation:
     - Channel 0: Ovary ultrasound sector / tissue ROI
     - Channel 1: Follicle instances
  2. 512-dimensional bottleneck embedding (E_cnn) via Global Average Pooling
     on the deepest encoder stage, strictly matching contract C2 (E_CNN_DIM=512).
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

from ovanet.contracts import IMAGE_SIZE, E_CNN_DIM


class OvaNetUNet(nn.Module):
    """
    U-Net model with ResNet-18 encoder.
    Outputs:
      - segmentation masks (2 channels: ovary, follicles)
      - label-free encoder embedding (512-dim) for fusion
    """

    def __init__(
        self,
        encoder_name: str = "resnet18",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 1,
        classes: int = 2,
    ):
        super().__init__()
        if smp is None:
            raise ImportError(
                "segmentation-models-pytorch is required for OvaNetUNet. "
                "Install via 'pip install segmentation-models-pytorch' or from requirements-train.txt."
            )
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.classes = classes

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,  # Output raw logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning segmentation logits.

        Parameters
        ----------
        x : torch.Tensor of shape (B, in_channels, H, W)

        Returns
        -------
        logits : torch.Tensor of shape (B, classes, H, W)
        """
        return self.unet(x)

    def extract_encoder_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 512-dim bottleneck embedding E_cnn from the encoder using
        Global Average Pooling (GAP) on the final stage feature map.

        Parameters
        ----------
        x : torch.Tensor of shape (B, in_channels, H, W)

        Returns
        -------
        embedding : torch.Tensor of shape (B, E_CNN_DIM)
        """
        # smp encoder returns features from all stages: [stage0, stage1, stage2, stage3, stage4, stage5]
        # For ResNet18: stage5 has 512 channels, spatial resolution H/32 x W/32 (e.g., 8x8 for 256x256)
        features = self.unet.encoder(x)
        deepest = features[-1]  # shape: (B, 512, H_bot, W_bot)
        gap = torch.mean(deepest, dim=[-2, -1])  # shape: (B, 512)
        return gap

    @torch.no_grad()
    def predict_masks(
        self,
        raw: np.ndarray,
        device: torch.device | str = "cpu",
        threshold: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Inference helper on a single 2D raw image array float32 [0, 1].

        Returns
        -------
        ovary_mask : uint8 [256, 256] binary {0, 1}
        follicle_mask : uint8 [256, 256] binary {0, 1}
        e_cnn : float32 [512] embedding vector
        """
        self.eval()
        dev = torch.device(device)

        if raw.shape != (IMAGE_SIZE, IMAGE_SIZE):
            import cv2
            raw = cv2.resize(raw, (IMAGE_SIZE, IMAGE_SIZE))

        # Add batch & channel dimensions: (1, 1, 256, 256)
        tensor = torch.from_numpy(raw).float().unsqueeze(0).unsqueeze(0).to(dev)

        # Forward passes
        logits = self.unet(tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (2, 256, 256)
        embedding = self.extract_encoder_embedding(tensor).squeeze(0).cpu().numpy()

        ovary_mask = (probs[0] >= threshold).astype(np.uint8)
        follicle_mask = (probs[1] >= threshold).astype(np.uint8)

        return ovary_mask, follicle_mask, embedding.astype(np.float32)

    def save_checkpoint(self, path: str | Path, epoch: int | None = None, metrics: dict | None = None):
        """Save model weights and metadata to file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "encoder_name": self.encoder_name,
            "in_channels": self.in_channels,
            "classes": self.classes,
            "epoch": epoch,
            "metrics": metrics or {},
        }
        torch.save(payload, target)

    def load_checkpoint(self, path: str | Path, map_location: str = "cpu"):
        """Load weights from file."""
        ckpt = torch.load(path, map_location=map_location)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            self.load_state_dict(ckpt["state_dict"])
        else:
            self.load_state_dict(ckpt)


def build_unet(
    encoder_name: str = "resnet18",
    encoder_weights: str | None = None,
    in_channels: int = 1,
    classes: int = 2,
) -> OvaNetUNet:
    """Factory function for OvaNetUNet."""
    return OvaNetUNet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
