"""
tests/test_unet.py

Unit tests for U-Net architecture, embedding extraction, and dataset loader.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

# Skip U-Net tests in lightweight environments where training deps are not installed
pytest.importorskip("segmentation_models_pytorch")

from ovanet.contracts import IMAGE_SIZE, E_CNN_DIM
from ovanet.vision.unet import OvaNetUNet, build_unet
from ovanet.vision.dataset import OvaNetSegmentationDataset


def test_unet_initialization_and_forward():
    """Verify U-Net instantiates and outputs expected 2-channel logits."""
    model = build_unet(encoder_name="resnet18", encoder_weights=None, in_channels=1, classes=2)
    model.eval()

    dummy_input = torch.randn(2, 1, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (2, 2, IMAGE_SIZE, IMAGE_SIZE), f"Unexpected shape {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN values found in logits"


def test_unet_bottleneck_embedding_dim():
    """Verify bottleneck feature extraction matches contract E_CNN_DIM=512."""
    model = build_unet(encoder_name="resnet18", encoder_weights=None, in_channels=1, classes=2)
    model.eval()

    dummy_input = torch.randn(3, 1, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        embedding = model.extract_encoder_embedding(dummy_input)

    assert embedding.shape == (3, E_CNN_DIM), f"Expected shape (3, {E_CNN_DIM}), got {embedding.shape}"
    assert not torch.isnan(embedding).any()


def test_predict_masks_helper():
    """Verify inference helper returns correct types and shapes."""
    model = build_unet(encoder_name="resnet18", encoder_weights=None, in_channels=1, classes=2)
    model.eval()

    raw_img = np.random.uniform(0.0, 1.0, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    ovary_mask, follicle_mask, e_cnn = model.predict_masks(raw_img)

    assert ovary_mask.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert follicle_mask.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert e_cnn.shape == (E_CNN_DIM,)
    assert set(np.unique(ovary_mask)).issubset({0, 1})
    assert set(np.unique(follicle_mask)).issubset({0, 1})


def test_segmentation_dataset_item(tmp_path: Path):
    """Verify dataset correctly loads image, mask npz, and applies augmentations."""
    # Create a synthetic image and mask
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    import cv2
    img_path = img_dir / "sample_001.png"
    dummy_img = (np.random.rand(IMAGE_SIZE, IMAGE_SIZE) * 255).astype(np.uint8)
    cv2.imwrite(str(img_path), dummy_img)

    dummy_ovary = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    dummy_follicles = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    dummy_follicles[100:120, 100:120] = 1

    np.savez_compressed(
        mask_dir / "sample_001.npz",
        ovary=dummy_ovary,
        follicles=dummy_follicles,
    )

    dataset = OvaNetSegmentationDataset(
        manifest=[img_path],
        raw_root=img_dir,
        mask_dir=mask_dir,
        is_train=True,
    )

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["image"].shape == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert sample["mask"].shape == (2, IMAGE_SIZE, IMAGE_SIZE)
    assert sample["name"] == "sample_001"
