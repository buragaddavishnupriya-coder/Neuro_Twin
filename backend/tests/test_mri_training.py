from pathlib import Path

import torch

from app.models.mri.swin_transformer import MRI_CLASSES
from app.models.mri.training import (
    collect_mri_samples,
    create_mri_dataloaders,
    create_mri_transforms,
)


MRI_ROOT = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "raw"
    / "mri"
)


def test_collect_mri_samples():
    image_paths, labels = collect_mri_samples(
        MRI_ROOT,
        "Training",
    )

    assert len(image_paths) == 5600
    assert len(labels) == 5600
    assert set(labels) == set(range(len(MRI_CLASSES)))


def test_mri_dataloader_batch_shape():
    train_loader, test_loader = create_mri_dataloaders(
        MRI_ROOT,
        batch_size=2,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))

    assert images.shape == (2, 3, 224, 224)
    assert labels.shape == (2,)
    assert images.dtype == torch.float32
    assert len(test_loader.dataset) == 1600


def test_mri_transforms():
    transform = create_mri_transforms(
        training=False
    )

    assert transform is not None