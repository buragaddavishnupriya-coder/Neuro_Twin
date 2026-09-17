from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from app.models.mri.swin_transformer import (
    DEVICE,
    MRI_CLASSES,
    create_swin_model,
    save_swin_model,
)


IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 8


class MRIDataset(Dataset):
    """Dataset for MRI images arranged in class folders."""

    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, int]:
        image = Image.open(
            self.image_paths[index]
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, self.labels[index]


def collect_mri_samples(
    dataset_root: str | Path,
    split: str,
    max_samples_per_class: int | None = None,
) -> tuple[list[Path], list[int]]:
    """
    Collect MRI image paths and labels.

    If max_samples_per_class is provided, only that many
    images are collected from each class.
    """
    split_root = Path(dataset_root) / split

    if not split_root.exists():
        raise FileNotFoundError(
            f"MRI split directory not found: {split_root}"
        )

    image_paths: list[Path] = []
    labels: list[int] = []

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    for label, class_name in enumerate(MRI_CLASSES):
        class_dir = split_root / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"MRI class directory not found: {class_dir}"
            )

        class_images = [
            image_path
            for image_path in sorted(class_dir.iterdir())
            if (
                image_path.is_file()
                and image_path.suffix.lower()
                in valid_extensions
            )
        ]

        if max_samples_per_class is not None:
            class_images = class_images[
                :max_samples_per_class
            ]

        image_paths.extend(class_images)
        labels.extend(
            [label] * len(class_images)
        )

    if not image_paths:
        raise ValueError(
            f"No MRI images found in {split_root}"
        )

    return image_paths, labels
    
    

def create_mri_transforms(
    training: bool = True,
) -> transforms.Compose:
    """Create image preprocessing and augmentation transforms."""
    if training:
        return transforms.Compose(
            [
                transforms.Resize(
                    (IMAGE_SIZE, IMAGE_SIZE)
                ),
                transforms.RandomHorizontalFlip(
                    p=0.5
                ),
                transforms.RandomRotation(
                    degrees=10
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def create_mri_dataloaders(
    dataset_root: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
    max_samples_per_class: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create training and testing DataLoaders."""
    train_paths, train_labels = collect_mri_samples(
        dataset_root,
        "Training",
        max_samples_per_class=max_samples_per_class,
    )

    test_paths, test_labels = collect_mri_samples(
        dataset_root,
        "Testing",
        max_samples_per_class=max_samples_per_class,
    )

    train_dataset = MRIDataset(
        train_paths,
        train_labels,
        create_mri_transforms(training=True),
    )

    test_dataset = MRIDataset(
        test_paths,
        test_labels,
        create_mri_transforms(training=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader


def calculate_class_weights(
    labels: list[int],
) -> torch.Tensor:
    """Calculate inverse-frequency class weights."""
    counts = np.bincount(
        labels,
        minlength=len(MRI_CLASSES),
    )

    if np.any(counts == 0):
        raise ValueError(
            "Every MRI class must contain at least one image."
        )

    weights = len(labels) / (
        len(MRI_CLASSES) * counts
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train the model for one epoch."""
    model.train()

    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def evaluate_mri_model(
    model: nn.Module,
    loader: DataLoader,
) -> dict[str, Any]:
    """Evaluate the MRI model on a dataset."""
    model.eval()

    all_labels: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.numpy().tolist())
            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

    return {
        "accuracy": float(
            accuracy_score(
                all_labels,
                all_predictions,
            )
        ),
        "precision": float(
            precision_score(
                all_labels,
                all_predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                all_labels,
                all_predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                all_labels,
                all_predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            all_labels,
            all_predictions,
        ).tolist(),
        "classification_report": classification_report(
            all_labels,
            all_predictions,
            target_names=MRI_CLASSES,
            zero_division=0,
        ),
    }


def train_mri_model(
    dataset_root: str | Path,
    output_path: str | Path = (
        "artifacts/mri/swin_mri_model.pth"
    ),
    epochs: int = 5,
    batch_size: int = DEFAULT_BATCH_SIZE,
    pretrained: bool = True,
    max_samples_per_class: int | None = None,
) -> dict[str, Any]:
    """
    Train and evaluate the Swin Transformer MRI model.

    The best checkpoint is selected using validation accuracy.
    """

    train_paths, train_labels = collect_mri_samples(
        dataset_root,
        "Training",
        max_samples_per_class=max_samples_per_class,
    )

    train_loader, test_loader = create_mri_dataloaders(
        dataset_root,
        batch_size=batch_size,
        num_workers=0,
        max_samples_per_class=max_samples_per_class,
    )

    model = create_swin_model(
        num_classes=len(MRI_CLASSES),
        pretrained=pretrained,
        freeze_backbone=True,
    )

    class_weights = calculate_class_weights(
        train_labels
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = AdamW(
        trainable_parameters,
        lr=1e-3,
        weight_decay=1e-4,
    )

    history: list[dict[str, float]] = []

    best_accuracy = -1.0
    best_metrics: dict[str, Any] | None = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        metrics = evaluate_mri_model(
            model,
            test_loader,
        )

        accuracy = metrics["accuracy"]

        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "test_accuracy": accuracy,
            }
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Accuracy: {accuracy:.4f}"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_metrics = metrics

            save_swin_model(
                model,
                output_path,
            )

            print(
                f"Best checkpoint saved "
                f"with accuracy: {best_accuracy:.4f}"
            )

    if best_metrics is None:
        raise RuntimeError(
            "Training completed without producing metrics."
        )

    return {
        "device": str(DEVICE),
        "train_samples": len(train_paths),
        "test_samples": len(test_loader.dataset),
        "classes": MRI_CLASSES,
        "history": history,
        "metrics": best_metrics,
        "best_accuracy": best_accuracy,
        "model_path": str(output_path),
    }