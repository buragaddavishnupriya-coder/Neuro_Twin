from pathlib import Path

import torch
import torch.nn as nn
import timm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MRI_CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]


def create_swin_model(
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Create a Swin Transformer model for MRI classification.

    When freeze_backbone=True, only the classification head is trainable.
    """

    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

        # The Swin classifier is the final head.
        if hasattr(model, "head"):
            for parameter in model.head.parameters():
                parameter.requires_grad = True

    return model.to(DEVICE)


def save_swin_model(model: nn.Module, output_path: str) -> None:
    """Save model weights and class names."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": MRI_CLASSES,
        },
        output_path,
    )


def load_swin_model(
    model_path: str,
    num_classes: int = 4,
    pretrained: bool = False,
) -> nn.Module:
    """Load a saved Swin Transformer checkpoint."""

    model = create_swin_model(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=False,
    )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model