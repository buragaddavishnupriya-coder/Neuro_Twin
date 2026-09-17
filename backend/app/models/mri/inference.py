from pathlib import Path
from typing import Any

import torch
from PIL import Image

from app.models.mri.swin_transformer import (
    DEVICE,
    MRI_CLASSES,
    load_swin_model,
)
from app.models.mri.training import (
    create_mri_transforms,
)


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "mri"
    / "swin_mri_model.pth"
)


def load_mri_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
):
    """Load the trained Swin Transformer MRI model."""

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"MRI model not found: {model_path}"
        )

    return load_swin_model(
        model_path=str(model_path),
        num_classes=len(MRI_CLASSES),
        pretrained=False,
    )


def predict_mri(
    image_path: str | Path,
    model,
) -> dict[str, Any]:
    """Predict the MRI class and confidence for one image."""

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"MRI image not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    transform = create_mri_transforms(
        training=False
    )

    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(
            outputs,
            dim=1,
        )[0]

    predicted_index = int(
        probabilities.argmax().item()
    )

    predicted_class = MRI_CLASSES[predicted_index]
    confidence = float(
        probabilities[predicted_index].item()
    )

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            class_name: float(probability.item())
            for class_name, probability in zip(
                MRI_CLASSES,
                probabilities,
            )
        },
    }