from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SIZE = (224, 224)


class MRITransformationError(Exception):
    """Raised when an MRI image cannot be transformed."""


def transform_mri_image(
    image_path: str | Path,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """
    Load and transform one MRI image.

    Returns:
        NumPy array with shape (height, width, 3)
        and pixel values in the range [0, 1].
    """

    path = Path(image_path)

    if not path.exists():
        raise MRITransformationError(f"Image not found: {path}")

    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = image.resize(image_size)
            image_array = np.asarray(image, dtype=np.float32) / 255.0
    except Exception as exc:
        raise MRITransformationError(
            f"Could not transform image: {path}"
        ) from exc

    if image_array.shape != (image_size[1], image_size[0], 3):
        raise MRITransformationError(
            f"Unexpected image shape: {image_array.shape}"
        )

    return image_array