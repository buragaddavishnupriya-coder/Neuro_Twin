from pathlib import Path

import numpy as np
from PIL import Image

from app.preprocessing.mri_transformer import (
    IMAGE_SIZE,
    MRITransformationError,
    transform_mri_image,
)


def create_sample_image(tmp_path: Path) -> Path:
    image_path = tmp_path / "sample.jpg"

    image = Image.new("L", (100, 80), color=128)
    image.save(image_path)

    return image_path


def test_transform_mri_image(tmp_path: Path) -> None:
    image_path = create_sample_image(tmp_path)

    transformed = transform_mri_image(image_path)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == (224, 224, 3)
    assert transformed.dtype == np.float32
    assert transformed.min() >= 0.0
    assert transformed.max() <= 1.0


def test_transform_preserves_custom_size(tmp_path: Path) -> None:
    image_path = create_sample_image(tmp_path)

    transformed = transform_mri_image(
        image_path,
        image_size=(128, 128),
    )

    assert transformed.shape == (128, 128, 3)


def test_missing_image_raises_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.jpg"

    try:
        transform_mri_image(missing_path)
        assert False, "Expected MRITransformationError"
    except MRITransformationError:
        assert True