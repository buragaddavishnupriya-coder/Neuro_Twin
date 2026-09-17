from pathlib import Path

import numpy as np
import pandas as pd

from app.preprocessing.mri_loader import load_mri_dataset
from app.preprocessing.mri_transformer import transform_mri_image
from app.preprocessing.mri_validator import validate_mri_dataset


def prepare_mri_dataset(
    dataset_path: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Validate and load an MRI dataset.

    Images are not loaded into memory here.
    Only their paths and labels are returned.
    """

    validation_report = validate_mri_dataset(dataset_path)

    if not validation_report["is_valid"]:
        raise ValueError(
            f"Invalid MRI dataset: {validation_report['errors']}"
        )

    data = load_mri_dataset(dataset_path)

    return data, validation_report


def load_transformed_mri_image(
    image_path: str | Path,
) -> np.ndarray:
    """Load and transform one MRI image."""

    return transform_mri_image(image_path)