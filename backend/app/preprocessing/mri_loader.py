from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class MRIDataLoadingError(Exception):
    """Raised when an MRI dataset cannot be loaded."""


def load_mri_dataset(dataset_path: str | Path) -> pd.DataFrame:
    """
    Load MRI image paths and labels from a directory structure.

    Expected structure:
        dataset_path/
            glioma/
            meningioma/
            notumor/
            pituitary/
    """

    root = Path(dataset_path)

    if not root.exists():
        raise MRIDataLoadingError(f"Dataset directory not found: {root}")

    if not root.is_dir():
        raise MRIDataLoadingError(f"Path is not a directory: {root}")

    records: list[dict[str, str]] = []

    for class_directory in sorted(root.iterdir()):
        if not class_directory.is_dir():
            continue

        label = class_directory.name

        for image_path in sorted(class_directory.iterdir()):
            if (
                image_path.is_file()
                and image_path.suffix.lower() in SUPPORTED_EXTENSIONS
            ):
                records.append(
                    {
                        "image_path": str(image_path),
                        "label": label,
                    }
                )

    if not records:
        raise MRIDataLoadingError(
            f"No supported MRI images found in: {root}"
        )

    return pd.DataFrame(records)