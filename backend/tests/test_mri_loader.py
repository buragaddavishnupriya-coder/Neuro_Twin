from pathlib import Path

import pandas as pd
import pytest

from app.preprocessing.mri_loader import (
    MRIDataLoadingError,
    load_mri_dataset,
)


def create_sample_dataset(tmp_path: Path) -> Path:
    dataset_path = tmp_path / "mri"

    for label in ["glioma", "meningioma", "notumor", "pituitary"]:
        class_directory = dataset_path / label
        class_directory.mkdir(parents=True)

        (class_directory / "image1.jpg").write_bytes(b"test")
        (class_directory / "image2.jpg").write_bytes(b"test")

    return dataset_path


def test_load_mri_dataset(tmp_path: Path) -> None:
    dataset_path = create_sample_dataset(tmp_path)

    data = load_mri_dataset(dataset_path)

    assert isinstance(data, pd.DataFrame)
    assert len(data) == 8
    assert set(data.columns) == {"image_path", "label"}
    assert set(data["label"]) == {
        "glioma",
        "meningioma",
        "notumor",
        "pituitary",
    }


def test_missing_dataset_raises_error(tmp_path: Path) -> None:
    with pytest.raises(MRIDataLoadingError):
        load_mri_dataset(tmp_path / "missing")


def test_empty_dataset_raises_error(tmp_path: Path) -> None:
    empty_dataset = tmp_path / "empty"
    empty_dataset.mkdir()

    with pytest.raises(MRIDataLoadingError):
        load_mri_dataset(empty_dataset)