import shutil
from pathlib import Path

from PIL import Image

from app.preprocessing.mri_validator import validate_mri_dataset


def create_sample_dataset(tmp_path: Path) -> Path:
    dataset_path = tmp_path / "mri"

    for label in ["glioma", "meningioma", "notumor", "pituitary"]:
        class_directory = dataset_path / label
        class_directory.mkdir(parents=True)

        image = Image.new("RGB", (64, 64), color="white")
        image.save(class_directory / "image1.jpg")

    return dataset_path


def test_valid_mri_dataset(tmp_path: Path) -> None:
    dataset_path = create_sample_dataset(tmp_path)

    report = validate_mri_dataset(dataset_path)

    assert report["is_valid"] is True
    assert report["total_images"] == 4
    assert report["class_counts"]["glioma"] == 1
    assert report["errors"] == []


def test_missing_class_is_reported(tmp_path: Path) -> None:
    dataset_path = create_sample_dataset(tmp_path)

    shutil.rmtree(dataset_path / "pituitary")

    report = validate_mri_dataset(dataset_path)

    assert report["is_valid"] is False
    assert "pituitary" in str(report["errors"])


def test_missing_dataset_is_reported(tmp_path: Path) -> None:
    report = validate_mri_dataset(tmp_path / "missing")

    assert report["is_valid"] is False
    assert len(report["errors"]) == 1