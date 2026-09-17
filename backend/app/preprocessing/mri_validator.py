from pathlib import Path

from PIL import Image

from app.preprocessing.mri_loader import SUPPORTED_EXTENSIONS


EXPECTED_CLASSES = {
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
}


def validate_mri_dataset(dataset_path: str | Path) -> dict:
    """
    Validate an MRI dataset organized into class folders.

    Returns a report containing errors, warnings, and image counts.
    """

    root = Path(dataset_path)

    report = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "class_counts": {},
        "total_images": 0,
    }

    if not root.exists():
        report["is_valid"] = False
        report["errors"].append(f"Dataset directory not found: {root}")
        return report

    if not root.is_dir():
        report["is_valid"] = False
        report["errors"].append(f"Path is not a directory: {root}")
        return report

    actual_classes = {
        directory.name
        for directory in root.iterdir()
        if directory.is_dir()
    }

    missing_classes = EXPECTED_CLASSES - actual_classes

    if missing_classes:
        report["is_valid"] = False
        report["errors"].append(
            f"Missing class folders: {sorted(missing_classes)}"
        )

    for class_name in sorted(actual_classes):
        class_directory = root / class_name

        image_files = [
            path
            for path in class_directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        report["class_counts"][class_name] = len(image_files)
        report["total_images"] += len(image_files)

        if len(image_files) == 0:
            report["warnings"].append(
                f"Class folder contains no supported images: {class_name}"
            )

        for image_path in image_files:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception:
                report["is_valid"] = False
                report["errors"].append(
                    f"Unreadable image: {image_path}"
                )

    return report