from pathlib import Path

from app.models.mri.inference import (
    load_mri_model,
    predict_mri,
)


MRI_ROOT = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "raw"
    / "mri"
)

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "mri"
    / "swin_mri_model.pth"
)


if __name__ == "__main__":
    image_path = next(
        (
            MRI_ROOT
            / "Testing"
            / "glioma"
        ).glob("*.jpg")
    )

    print(f"Loading model: {MODEL_PATH}")

    model = load_mri_model(MODEL_PATH)

    print(f"Testing image: {image_path}")

    result = predict_mri(
        image_path,
        model,
    )

    print("\nMRI prediction:")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")

    print("\nClass probabilities:")
    for class_name, probability in result["probabilities"].items():
        print(f"{class_name}: {probability:.4f}")