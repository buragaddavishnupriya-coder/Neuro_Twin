from pathlib import Path

from app.models.mri.training import train_mri_model


MRI_ROOT = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "raw"
    / "mri"
)


if __name__ == "__main__":
    results = train_mri_model(
        dataset_root=MRI_ROOT,
        output_path="artifacts/mri/swin_mri_model.pth",
        epochs=5,
        batch_size=8,
        pretrained=True,
        max_samples_per_class=None,
    )

    print("\nTraining completed.")
    print(f"Device: {results['device']}")
    print(f"Training samples: {results['train_samples']}")
    print(f"Testing samples: {results['test_samples']}")
    print(f"Best accuracy: {results['best_accuracy']:.4f}")
    print(f"Model saved at: {results['model_path']}")

    print("\nClassification report:")
    print(results["metrics"]["classification_report"])

    print("Confusion matrix:")
    for row in results["metrics"]["confusion_matrix"]:
        print(row)