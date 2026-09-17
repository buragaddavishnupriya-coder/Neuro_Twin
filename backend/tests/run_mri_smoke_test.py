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
        output_path="artifacts/mri/swin_smoke_test.pth",
        epochs=1,
        batch_size=2,
        pretrained=True,
        max_samples_per_class=2,
    )

    print(results["metrics"])