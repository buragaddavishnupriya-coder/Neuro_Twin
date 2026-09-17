from pathlib import Path

import pandas as pd

from app.models.digital_twin.integration import (
    DigitalTwinIntegration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLINICAL_DATASET = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "clinical"
    / "Parkinsson disease.csv"
)

MRI_IMAGE = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "mri"
    / "Testing"
    / "glioma"
    / "Te-gl_1.jpg"
)


if __name__ == "__main__":
    print("Loading clinical and MRI models...")

    integration = DigitalTwinIntegration()

    raw_data = pd.read_csv(CLINICAL_DATASET)

    clinical_features = (
        raw_data.drop(
            columns=["name", "status"]
        )
        .iloc[0]
        .to_dict()
    )

    print("Running integrated patient assessment...")

    result = integration.assess_patient(
        patient_id="patient_001",
        clinical_features=clinical_features,
        mri_image_path=MRI_IMAGE,
        symptoms={
            "tremor": 0.6,
            "rigidity": 0.4,
        },
        notes="Initial integrated assessment",
    )

    print("\nClinical prediction:")
    print(result["clinical_prediction"])

    print("\nMRI prediction:")
    print(result["mri_prediction"])

    print("\nDigital Twin state:")
    print(result["patient"])