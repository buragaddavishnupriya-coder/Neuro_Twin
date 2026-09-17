from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.models.digital_twin.engine import (
    DigitalTwinEngine,
)
from app.models.digital_twin.state import (
    AssessmentRecord,
)
from app.models.mri.inference import (
    load_mri_model,
    predict_mri,
)
from app.models.parkinsons_inference import (
    load_xgboost_model,
    predict_clinical_risk,
)


DEFAULT_CLINICAL_MODEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "clinical"
    / "xgboost_model.joblib"
)

DEFAULT_MRI_MODEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "mri"
    / "swin_mri_model.pth"
)


class DigitalTwinIntegration:
    """Connect clinical and MRI predictions to a Digital Twin."""

    def __init__(
        self,
        clinical_model_path: str | Path = DEFAULT_CLINICAL_MODEL_PATH,
        mri_model_path: str | Path = DEFAULT_MRI_MODEL_PATH,
    ) -> None:
        self.engine = DigitalTwinEngine()

        self.clinical_model = load_xgboost_model(
            clinical_model_path
        )

        self.mri_model = load_mri_model(
            mri_model_path
        )

    def assess_patient(
        self,
        patient_id: str,
        clinical_features: dict[str, float],
        mri_image_path: str | Path,
        symptoms: dict[str, float] | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Run clinical and MRI assessment and update the twin."""

        clinical_result = predict_clinical_risk(
            clinical_features,
            self.clinical_model,
        )

        mri_result = predict_mri(
            mri_image_path,
            self.mri_model,
        )

        if patient_id not in self.engine.list_patients():
            self.engine.create_patient(patient_id)

        assessment = AssessmentRecord(
            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),
            clinical_risk=clinical_result["clinical_risk"],
            mri_risk=mri_result["confidence"],
            predicted_disease=clinical_result[
                "predicted_disease"
            ],
            symptoms=symptoms or {},
            notes=notes,
        )

        patient = self.engine.update_patient(
            patient_id,
            assessment,
        )

        return {
            "patient": patient.to_dict(),
            "clinical_prediction": clinical_result,
            "mri_prediction": mri_result,
        }