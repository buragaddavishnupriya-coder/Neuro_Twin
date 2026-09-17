from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class AssessmentRecord:
    """One clinical or MRI assessment of a patient."""

    timestamp: str
    clinical_risk: float | None = None
    mri_risk: float | None = None
    predicted_disease: str | None = None
    symptoms: dict[str, float] = field(default_factory=dict)
    notes: str | None = None


@dataclass
class PatientState:
    """
    Current patient state maintained by the Digital Twin.

    This is a research representation of patient information.
    """

    patient_id: str
    age: int | None = None
    sex: str | None = None

    clinical_risk: float = 0.0
    mri_risk: float = 0.0
    combined_risk: float = 0.0

    predicted_disease: str | None = None
    disease_stage: str = "unknown"

    symptoms: dict[str, float] = field(default_factory=dict)
    assessments: list[AssessmentRecord] = field(
        default_factory=list
    )

    last_updated: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    def update_from_assessment(
        self,
        assessment: AssessmentRecord,
    ) -> None:
        """Update the current twin state from a new assessment."""

        if assessment.clinical_risk is not None:
            self.clinical_risk = assessment.clinical_risk

        if assessment.mri_risk is not None:
            self.mri_risk = assessment.mri_risk

        self.combined_risk = calculate_combined_risk(
            clinical_risk=self.clinical_risk,
            mri_risk=self.mri_risk,
        )

        if assessment.predicted_disease is not None:
            self.predicted_disease = (
                assessment.predicted_disease
            )

        if assessment.symptoms:
            self.symptoms.update(assessment.symptoms)

        self.disease_stage = determine_disease_stage(
            self.combined_risk
        )

        self.assessments.append(assessment)

        self.last_updated = datetime.now(
            timezone.utc
        ).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert the complete patient state to a dictionary."""

        return asdict(self)


def calculate_combined_risk(
    clinical_risk: float,
    mri_risk: float,
    clinical_weight: float = 0.5,
    mri_weight: float = 0.5,
) -> float:
    """
    Calculate a research-level multimodal risk score.

    The initial version uses equal weighting.
    """

    if not 0.0 <= clinical_risk <= 1.0:
        raise ValueError(
            "Clinical risk must be between 0 and 1."
        )

    if not 0.0 <= mri_risk <= 1.0:
        raise ValueError(
            "MRI risk must be between 0 and 1."
        )

    if clinical_weight < 0 or mri_weight < 0:
        raise ValueError(
            "Risk weights cannot be negative."
        )

    total_weight = clinical_weight + mri_weight

    if total_weight == 0:
        raise ValueError(
            "At least one risk weight must be positive."
        )

    combined_risk = (
        clinical_risk * clinical_weight
        + mri_risk * mri_weight
    ) / total_weight

    return round(combined_risk, 4)


def determine_disease_stage(risk: float) -> str:
    """Map the research risk score to a simple state label."""

    if not 0.0 <= risk <= 1.0:
        raise ValueError(
            "Risk must be between 0 and 1."
        )

    if risk < 0.25:
        return "low_risk"

    if risk < 0.50:
        return "moderate_risk"

    if risk < 0.75:
        return "high_risk"

    return "very_high_risk"