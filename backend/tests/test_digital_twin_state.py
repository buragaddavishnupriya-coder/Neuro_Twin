import pytest

from app.models.digital_twin.state import (
    AssessmentRecord,
    PatientState,
    calculate_combined_risk,
    determine_disease_stage,
)


def test_combined_risk_uses_equal_weights():
    result = calculate_combined_risk(
        clinical_risk=0.4,
        mri_risk=0.6,
    )

    assert result == 0.5


def test_combined_risk_rejects_invalid_values():
    with pytest.raises(ValueError):
        calculate_combined_risk(
            clinical_risk=1.2,
            mri_risk=0.5,
        )


def test_disease_stage():
    assert determine_disease_stage(0.10) == "low_risk"
    assert determine_disease_stage(0.40) == "moderate_risk"
    assert determine_disease_stage(0.60) == "high_risk"
    assert determine_disease_stage(0.90) == "very_high_risk"


def test_patient_state_updates():
    patient = PatientState(
        patient_id="demo-001",
        age=55,
    )

    assessment = AssessmentRecord(
        timestamp="2026-09-06T00:00:00+00:00",
        clinical_risk=0.4,
        mri_risk=0.6,
        predicted_disease="neurological_risk",
        symptoms={"tremor": 0.3},
    )

    patient.update_from_assessment(assessment)

    assert patient.clinical_risk == 0.4
    assert patient.mri_risk == 0.6
    assert patient.combined_risk == 0.5
    assert patient.predicted_disease == "neurological_risk"
    assert patient.disease_stage == "high_risk"
    assert len(patient.assessments) == 1
    assert patient.symptoms["tremor"] == 0.3


def test_patient_state_serialization():
    patient = PatientState(
        patient_id="demo-002",
    )

    data = patient.to_dict()

    assert data["patient_id"] == "demo-002"
    assert "combined_risk" in data
    assert "assessments" in data