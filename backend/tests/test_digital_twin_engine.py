import pytest

from app.models.digital_twin.engine import (
    DigitalTwinEngine,
)
from app.models.digital_twin.state import (
    AssessmentRecord,
)


def test_create_and_get_patient():
    engine = DigitalTwinEngine()

    patient = engine.create_patient(
        patient_id="demo-001",
        age=55,
        sex="unknown",
    )

    assert patient.patient_id == "demo-001"
    assert engine.get_patient("demo-001") is patient


def test_duplicate_patient_is_rejected():
    engine = DigitalTwinEngine()

    engine.create_patient("demo-001")

    with pytest.raises(ValueError):
        engine.create_patient("demo-001")


def test_update_patient_history():
    engine = DigitalTwinEngine()

    engine.create_patient("demo-001")

    first_assessment = AssessmentRecord(
        timestamp="2026-09-07T00:00:00+00:00",
        clinical_risk=0.30,
        mri_risk=0.40,
    )

    second_assessment = AssessmentRecord(
        timestamp="2026-10-07T00:00:00+00:00",
        clinical_risk=0.60,
        mri_risk=0.70,
    )

    engine.update_patient(
        "demo-001",
        first_assessment,
    )

    patient = engine.update_patient(
        "demo-001",
        second_assessment,
    )

    assert patient.combined_risk == 0.65
    assert patient.disease_stage == "high_risk"
    assert len(patient.assessments) == 2


def test_missing_patient_is_rejected():
    engine = DigitalTwinEngine()

    with pytest.raises(KeyError):
        engine.get_patient("missing-patient")


def test_export_patient():
    engine = DigitalTwinEngine()

    engine.create_patient("demo-001", age=60)

    exported = engine.export_patient("demo-001")

    assert exported["patient_id"] == "demo-001"
    assert exported["age"] == 60
    assert "assessments" in exported