from __future__ import annotations

from typing import Any

from app.models.digital_twin.state import (
    AssessmentRecord,
    PatientState,
)


class DigitalTwinEngine:
    """Manage patient Digital Twin states."""

    def __init__(self) -> None:
        self._patients: dict[str, PatientState] = {}

    def create_patient(
        self,
        patient_id: str,
        age: int | None = None,
        sex: str | None = None,
    ) -> PatientState:
        """Create a new patient Digital Twin."""

        if patient_id in self._patients:
            raise ValueError(
                f"Patient already exists: {patient_id}"
            )

        patient = PatientState(
            patient_id=patient_id,
            age=age,
            sex=sex,
        )

        self._patients[patient_id] = patient

        return patient

    def get_patient(
        self,
        patient_id: str,
    ) -> PatientState:
        """Retrieve an existing patient Digital Twin."""

        if patient_id not in self._patients:
            raise KeyError(
                f"Patient not found: {patient_id}"
            )

        return self._patients[patient_id]

    def update_patient(
        self,
        patient_id: str,
        assessment: AssessmentRecord,
    ) -> PatientState:
        """Add a new assessment and update the patient state."""

        patient = self.get_patient(patient_id)
        patient.update_from_assessment(assessment)

        return patient

    def list_patients(self) -> list[str]:
        """Return all registered patient IDs."""

        return list(self._patients.keys())

    def delete_patient(
        self,
        patient_id: str,
    ) -> None:
        """Delete a patient Digital Twin."""

        if patient_id not in self._patients:
            raise KeyError(
                f"Patient not found: {patient_id}"
            )

        del self._patients[patient_id]

    def export_patient(
        self,
        patient_id: str,
    ) -> dict[str, Any]:
        """Export a patient state as a dictionary."""

        return self.get_patient(patient_id).to_dict()