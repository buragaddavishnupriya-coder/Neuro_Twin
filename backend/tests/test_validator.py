import pandas as pd

from app.preprocessing.validator import validate_clinical_data


def test_valid_clinical_data():
    data = pd.DataFrame(
        {
            "patient_id": ["P001", "P002"],
            "age": [62, 71],
            "gender": ["M", "F"],
            "blood_pressure": [145, 150],
            "glucose": [118, 122],
            "diagnosis": ["Parkinson", "Alzheimer's"],
        }
    )

    result = validate_clinical_data(data)

    assert result.is_valid is True
    assert result.errors == []


def test_duplicate_patient_ids():
    data = pd.DataFrame(
        {
            "patient_id": ["P001", "P001"],
            "age": [62, 71],
            "gender": ["M", "F"],
            "blood_pressure": [145, 150],
            "glucose": [118, 122],
            "diagnosis": ["Parkinson", "Alzheimer's"],
        }
    )

    result = validate_clinical_data(data)

    assert result.is_valid is False
    assert "Patient IDs must be unique." in result.errors


def test_missing_required_column():
    data = pd.DataFrame(
        {
            "patient_id": ["P001"],
            "age": [62],
        }
    )

    result = validate_clinical_data(data)

    assert result.is_valid is False
    assert any("Missing required columns" in error for error in result.errors)