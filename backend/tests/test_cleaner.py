import pandas as pd

from app.preprocessing.cleaner import clean_clinical_data


def test_clean_numeric_missing_values():
    data = pd.DataFrame(
        {
            "patient_id": ["P001", "P002", "P003"],
            "age": [62, None, 55],
            "gender": ["M", "F", "M"],
            "blood_pressure": [145, 150, None],
            "glucose": [118, 122, 105],
            "diagnosis": ["Parkinson", "Alzheimer's", "Healthy"],
        }
    )

    cleaned = clean_clinical_data(data)

    assert cleaned["age"].isna().sum() == 0
    assert cleaned["blood_pressure"].isna().sum() == 0


def test_clean_categorical_missing_values():
    data = pd.DataFrame(
        {
            "patient_id": ["P001", "P002", "P003"],
            "age": [62, 71, 55],
            "gender": ["M", None, "M"],
            "blood_pressure": [145, 150, 130],
            "glucose": [118, 122, 105],
            "diagnosis": ["Parkinson", "Alzheimer's", None],
        }
    )

    cleaned = clean_clinical_data(data)

    assert cleaned["gender"].isna().sum() == 0
    assert cleaned["diagnosis"].isna().sum() == 0


def test_cleaner_does_not_modify_original():
    data = pd.DataFrame(
        {
            "patient_id": ["P001"],
            "age": [None],
            "gender": ["M"],
            "blood_pressure": [145],
            "glucose": [118],
            "diagnosis": ["Parkinson"],
        }
    )

    original = data.copy()

    clean_clinical_data(data)

    assert data.equals(original)