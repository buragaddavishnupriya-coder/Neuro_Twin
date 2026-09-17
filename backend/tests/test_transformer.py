import pandas as pd

from app.preprocessing.transformer import (
    transform_clinical_data,
)


def create_sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["P001", "P002", "P003"],
            "age": [62, 71, 55],
            "gender": ["M", "F", "M"],
            "blood_pressure": [145, 150, 130],
            "glucose": [118, 122, 105],
            "diagnosis": [
                "Parkinson",
                "Alzheimer's",
                "Healthy",
            ],
        }
    )


def test_transform_clinical_data():
    data = create_sample_data()

    X, y, transformer = transform_clinical_data(data)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

    assert len(X) == 3
    assert len(y) == 3

    assert "patient_id" not in X.columns
    assert "diagnosis" not in X.columns

    assert X.shape[1] == 5


def test_target_is_separated():
    data = create_sample_data()

    X, y, _ = transform_clinical_data(data)

    assert list(y) == [
        "Parkinson",
        "Alzheimer's",
        "Healthy",
    ]

    assert "diagnosis" not in X.columns


def test_missing_column_raises_error():
    data = create_sample_data().drop(columns=["glucose"])

    try:
        transform_clinical_data(data)
        assert False, "Expected ValueError"
    except ValueError as error:
        assert "glucose" in str(error)