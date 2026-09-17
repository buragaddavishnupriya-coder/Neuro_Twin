from pathlib import Path

import pandas as pd
import pytest

from app.preprocessing.loader import (
    DataLoadingError,
    load_clinical_data,
)


DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "raw"
    / "clinical"
    / "patients.csv"
)


def test_load_clinical_data():
    data = load_clinical_data(DATASET_PATH)

    assert isinstance(data, pd.DataFrame)
    assert len(data) == 3
    assert "patient_id" in data.columns
    assert "diagnosis" in data.columns


def test_missing_file():
    with pytest.raises(DataLoadingError):
        load_clinical_data("missing.csv")