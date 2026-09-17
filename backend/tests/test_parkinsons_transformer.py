import pandas as pd
import pytest

from app.preprocessing.parkinsons_transformer import (
    transform_parkinsons_data,
)


def create_sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["sample_1", "sample_2", "sample_3"],
            "feature_1": [1.0, 2.0, 3.0],
            "feature_2": [4.0, 5.0, 6.0],
            "status": [0, 1, 1],
        }
    )


def test_features_and_target_are_separated() -> None:
    data = create_sample_data()

    X, y, transformer = transform_parkinsons_data(data)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (3, 2)
    assert list(y) == [0, 1, 1]
    assert "name" not in X.columns
    assert "status" not in X.columns
    assert transformer is not None


def test_features_are_scaled() -> None:
    data = create_sample_data()

    X, _, _ = transform_parkinsons_data(data)

    for column in X.columns:
        assert abs(X[column].mean()) < 1e-6


def test_missing_required_column_raises_error() -> None:
    data = create_sample_data().drop(columns=["status"])

    with pytest.raises(ValueError, match="Missing required columns"):
        transform_parkinsons_data(data)