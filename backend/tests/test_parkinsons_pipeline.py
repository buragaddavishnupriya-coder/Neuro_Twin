from pathlib import Path

from app.preprocessing.parkinsons_pipeline import (
    prepare_parkinsons_data,
)


DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "raw"
    / "clinical"
    / "Parkinsson disease.csv"
)


def test_prepare_parkinsons_data() -> None:
    result = prepare_parkinsons_data(DATASET_PATH)

    assert result["X_train"].shape[1] == 22
    assert result["X_test"].shape[1] == 22

    assert len(result["X_train"]) == 156
    assert len(result["X_test"]) == 39

    assert len(result["y_train"]) == 156
    assert len(result["y_test"]) == 39

    assert result["validation_report"].is_valid is True


def test_stratified_split_preserves_both_classes() -> None:
    result = prepare_parkinsons_data(DATASET_PATH)

    assert set(result["y_train"].unique()) == {0, 1}
    assert set(result["y_test"].unique()) == {0, 1}


def test_scaler_is_fitted() -> None:
    result = prepare_parkinsons_data(DATASET_PATH)

    assert result["transformer"].mean_.shape[0] == 22