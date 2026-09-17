from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.preprocessing.parkinsons_validator import (
    validate_parkinsons_data,
)


DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "raw"
    / "clinical"
    / "Parkinsson disease.csv"
)


def prepare_parkinsons_data(
    dataset_path: str | Path = DATASET_PATH,
) -> dict:
    """
    Load, validate, split, and scale the Parkinson's dataset.

    The scaler is fitted only on the training features.
    """

    data = pd.read_csv(dataset_path)

    validation_report = validate_parkinsons_data(data)

    if not validation_report.is_valid:
        raise ValueError(
            f"Invalid Parkinson's dataset: "
            f"{validation_report.errors}"
        )

    feature_columns = [
        column
        for column in data.columns
        if column not in {"name", "status"}
    ]

    X = data[feature_columns].copy()
    y = data["status"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    transformer = StandardScaler()

    X_train_scaled = transformer.fit_transform(X_train)
    X_test_scaled = transformer.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=feature_columns,
        index=X_train.index,
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=feature_columns,
        index=X_test.index,
    )

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_columns": feature_columns,
        "transformer": transformer,
        "validation_report": validation_report,
    }