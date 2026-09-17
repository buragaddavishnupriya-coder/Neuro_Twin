from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.preprocessing.stroke_validator import validate_stroke_data

DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "raw"
    / "clinical"
    / "brain_stroke.csv"
)


def prepare_stroke_data(
    dataset_path: str | Path = DATASET_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Load, clean, encode, split, and scale the Brain Stroke clinical dataset."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Brain Stroke dataset not found at {path}")

    df = pd.read_csv(path)

    val_result = validate_stroke_data(df)
    if not val_result.is_valid:
        raise ValueError(f"Invalid stroke dataset: {val_result.errors}")

    # Drop non-predictive id column
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Clean gender: remove very rare 'Other'
    if "gender" in df.columns:
        df = df[df["gender"] != "Other"].copy()
        df["gender"] = (df["gender"] == "Male").astype(int)

    # Encode binary columns
    if "ever_married" in df.columns:
        df["ever_married"] = (df["ever_married"] == "Yes").astype(int)

    if "Residence_type" in df.columns:
        df["Residence_type"] = (df["Residence_type"] == "Urban").astype(int)

    # Impute missing BMI with median
    if "bmi" in df.columns:
        df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    # One-hot encode remaining categoricals
    categorical_cols = [col for col in ["work_type", "smoking_status"] if col in df.columns]
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    feature_columns = [col for col in df.columns if col != "stroke"]
    X = df[feature_columns].copy()
    y = df["stroke"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
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
        "validation_report": val_result,
        "raw_df": df,
    }
