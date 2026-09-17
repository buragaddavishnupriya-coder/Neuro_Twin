from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.preprocessing.alzheimers_validator import validate_alzheimers_data

DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "raw"
    / "clinical"
    / "alzheimers_disease.csv"
)


def prepare_alzheimers_data(
    dataset_path: str | Path = DATASET_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Load, clean, validate, split, and scale the OASIS Alzheimer's clinical dataset.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Alzheimer's dataset not found at {path}")

    data = pd.read_csv(path)

    validation_report = validate_alzheimers_data(data)
    if not validation_report.is_valid:
        raise ValueError(f"Invalid Alzheimer's dataset: {validation_report.errors}")

    # Clean and standardize target label
    # In OASIS, CDR (Clinical Dementia Rating) determines Alzheimer's:
    # CDR == 0: Non-Demented (0), CDR > 0: Demented / Early Alzheimer's (1)
    df = data.copy()

    # If CDR is NaN, we drop those rows to maintain rigorous ground truth
    df = df.dropna(subset=["CDR"]).copy()
    df["status"] = (df["CDR"] > 0).astype(int)

    # Encode categorical columns
    if "M/F" in df.columns:
        df["Gender_M"] = (df["M/F"] == "M").astype(int)

    # Candidate feature columns
    candidate_features = ["Age", "Educ", "SES", "MMSE", "eTIV", "nWBV", "ASF", "Gender_M"]
    feature_columns = [col for col in candidate_features if col in df.columns]

    # Impute missing values with column median
    for col in feature_columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    X = df[feature_columns].copy()
    y = df["status"].copy()

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
        "validation_report": validation_report,
        "raw_df": df,
    }
