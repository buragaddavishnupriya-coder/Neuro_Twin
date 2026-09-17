from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ParkinsonsValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


REQUIRED_COLUMNS = {
    "name",
    "status",
}

EXPECTED_FEATURE_COUNT = 22


def validate_parkinsons_data(
    data: pd.DataFrame,
) -> ParkinsonsValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(data, pd.DataFrame):
        return ParkinsonsValidationResult(
            is_valid=False,
            errors=["Input must be a pandas DataFrame."],
        )

    if data.empty:
        return ParkinsonsValidationResult(
            is_valid=False,
            errors=["Parkinson's dataset is empty."],
        )

    missing_columns = REQUIRED_COLUMNS - set(data.columns)

    if missing_columns:
        errors.append(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if errors:
        return ParkinsonsValidationResult(False, errors, warnings)

    if data["name"].isna().any():
        errors.append("Recording IDs contain missing values.")

    if data["name"].duplicated().any():
        errors.append("Recording IDs must be unique.")

    if data["status"].isna().any():
        errors.append("Status contains missing values.")

    if not data["status"].isin([0, 1]).all():
        errors.append("Status must contain only 0 or 1.")

    feature_columns = [
        column
        for column in data.columns
        if column not in {"name", "status"}
    ]

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        errors.append(
            f"Expected {EXPECTED_FEATURE_COUNT} voice features, "
            f"found {len(feature_columns)}."
        )

    for column in feature_columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            errors.append(f"Feature '{column}' must be numeric.")

    if data.isna().any().any():
        warnings.append("Dataset contains missing values.")

    class_counts = data["status"].value_counts()

    if len(class_counts) == 2:
        minority_count = class_counts.min()
        majority_count = class_counts.max()

        if minority_count / majority_count < 0.5:
            warnings.append(
                "Dataset is imbalanced; use stratified splitting "
                "and class-aware evaluation."
            )

    return ParkinsonsValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )