from dataclasses import dataclass, field

import pandas as pd


REQUIRED_COLUMNS = {
    "patient_id",
    "age",
    "gender",
    "blood_pressure",
    "glucose",
    "diagnosis",
}


@dataclass
class ValidationResult:
    """Contains the outcome of clinical data validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_clinical_data(data: pd.DataFrame) -> ValidationResult:
    """
    Validate the structure and basic quality of clinical data.

    This function does not modify the input DataFrame.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(data, pd.DataFrame):
        return ValidationResult(
            is_valid=False,
            errors=["Input must be a pandas DataFrame."],
        )

    if data.empty:
        errors.append("Clinical dataset is empty.")
        return ValidationResult(False, errors, warnings)

    missing_columns = REQUIRED_COLUMNS - set(data.columns)

    if missing_columns:
        errors.append(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if errors:
        return ValidationResult(False, errors, warnings)

    if data["patient_id"].isna().any():
        errors.append("Patient IDs contain missing values.")

    if data["patient_id"].duplicated().any():
        errors.append("Patient IDs must be unique.")

    numeric_columns = [
        "age",
        "blood_pressure",
        "glucose",
    ]

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            errors.append(f"Column '{column}' must be numeric.")

    if data["age"].isna().any():
        errors.append("Age contains missing values.")

    if data["diagnosis"].isna().any():
        errors.append("Diagnosis contains missing values.")

    if (data["age"] <= 0).any() or (data["age"] > 120).any():
        errors.append("Age must be between 1 and 120.")

    if (data["blood_pressure"] <= 0).any():
        errors.append("Blood pressure must be positive.")

    if (data["glucose"] <= 0).any():
        errors.append("Glucose must be positive.")

    if data.isna().any().any():
        warnings.append("Dataset contains missing values.")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )