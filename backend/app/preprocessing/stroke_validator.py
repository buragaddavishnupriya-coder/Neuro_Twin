from dataclasses import dataclass, field
import pandas as pd


@dataclass
class StrokeValidationResult:
    """Contains the outcome of Brain Stroke clinical dataset validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


REQUIRED_COLUMNS = {
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "stroke",
}


def validate_stroke_data(data: pd.DataFrame) -> StrokeValidationResult:
    """Validate structure, biometric ranges, and clinical consistency of Stroke data."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(data, pd.DataFrame):
        return StrokeValidationResult(is_valid=False, errors=["Input must be a pandas DataFrame."])

    if data.empty:
        return StrokeValidationResult(is_valid=False, errors=["Stroke dataset is empty."])

    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
        return StrokeValidationResult(is_valid=False, errors=errors)

    if (data["age"] < 0).any() or (data["age"] > 120).any():
        errors.append("Patient age contains values outside realistic clinical range [0, 120].")

    if (data["avg_glucose_level"] < 30).any() or (data["avg_glucose_level"] > 400).any():
        errors.append("Average glucose levels contain values outside standard physiological range.")

    if not data["stroke"].isin([0, 1]).all():
        errors.append("Target column 'stroke' must contain binary values (0 or 1).")

    if data.isna().any().any():
        warnings.append("Dataset contains missing values (e.g. BMI) that require imputation.")

    return StrokeValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
