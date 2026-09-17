from dataclasses import dataclass, field
import pandas as pd


@dataclass
class AlzheimersValidationResult:
    """Contains the outcome of Alzheimer's data validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


REQUIRED_COLUMNS = {
    "ID",
    "Age",
    "MMSE",
    "eTIV",
    "nWBV",
    "ASF",
}


def validate_alzheimers_data(data: pd.DataFrame) -> AlzheimersValidationResult:
    """
    Validate structure, ranges, and consistency of the Alzheimer's OASIS dataset.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(data, pd.DataFrame):
        return AlzheimersValidationResult(
            is_valid=False,
            errors=["Input must be a pandas DataFrame."],
        )

    if data.empty:
        return AlzheimersValidationResult(
            is_valid=False,
            errors=["Alzheimer's dataset is empty."],
        )

    missing_cols = REQUIRED_COLUMNS - set(data.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {sorted(missing_cols)}")
        return AlzheimersValidationResult(is_valid=False, errors=errors)

    if data["ID"].duplicated().any():
        warnings.append("Some patient IDs appear multiple times (longitudinal/repeat visits).")

    # Range validations
    if (data["Age"] < 18).any() or (data["Age"] > 115).any():
        errors.append("Patient Age contains values outside valid clinical range (18-115).")

    valid_mmse = data["MMSE"].dropna()
    if (valid_mmse < 0).any() or (valid_mmse > 30).any():
        errors.append("MMSE scores must be within the standardized range [0, 30].")

    valid_nwbv = data["nWBV"].dropna()
    if (valid_nwbv <= 0.4).any() or (valid_nwbv > 1.0).any():
        errors.append("nWBV (Normalized Whole Brain Volume) must be realistic brain fraction (0.4 to 1.0).")

    if data.isna().any().any():
        warnings.append("Dataset contains missing values that require imputation (e.g. SES, MMSE).")

    return AlzheimersValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
