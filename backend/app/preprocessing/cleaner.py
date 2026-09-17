import pandas as pd


def clean_clinical_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean clinical data by handling missing values.

    Numeric columns use median imputation.
    Categorical columns use mode imputation.

    Returns:
        A new cleaned DataFrame.
    """
    cleaned = data.copy()

    numeric_columns = [
        "age",
        "blood_pressure",
        "glucose",
    ]

    categorical_columns = [
        "gender",
        "diagnosis",
    ]

    for column in numeric_columns:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna(
                cleaned[column].median()
            )

    for column in categorical_columns:
        if column in cleaned.columns:
            mode = cleaned[column].mode()

            if not mode.empty:
                cleaned[column] = cleaned[column].fillna(mode.iloc[0])

    return cleaned