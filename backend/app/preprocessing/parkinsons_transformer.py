import pandas as pd
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "status"
ID_COLUMN = "name"


def create_parkinsons_transformer() -> StandardScaler:
    """Create a standard scaler for Parkinson's voice features."""
    return StandardScaler()


def transform_parkinsons_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """
    Separate identifiers, features, and target, then scale the features.
    """

    required_columns = {ID_COLUMN, TARGET_COLUMN}

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    feature_columns = [
        column
        for column in data.columns
        if column not in {ID_COLUMN, TARGET_COLUMN}
    ]

    if not feature_columns:
        raise ValueError("No feature columns found.")

    X = data[feature_columns].copy()
    y = data[TARGET_COLUMN].copy()

    transformer = create_parkinsons_transformer()

    transformed_array = transformer.fit_transform(X)

    transformed_data = pd.DataFrame(
        transformed_array,
        columns=feature_columns,
        index=data.index,
    )

    return transformed_data, y, transformer