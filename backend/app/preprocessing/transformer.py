import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "age",
    "blood_pressure",
    "glucose",
]

CATEGORICAL_FEATURES = [
    "gender",
]


def create_transformer() -> ColumnTransformer:
    """
    Create the preprocessing transformer for clinical features.

    Numerical features are standardized.
    Categorical features are one-hot encoded.
    """
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def transform_clinical_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Transform clinical features and separate the diagnosis target.

    Returns:
        X: Transformed numerical feature DataFrame.
        y: Diagnosis target Series.
        transformer: Fitted ColumnTransformer.
    """
    required_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + ["diagnosis"]
    )

    missing_columns = set(required_columns) - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = data["diagnosis"].copy()

    transformer = create_transformer()

    transformed_array = transformer.fit_transform(X)

    feature_names = transformer.get_feature_names_out()

    transformed_data = pd.DataFrame(
        transformed_array,
        columns=feature_names,
        index=data.index,
    )

    return transformed_data, y, transformer