from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.models.parkinsons_xgboost import train_xgboost_model


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "clinical"
    / "xgboost_model.joblib"
)


def train_and_save_xgboost_model(
    output_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Train the clinical model and save its inference bundle."""

    result = train_xgboost_model()

    bundle = {
        "model": result["model"],
        "transformer": result["transformer"],
        "feature_columns": result["feature_columns"],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, output_path)

    return {
        "model_path": str(output_path),
        "metrics": result["metrics"],
        "feature_columns": result["feature_columns"],
    }


def load_xgboost_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Load a saved clinical inference bundle."""

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Clinical model not found: {model_path}"
        )

    return joblib.load(model_path)


def predict_clinical_risk(
    clinical_features: dict[str, float],
    model_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Predict Parkinson's risk for one patient."""

    feature_columns = model_bundle["feature_columns"]
    model = model_bundle["model"]
    transformer = model_bundle["transformer"]

    missing_features = [
        column
        for column in feature_columns
        if column not in clinical_features
    ]

    if missing_features:
        raise ValueError(
            f"Missing clinical features: {missing_features}"
        )

    input_data = pd.DataFrame(
        [
            {
                column: clinical_features[column]
                for column in feature_columns
            }
        ],
        columns=feature_columns,
    )

    input_scaled = transformer.transform(input_data)

    probability = float(
        model.predict_proba(input_scaled)[0][1]
    )

    predicted_disease = (
        "Parkinson's disease"
        if probability >= 0.5
        else "No Parkinson's disease detected"
    )

    return {
        "predicted_disease": predicted_disease,
        "clinical_risk": probability,
        "prediction": int(probability >= 0.5),
    }