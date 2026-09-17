from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.models.alzheimers_xgboost import train_alzheimers_xgboost

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "clinical"
    / "alzheimers_xgboost_model.joblib"
)


def train_and_save_alzheimers_model(
    output_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Train the Alzheimer's clinical model and save inference bundle."""
    result = train_alzheimers_xgboost()

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


def load_alzheimers_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Load a saved Alzheimer's inference bundle."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Alzheimer's model not found: {model_path}")
    return joblib.load(model_path)


def predict_alzheimers_risk(
    clinical_features: dict[str, float],
    model_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Predict Alzheimer's disease probability for one patient record."""
    feature_columns = model_bundle["feature_columns"]
    model = model_bundle["model"]
    transformer = model_bundle["transformer"]

    missing_features = [col for col in feature_columns if col not in clinical_features]
    if missing_features:
        raise ValueError(f"Missing Alzheimer's clinical features: {missing_features}")

    input_data = pd.DataFrame(
        [{col: clinical_features[col] for col in feature_columns}],
        columns=feature_columns,
    )

    input_scaled = transformer.transform(input_data)
    probability = float(model.predict_proba(input_scaled)[0][1])

    predicted_disease = (
        "Alzheimer's Dementia"
        if probability >= 0.5
        else "No Alzheimer's Dementia Detected"
    )

    return {
        "predicted_disease": predicted_disease,
        "alzheimers_risk": probability,
        "prediction": int(probability >= 0.5),
    }
