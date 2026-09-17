from pathlib import Path
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.preprocessing.stroke_pipeline import prepare_stroke_data

DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "raw"
    / "clinical"
    / "brain_stroke.csv"
)


def train_stroke_baseline(
    dataset_path: str | Path = DATASET_PATH,
) -> dict[str, Any]:
    """Train and evaluate balanced Logistic Regression baseline for Brain Stroke risk."""
    dataset = prepare_stroke_data(dataset_path)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(
        dataset["X_train"],
        dataset["y_train"],
    )

    predictions = model.predict(dataset["X_test"])
    probabilities = model.predict_proba(dataset["X_test"])[:, 1]

    cm = confusion_matrix(dataset["y_test"], predictions)
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": float(accuracy_score(dataset["y_test"], predictions)),
        "precision": float(precision_score(dataset["y_test"], predictions, zero_division=0)),
        "recall": float(recall_score(dataset["y_test"], predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(dataset["y_test"], predictions, zero_division=0)),
        "specificity": specificity,
        "f1_score": float(f1_score(dataset["y_test"], predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(dataset["y_test"], probabilities)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            dataset["y_test"],
            predictions,
            zero_division=0,
        ),
    }

    return {
        "model": model,
        "metrics": metrics,
        "feature_columns": dataset["feature_columns"],
        "transformer": dataset["transformer"],
        "X_test": dataset["X_test"],
        "y_test": dataset["y_test"],
        "y_prob": probabilities,
        "y_pred": predictions,
    }
