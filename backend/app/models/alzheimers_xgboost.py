from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from app.preprocessing.alzheimers_pipeline import prepare_alzheimers_data

DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "raw"
    / "clinical"
    / "alzheimers_disease.csv"
)


def train_alzheimers_xgboost(
    dataset_path: str | Path = DATASET_PATH,
) -> dict[str, Any]:
    """Train and evaluate an XGBoost classifier for Alzheimer's detection."""
    dataset = prepare_alzheimers_data(dataset_path)

    # Class balance adjustment
    class_counts = dataset["y_train"].value_counts()
    scale_pos_weight = (
        class_counts[0] / class_counts[1] if 1 in class_counts and class_counts[1] > 0 else 1.0
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
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
        "f1_score": float(f1_score(dataset["y_test"], predictions, zero_division=0)),
        "specificity": specificity,
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
