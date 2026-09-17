from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from app.preprocessing.parkinsons_pipeline import prepare_parkinsons_data


DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "raw"
    / "clinical"
    / "Parkinsson disease.csv"
)


def create_models() -> dict:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate_model(model, X, y, cv) -> dict:
    scores = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "roc_auc": [],
    }

    for train_index, test_index in cv.split(X, y):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        current_model = clone(model)
        current_model.fit(X_train, y_train)

        predictions = current_model.predict(X_test)
        probabilities = current_model.predict_proba(X_test)[:, 1]

        scores["accuracy"].append(accuracy_score(y_test, predictions))
        scores["precision"].append(
            precision_score(y_test, predictions, zero_division=0)
        )
        scores["recall"].append(
            recall_score(y_test, predictions, zero_division=0)
        )
        scores["f1_score"].append(
            f1_score(y_test, predictions, zero_division=0)
        )
        scores["roc_auc"].append(roc_auc_score(y_test, probabilities))

    return {
        metric: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
        for metric, values in scores.items()
    }


def run_cross_validation() -> dict:
    dataset = pd.read_csv(DATASET_PATH)

    feature_columns = [
        column for column in dataset.columns
        if column not in {"name", "status"}
    ]

    X = dataset[feature_columns]
    y = dataset["status"]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = {}

    for model_name, model in create_models().items():
        results[model_name] = evaluate_model(model, X, y, cv)

    return results