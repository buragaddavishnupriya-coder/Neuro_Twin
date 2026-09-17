from __future__ import annotations
from pathlib import Path

import csv

import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from app.models.parkinsons_tabtransformer import (
    DEVICE,
    TabTransformer,
)



def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
    }


def _train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    epochs: int,
    seed: int,
) -> np.ndarray:
    _set_seed(seed)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(
        X_train_scaled,
        dtype=torch.float32,
    ).to(DEVICE)

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32,
    ).to(DEVICE)

    X_test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32,
    ).to(DEVICE)

    model = TabTransformer(
        num_features=X_train.shape[1]
    ).to(DEVICE)

    class_counts = np.bincount(y_train.astype(int))

    positive_weight = class_counts[0] / max(
        class_counts[1],
        1,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            positive_weight,
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    optimizer = AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )

    dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor,
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
    )

    model.train()

    for _ in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()

            logits = model(batch_X).reshape(-1)

            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

    model.eval()

    with torch.no_grad():
        logits = model(X_test_tensor).reshape(-1)
        probabilities = torch.sigmoid(logits)

    return probabilities.cpu().numpy()


def evaluate_tabtransformer_cross_validation(
    n_splits: int = 5,
    epochs: int = 50,
    random_state: int = 42,
) -> dict[str, dict[str, dict[str, float]]]:
    csv_path = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "raw"
        / "clinical"
        / "Parkinsson disease.csv"
    )

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    data = pd.DataFrame(rows)

    for column in data.columns:
        if column not in ["name", "status"]:
            data[column] = pd.to_numeric(data[column])

    data["status"] = pd.to_numeric(data["status"]).astype(int)

    feature_columns = [
        column
        for column in data.columns
        if column not in ["name", "status"]
    ]

    X = data[feature_columns].to_numpy(dtype=np.float32)
    y = data["status"].to_numpy(dtype=np.int64)
    
    
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy(dtype=np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)

    if isinstance(y, pd.Series):
        y = y.to_numpy(dtype=np.int64)
    else:
        y = np.asarray(y, dtype=np.int64)

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_metrics: dict[str, list[float]] = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "roc_auc": [],
    }

    for fold_number, (train_indices, test_indices) in enumerate(
        splitter.split(X, y),
        start=1,
    ):
        probabilities = _train_one_fold(
            X_train=X[train_indices],
            y_train=y[train_indices],
            X_test=X[test_indices],
            epochs=epochs,
            seed=random_state + fold_number,
        )

        metrics = _calculate_metrics(
            y_true=y[test_indices],
            probabilities=probabilities,
        )

        for metric_name, metric_value in metrics.items():
            fold_metrics[metric_name].append(metric_value)

    summary = {}

    for metric_name, values in fold_metrics.items():
        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    return {
        "TabTransformer": summary
    }


if __name__ == "__main__":
    results = evaluate_tabtransformer_cross_validation()

    for model_name, metrics in results.items():
        print(f"\n{model_name}")

        for metric_name, values in metrics.items():
            print(
                f"{metric_name}: "
                f"{values['mean']:.4f} "
                f"+/- {values['std']:.4f}"
            )