from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.preprocessing.stroke_pipeline import prepare_stroke_data

DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "raw"
    / "clinical"
    / "brain_stroke.csv"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StrokeTabTransformer(nn.Module):
    """Tabular Transformer Architecture for Brain Stroke biometric risk estimation."""

    def __init__(
        self,
        num_features: int,
        embedding_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.feature_embeddings = nn.Parameter(
            torch.randn(num_features, embedding_dim)
        )
        self.value_projection = nn.Linear(1, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = self.value_projection(x.unsqueeze(-1))
        embeddings = values + self.feature_embeddings.unsqueeze(0)
        encoded = self.encoder(embeddings)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled).squeeze(-1)


def train_stroke_tabtransformer(
    dataset_path: str | Path = DATASET_PATH,
    epochs: int = 45,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Train and evaluate the TabTransformer deep learning model on Brain Stroke data."""
    dataset = prepare_stroke_data(dataset_path)

    X_train_np = dataset["X_train"].to_numpy()
    X_test_np = dataset["X_test"].to_numpy()
    y_train_np = dataset["y_train"].to_numpy()
    y_test_np = dataset["y_test"].to_numpy()

    X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_np, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_np, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = StrokeTabTransformer(
        num_features=X_train_np.shape[1],
    ).to(DEVICE)

    pos_count = float(y_train_np.sum())
    neg_count = float(len(y_train_np) - pos_count)
    pos_weight = torch.tensor(
        [(neg_count / max(pos_count, 1.0)) * 0.5],
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
    )

    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        correct_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            batch_len = len(batch_y)
            total_loss += loss.item() * batch_len
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct_train += (preds == batch_y).sum().item()
            total_samples += batch_len

        train_loss = total_loss / max(total_samples, 1)
        train_acc = correct_train / max(total_samples, 1)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_test_t.to(DEVICE))
            val_loss = criterion(val_logits, y_test_t.to(DEVICE)).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_preds = (val_probs >= 0.35).astype(int)
            val_acc = float(accuracy_score(y_test_np, val_preds))

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
        })

    model.eval()
    with torch.no_grad():
        final_logits = model(X_test_t.to(DEVICE))
        probabilities = torch.sigmoid(final_logits).cpu().numpy()

    predictions = (probabilities >= 0.35).astype(int)
    cm = confusion_matrix(y_test_np, predictions)
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_test_np, predictions)),
        "precision": float(precision_score(y_test_np, predictions, zero_division=0)),
        "recall": float(recall_score(y_test_np, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_test_np, predictions, zero_division=0)),
        "specificity": specificity,
        "f1_score": float(f1_score(y_test_np, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test_np, probabilities)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test_np,
            predictions,
            zero_division=0,
        ),
    }

    return {
        "model": model,
        "metrics": metrics,
        "history": history,
        "feature_columns": dataset["feature_columns"],
        "transformer": dataset["transformer"],
        "X_test": dataset["X_test"],
        "y_test": dataset["y_test"],
        "y_prob": probabilities,
        "y_pred": predictions,
    }
