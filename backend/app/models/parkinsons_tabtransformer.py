from pathlib import Path

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

from app.preprocessing.parkinsons_pipeline import prepare_parkinsons_data


DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "raw"
    / "clinical"
    / "Parkinsson disease.csv"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TabTransformer(nn.Module):
    """A compact transformer encoder for numerical tabular features."""

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
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, num_features)
        values = self.value_projection(x.unsqueeze(-1))

        # Each feature receives its own learnable embedding.
        embeddings = values + self.feature_embeddings.unsqueeze(0)

        encoded = self.encoder(embeddings)

        # Aggregate the feature representations.
        pooled = encoded.mean(dim=1)

        return self.classifier(pooled).squeeze(-1)


def train_tabtransformer_model() -> dict:
    dataset = prepare_parkinsons_data(DATASET_PATH)

    X_train = torch.tensor(
        dataset["X_train"].to_numpy(),
        dtype=torch.float32,
    )
    X_test = torch.tensor(
        dataset["X_test"].to_numpy(),
        dtype=torch.float32,
    )
    y_train = torch.tensor(
        dataset["y_train"].to_numpy(),
        dtype=torch.float32,
    )
    y_test = torch.tensor(
        dataset["y_test"].to_numpy(),
        dtype=torch.float32,
    )

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )

    model = TabTransformer(
        num_features=X_train.shape[1],
    ).to(DEVICE)

    # Calculate a positive-class weight for the imbalanced dataset.
    positive_count = y_train.sum().item()
    negative_count = len(y_train) - positive_count
    pos_weight = torch.tensor(
        [negative_count / positive_count],
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )

    history: list[dict[str, float]] = []

    for epoch in range(1, 61):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_labels)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct_train += (preds == batch_labels).sum().item()
            total_train += len(batch_labels)

        train_loss = total_loss / max(total_train, 1)
        train_acc = correct_train / max(total_train, 1)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_test.to(DEVICE))
            val_loss = criterion(val_logits, y_test.to(DEVICE)).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_preds = (val_probs >= 0.5).astype(int)
            val_acc = float(accuracy_score(y_test.numpy(), val_preds))

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
        })

    model.eval()
    with torch.no_grad():
        test_features = X_test.to(DEVICE)
        logits = model(test_features)
        probabilities = torch.sigmoid(logits).cpu().numpy()

    predictions = (probabilities >= 0.5).astype(int)
    cm = confusion_matrix(y_test.numpy(), predictions)
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_test.numpy(), predictions)),
        "precision": float(precision_score(
            y_test.numpy(),
            predictions,
            zero_division=0,
        )),
        "recall": float(recall_score(
            y_test.numpy(),
            predictions,
            zero_division=0,
        )),
        "specificity": specificity,
        "f1_score": float(f1_score(
            y_test.numpy(),
            predictions,
            zero_division=0,
        )),
        "roc_auc": float(roc_auc_score(
            y_test.numpy(),
            probabilities,
        )),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test.numpy(),
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