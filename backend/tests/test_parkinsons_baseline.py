from pathlib import Path

from app.models.parkinsons_baseline import train_baseline_model


def test_baseline_model_trains() -> None:
    result = train_baseline_model()

    assert result["model"] is not None
    assert len(result["feature_columns"]) == 22


def test_baseline_metrics_exist() -> None:
    result = train_baseline_model()
    metrics = result["metrics"]

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_confusion_matrix_shape() -> None:
    result = train_baseline_model()

    assert len(result["metrics"]["confusion_matrix"]) == 2
    assert len(result["metrics"]["confusion_matrix"][0]) == 2