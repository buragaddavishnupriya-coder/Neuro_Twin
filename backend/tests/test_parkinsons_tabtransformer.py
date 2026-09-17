from app.models.parkinsons_tabtransformer import train_tabtransformer_model


def test_tabtransformer_model_trains() -> None:
    result = train_tabtransformer_model()

    assert result["model"] is not None
    assert len(result["feature_columns"]) == 22


def test_tabtransformer_metrics_exist() -> None:
    result = train_tabtransformer_model()
    metrics = result["metrics"]

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_tabtransformer_confusion_matrix_shape() -> None:
    result = train_tabtransformer_model()
    confusion = result["metrics"]["confusion_matrix"]

    assert len(confusion) == 2
    assert len(confusion[0]) == 2