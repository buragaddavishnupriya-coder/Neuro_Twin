from app.models.parkinsons_xgboost import train_xgboost_model


def test_xgboost_model_trains() -> None:
    result = train_xgboost_model()

    assert result["model"] is not None
    assert len(result["feature_columns"]) == 22


def test_xgboost_metrics_exist() -> None:
    result = train_xgboost_model()
    metrics = result["metrics"]

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_xgboost_confusion_matrix_shape() -> None:
    result = train_xgboost_model()
    confusion = result["metrics"]["confusion_matrix"]

    assert len(confusion) == 2
    assert len(confusion[0]) == 2