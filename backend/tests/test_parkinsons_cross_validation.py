from app.models.parkinsons_cross_validation import run_cross_validation


def test_cross_validation_returns_all_models() -> None:
    results = run_cross_validation()

    assert "Logistic Regression" in results
    assert "Random Forest" in results
    assert "XGBoost" in results


def test_cross_validation_metrics_are_valid() -> None:
    results = run_cross_validation()

    for model_metrics in results.values():
        for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            assert 0.0 <= model_metrics[metric]["mean"] <= 1.0
            assert model_metrics[metric]["std"] >= 0.0