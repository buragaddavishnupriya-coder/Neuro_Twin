from app.models.parkinsons_tabtransformer_cv import (
    evaluate_tabtransformer_cross_validation,
)


def test_tabtransformer_cross_validation_returns_metrics():
    results = evaluate_tabtransformer_cross_validation(
        n_splits=2,
        epochs=2,
    )

    assert "TabTransformer" in results

    metrics = results["TabTransformer"]

    expected_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    ]

    for metric in expected_metrics:
        assert metric in metrics
        assert "mean" in metrics[metric]
        assert "std" in metrics[metric]

        assert 0.0 <= metrics[metric]["mean"] <= 1.0
        assert metrics[metric]["std"] >= 0.0