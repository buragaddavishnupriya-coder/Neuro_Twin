from typing import Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_ieee_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    y_prob: np.ndarray | list[float] | None = None,
) -> dict[str, Any]:
    """
    Compute full IEEE journal/conference evaluation metrics checklist:
    Accuracy, Precision, Recall/Sensitivity, Specificity, F1-score, ROC-AUC, Confusion Matrix.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))  # Sensitivity
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    else:
        # Multi-class extension (macro-averaged specificity)
        specificity_list = []
        for i in range(len(cm)):
            tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
            fp = np.sum(cm[:, i]) - cm[i, i]
            specificity_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        specificity = float(np.mean(specificity_list))

    auc = 0.0
    if y_prob is not None:
        try:
            y_prob = np.asarray(y_prob)
            if len(np.unique(y_true)) > 1:
                auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.0

    # Normalized confusion matrix
    cm_norm = (cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]).round(4).tolist()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "recall_sensitivity": rec,
        "specificity": specificity,
        "f1_score": f1,

        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm,
    }


def format_ieee_latex_table(results_by_disease: dict[str, dict[str, dict[str, Any]]]) -> str:
    """
    Generate an IEEE-compliant LaTeX table comparing all diseases, baseline models,
    machine learning (XGBoost), and proposed deep learning architectures.
    """
    latex_lines = [
        r"\begin{table*}[htbp]",
        r"\caption{Comparative Performance of NeuroTwin Models on Clinical Neurological Benchmarks}",
        r"\label{tab:neurotwin_results}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llcccccc}",
        r"\hline",
        r"\textbf{Disease Domain} & \textbf{Model Architecture} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Sensitivity} & \textbf{Specificity} & \textbf{F1-Score} & \textbf{ROC-AUC} \\",
        r"\hline",
    ]

    for disease_name, models in results_by_disease.items():
        first = True
        for model_name, metrics in models.items():
            disease_col = disease_name if first else ""
            first = False

            acc = f"{metrics.get('accuracy', 0.0) * 100:.2f}\\%"
            prec = f"{metrics.get('precision', 0.0) * 100:.2f}\\%"
            rec_val = metrics.get('recall_sensitivity', metrics.get('recall', 0.0))
            rec = f"{rec_val * 100:.2f}\\%"
            spec = f"{metrics.get('specificity', 0.0) * 100:.2f}\\%"
            f1 = f"{metrics.get('f1_score', 0.0) * 100:.2f}\\%"
            auc = f"{metrics.get('roc_auc', 0.0):.4f}"


            # Bold the best or proposed model
            if "Proposed" in model_name or "TabTransformer" in model_name:
                model_label = rf"\textbf{{{model_name}}}"
                acc = rf"\textbf{{{acc}}}"
                prec = rf"\textbf{{{prec}}}"
                rec = rf"\textbf{{{rec}}}"
                spec = rf"\textbf{{{spec}}}"
                f1 = rf"\textbf{{{f1}}}"
                auc = rf"\textbf{{{auc}}}"
            else:
                model_label = model_name

            latex_lines.append(
                f"{disease_col} & {model_label} & {acc} & {prec} & {rec} & {spec} & {f1} & {auc} \\\\"
            )
        latex_lines.append(r"\hline")

    latex_lines.extend([
        r"\end{tabular}",
        r"\end{table*}",
    ])

    return "\n".join(latex_lines)
