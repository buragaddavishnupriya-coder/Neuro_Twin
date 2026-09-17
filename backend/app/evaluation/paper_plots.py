from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "paper_results"

# Set publication-quality matplotlib defaults
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.2,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})


def ensure_output_dir(output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_training_validation_curves(
    history: list[dict[str, float]],
    disease_name: str,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """
    Generate Dual-Panel Training vs. Validation Loss and Accuracy Curves (300 DPI).
    """
    ensure_output_dir(output_dir)
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    train_acc = [item["train_acc"] * 100 for item in history]
    val_acc = [item["val_acc"] * 100 for item in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

    # 1. Loss Curve
    ax1.plot(epochs, train_loss, label="Training Loss", color="#1f77b4", marker="o", markersize=3)
    ax1.plot(epochs, val_loss, label="Validation Loss", color="#d62728", linestyle="--", marker="s", markersize=3)
    ax1.set_title(f"Training vs. Validation Loss ({disease_name})", fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary Cross-Entropy Loss")
    ax1.grid(True)
    ax1.legend(loc="upper right", frameon=True)

    # 2. Accuracy Curve
    ax2.plot(epochs, train_acc, label="Training Accuracy", color="#2ca02c", marker="o", markersize=3)
    ax2.plot(epochs, val_acc, label="Validation Accuracy", color="#9467bd", linestyle="--", marker="s", markersize=3)
    ax2.set_title(f"Training vs. Validation Accuracy ({disease_name})", fontweight="bold", pad=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim([min(min(train_acc), min(val_acc)) - 5, 102])
    ax2.grid(True)
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    output_path = output_dir / f"training_val_curves_{disease_name.lower().replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    model_name: str,
    disease_name: str,
    class_labels: list[str] = ["Healthy / Control", "Disease Detected"],
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Generate high-resolution annotated Confusion Matrix heatmap."""
    ensure_output_dir(output_dir)
    cm_arr = np.array(cm)

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax.set_title(f"Confusion Matrix: {model_name}\n({disease_name})", fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontweight="bold")
    ax.set_ylabel("True Ground Truth", fontweight="bold")

    plt.tight_layout()
    clean_m = model_name.lower().replace(" ", "_")
    clean_d = disease_name.lower().replace(" ", "_")
    output_path = output_dir / f"confusion_matrix_{clean_d}_{clean_m}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_roc_curves(
    models_data: dict[str, tuple[np.ndarray, np.ndarray]],  # name -> (y_true, y_prob)
    disease_name: str,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Generate comparative Receiver Operating Characteristic (ROC) curves."""
    ensure_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for (name, (y_true, y_prob)), color in zip(models_data.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", color=color)

    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random Chance (AUC = 0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontweight="bold")
    ax.set_title(f"ROC Curves - {disease_name}", fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True)

    plt.tight_layout()
    clean_d = disease_name.lower().replace(" ", "_")
    output_path = output_dir / f"roc_curves_{clean_d}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_precision_recall_curves(
    models_data: dict[str, tuple[np.ndarray, np.ndarray]],
    disease_name: str,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Generate Precision-Recall (PR) Curves with Average Precision (AP)."""
    ensure_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for (name, (y_true, y_prob)), color in zip(models_data.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, label=f"{name} (AP = {ap:.3f})", color=color)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontweight="bold")
    ax.set_ylabel("Precision", fontweight="bold")
    ax.set_title(f"Precision-Recall Curves - {disease_name}", fontweight="bold", pad=12)
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True)

    plt.tight_layout()
    clean_d = disease_name.lower().replace(" ", "_")
    output_path = output_dir / f"precision_recall_curves_{clean_d}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    disease_name: str,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Generate Actual vs. Predicted Confidence Distribution plot."""
    ensure_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=300)

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    h_probs = y_prob[y_true == 0]
    d_probs = y_prob[y_true == 1]

    ax.hist(h_probs, bins=15, alpha=0.65, label="Actual: Healthy / Control", color="#2ca02c", edgecolor="black")
    ax.hist(d_probs, bins=15, alpha=0.65, label=f"Actual: {disease_name}", color="#d62728", edgecolor="black")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.8, label="Decision Threshold (0.50)")

    ax.set_title(f"Actual vs. Predicted Risk Distribution\n{model_name} ({disease_name})", fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Disease Probability", fontweight="bold")
    ax.set_ylabel("Number of Patients", fontweight="bold")
    ax.legend(loc="upper center", frameon=True)
    ax.grid(True)

    plt.tight_layout()
    clean_m = model_name.lower().replace(" ", "_")
    clean_d = disease_name.lower().replace(" ", "_")
    output_path = output_dir / f"actual_vs_predicted_{clean_d}_{clean_m}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_literature_comparison(
    comparison_data: dict[str, float],
    disease_name: str,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """
    Generate Existing / Previous Work vs. Proposed NeuroTwin Comparison Graph.
    """
    ensure_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)

    methods = list(comparison_data.keys())
    accuracies = list(comparison_data.values())

    # Highlight proposed work in distinct vibrant green/teal
    colors = [
        "#2ca02c" if "Proposed" in m or "NeuroTwin" in m else "#4682b4"
        for m in methods
    ]

    bars = ax.barh(methods, accuracies, color=colors, height=0.55, edgecolor="black", linewidth=0.8)
    ax.set_xlim([70, 102])
    ax.set_xlabel("Classification Accuracy (%)", fontweight="bold")
    ax.set_title(f"Previous Literature vs. Proposed Work ({disease_name})", fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.6,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}%",
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=10,
        )

    plt.tight_layout()
    clean_d = disease_name.lower().replace(" ", "_")
    output_path = output_dir / f"literature_comparison_{clean_d}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
