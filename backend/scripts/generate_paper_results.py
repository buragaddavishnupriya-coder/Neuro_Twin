import json
import os
import sys
from pathlib import Path

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import numpy as np

from app.evaluation.ieee_metrics import compute_ieee_metrics, format_ieee_latex_table
from app.evaluation.paper_plots import (
    plot_actual_vs_predicted,
    plot_confusion_matrix,
    plot_literature_comparison,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_training_validation_curves,
)
from app.models.alzheimers_baseline import train_alzheimers_baseline
from app.models.alzheimers_tabtransformer import train_alzheimers_tabtransformer
from app.models.alzheimers_xgboost import train_alzheimers_xgboost
from app.models.parkinsons_baseline import train_baseline_model as train_parkinsons_baseline
from app.models.parkinsons_tabtransformer import (
    train_tabtransformer_model as train_parkinsons_tabtransformer,
)
from app.models.parkinsons_xgboost import (
    train_xgboost_model as train_parkinsons_xgboost,
)
from app.models.stroke_baseline import train_stroke_baseline
from app.models.stroke_tabtransformer import train_stroke_tabtransformer
from app.models.stroke_xgboost import train_stroke_xgboost

OUTPUT_DIR = backend_dir.parent / "artifacts" / "paper_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_all_evaluations():
    print("=" * 70)
    print("NeuroTwin: Generating IEEE Paper Results & High-Resolution Figures")
    print("=" * 70)

    # 1. PARKINSON'S EVALUATION
    print("\n[1/5] Training Parkinson's Disease Models...")
    p_base = train_parkinsons_baseline()
    print(f"  -> Baseline (Logistic Regression) Accuracy: {p_base['metrics']['accuracy'] * 100:.2f}% | Specificity: {p_base['metrics']['specificity'] * 100:.2f}% | AUC: {p_base['metrics']['roc_auc']:.4f}")

    p_xgb = train_parkinsons_xgboost()
    print(f"  -> XGBoost Classifier Accuracy: {p_xgb['metrics']['accuracy'] * 100:.2f}% | Specificity: {p_xgb['metrics']['specificity'] * 100:.2f}% | AUC: {p_xgb['metrics']['roc_auc']:.4f}")

    p_tab = train_parkinsons_tabtransformer()
    print(f"  -> Proposed TabTransformer Accuracy: {p_tab['metrics']['accuracy'] * 100:.2f}% | Specificity: {p_tab['metrics']['specificity'] * 100:.2f}% | AUC: {p_tab['metrics']['roc_auc']:.4f}")

    # 2. ALZHEIMER'S EVALUATION
    print("\n[2/5] Training Alzheimer's Disease Models...")
    a_base = train_alzheimers_baseline()
    print(f"  -> Baseline (Logistic Regression) Accuracy: {a_base['metrics']['accuracy'] * 100:.2f}% | Specificity: {a_base['metrics']['specificity'] * 100:.2f}% | AUC: {a_base['metrics']['roc_auc']:.4f}")

    a_xgb = train_alzheimers_xgboost()
    print(f"  -> XGBoost Classifier Accuracy: {a_xgb['metrics']['accuracy'] * 100:.2f}% | Specificity: {a_xgb['metrics']['specificity'] * 100:.2f}% | AUC: {a_xgb['metrics']['roc_auc']:.4f}")

    a_tab = train_alzheimers_tabtransformer()
    print(f"  -> Proposed TabTransformer Accuracy: {a_tab['metrics']['accuracy'] * 100:.2f}% | Specificity: {a_tab['metrics']['specificity'] * 100:.2f}% | AUC: {a_tab['metrics']['roc_auc']:.4f}")

    # 3. BRAIN STROKE EVALUATION
    print("\n[3/5] Training Brain Stroke Models...")
    s_base = train_stroke_baseline()
    print(f"  -> Baseline (Logistic Regression) Accuracy: {s_base['metrics']['accuracy'] * 100:.2f}% | Specificity: {s_base['metrics']['specificity'] * 100:.2f}% | AUC: {s_base['metrics']['roc_auc']:.4f}")

    s_xgb = train_stroke_xgboost()
    print(f"  -> XGBoost Classifier Accuracy: {s_xgb['metrics']['accuracy'] * 100:.2f}% | Specificity: {s_xgb['metrics']['specificity'] * 100:.2f}% | AUC: {s_xgb['metrics']['roc_auc']:.4f}")

    s_tab = train_stroke_tabtransformer()
    print(f"  -> Proposed TabTransformer Accuracy: {s_tab['metrics']['accuracy'] * 100:.2f}% | Specificity: {s_tab['metrics']['specificity'] * 100:.2f}% | AUC: {s_tab['metrics']['roc_auc']:.4f}")

    # Aggregate Results by Disease
    results_by_disease = {
        "Parkinson's Disease": {
            "Baseline (LogReg)": p_base["metrics"],
            "XGBoost Classifier": p_xgb["metrics"],
            "Proposed TabTransformer": p_tab["metrics"],
        },
        "Alzheimer's Disease": {
            "Baseline (LogReg)": a_base["metrics"],
            "XGBoost Classifier": a_xgb["metrics"],
            "Proposed TabTransformer": a_tab["metrics"],
        },
        "Brain Stroke": {
            "Baseline (LogReg)": s_base["metrics"],
            "XGBoost Classifier": s_xgb["metrics"],
            "Proposed TabTransformer": s_tab["metrics"],
        },
    }

    # 4. GENERATING PUBLICATION PLOTS (300 DPI)
    print("\n[4/5] Generating 300-DPI Publication-Grade Figures...")

    # Dual-Panel Training vs. Validation Loss and Accuracy curves (Deep Learning TabTransformer)
    if "history" in p_tab and p_tab["history"]:
        f = plot_training_validation_curves(p_tab["history"], "Parkinsons Disease", OUTPUT_DIR)
        print(f"  [Created] {f.name}")

    if "history" in a_tab and a_tab["history"]:
        f = plot_training_validation_curves(a_tab["history"], "Alzheimers Disease", OUTPUT_DIR)
        print(f"  [Created] {f.name}")

    if "history" in s_tab and s_tab["history"]:
        f = plot_training_validation_curves(s_tab["history"], "Brain Stroke", OUTPUT_DIR)
        print(f"  [Created] {f.name}")

    # Confusion Matrices
    for name, res in [("Baseline", p_base), ("XGBoost", p_xgb), ("TabTransformer", p_tab)]:
        f = plot_confusion_matrix(res["metrics"]["confusion_matrix"], name, "Parkinsons Disease", output_dir=OUTPUT_DIR)
        print(f"  [Created] {f.name}")

    for name, res in [("Baseline", a_base), ("XGBoost", a_xgb), ("TabTransformer", a_tab)]:
        f = plot_confusion_matrix(res["metrics"]["confusion_matrix"], name, "Alzheimers Disease", output_dir=OUTPUT_DIR)
        print(f"  [Created] {f.name}")

    for name, res in [("Baseline", s_base), ("XGBoost", s_xgb), ("TabTransformer", s_tab)]:
        f = plot_confusion_matrix(res["metrics"]["confusion_matrix"], name, "Brain Stroke", output_dir=OUTPUT_DIR)
        print(f"  [Created] {f.name}")

    # Multi-Model ROC Curves
    p_roc_data = {
        "Baseline": (np.array(p_base["y_test"]), np.array(p_base["y_prob"])),
        "XGBoost": (np.array(p_xgb["y_test"]), np.array(p_xgb["y_prob"])),
        "Proposed TabTransformer": (np.array(p_tab["y_test"]), np.array(p_tab["y_prob"])),
    }
    f = plot_roc_curves(p_roc_data, "Parkinsons Disease", OUTPUT_DIR)
    print(f"  [Created] {f.name}")

    a_roc_data = {
        "Baseline": (np.array(a_base["y_test"]), np.array(a_base["y_prob"])),
        "XGBoost": (np.array(a_xgb["y_test"]), np.array(a_xgb["y_prob"])),
        "Proposed TabTransformer": (np.array(a_tab["y_test"]), np.array(a_tab["y_prob"])),
    }
    f = plot_roc_curves(a_roc_data, "Alzheimers Disease", OUTPUT_DIR)
    print(f"  [Created] {f.name}")

    s_roc_data = {
        "Baseline": (np.array(s_base["y_test"]), np.array(s_base["y_prob"])),
        "XGBoost": (np.array(s_xgb["y_test"]), np.array(s_xgb["y_prob"])),
        "Proposed TabTransformer": (np.array(s_tab["y_test"]), np.array(s_tab["y_prob"])),
    }
    f = plot_roc_curves(s_roc_data, "Brain Stroke", OUTPUT_DIR)
    print(f"  [Created] {f.name}")

    # Precision-Recall Curves
    f = plot_precision_recall_curves(p_roc_data, "Parkinsons Disease", OUTPUT_DIR)
    print(f"  [Created] {f.name}")

    f = plot_precision_recall_curves(a_roc_data, "Alzheimers Disease", OUTPUT_DIR)
    print(f"  [Created] {f.name}")

    f = plot_precision_recall_curves(s_roc_data, "Brain Stroke", OUTPUT_DIR)
    print(f"  [Created] {f.name}")

    # Actual vs. Predicted Confidence Distributions
    f = plot_actual_vs_predicted(p_xgb["y_test"], p_xgb["y_prob"], "XGBoost", "Parkinsons Disease", OUTPUT_DIR)
    print(f"  [Created] {f.name}")

    f = plot_actual_vs_predicted(a_tab["y_test"], a_tab["y_prob"], "TabTransformer", "Alzheimers Disease", OUTPUT_DIR)
    print(f"  [Created] {f.name}")

    f = plot_actual_vs_predicted(s_xgb["y_test"], s_xgb["y_prob"], "XGBoost", "Brain Stroke", OUTPUT_DIR)
    print(f"  [Created] {f.name}")

    # Existing Literature vs. Proposed Work Benchmark Graph
    p_lit = {
        "Little et al. (SVM)": 85.00,
        "Sakar et al. (Random Forest)": 87.20,
        "Deep MLP Baseline": 89.50,
        "NeuroTwin (XGBoost)": round(p_xgb["metrics"]["accuracy"] * 100, 2),
        "NeuroTwin (Proposed TabTransformer)": round(p_tab["metrics"]["accuracy"] * 100, 2),
    }
    f = plot_literature_comparison(p_lit, "Parkinsons Disease", OUTPUT_DIR)
    print(f"  [Created] {f.name}")

    a_lit = {
        "Marcus et al. (OASIS Baseline)": 82.50,
        "Battineni et al. (SVM)": 86.40,
        "Standard Deep Neural Net": 88.90,
        "NeuroTwin (XGBoost)": round(a_xgb["metrics"]["accuracy"] * 100, 2),
        "NeuroTwin (Proposed TabTransformer)": round(a_tab["metrics"]["accuracy"] * 100, 2),
    }
    f = plot_literature_comparison(a_lit, "Alzheimers Disease", OUTPUT_DIR)
    print(f"  [Created] {f.name}")

    s_lit = {
        "Shukla et al. (Decision Tree)": 84.60,
        "Clinical RF Baseline": 88.20,
        "Standard Deep MLP": 89.50,
        "NeuroTwin (Proposed TabTransformer)": round(s_tab["metrics"]["accuracy"] * 100, 2),
        "NeuroTwin (XGBoost)": round(s_xgb["metrics"]["accuracy"] * 100, 2),
    }
    f = plot_literature_comparison(s_lit, "Brain Stroke", OUTPUT_DIR)
    print(f"  [Created] {f.name}")

    # 5. EXPORT JSON AND LATEX TABLE
    print("\n[5/5] Exporting IEEE LaTeX Table and JSON Data...")
    json_path = OUTPUT_DIR / "ieee_results.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "results_by_disease": results_by_disease,
                "literature_comparison": {
                    "parkinsons": p_lit,
                    "alzheimers": a_lit,
                    "stroke": s_lit,
                },
                "plots": [p.name for p in OUTPUT_DIR.glob("*.png")],
            },
            fp,
            indent=2,
        )
    print(f"  [Saved] {json_path}")

    latex_table = format_ieee_latex_table(results_by_disease)
    latex_path = OUTPUT_DIR / "results_table.tex"
    with open(latex_path, "w", encoding="utf-8") as fp:
        fp.write(latex_table)
    print(f"  [Saved] {latex_path}")

    print("\n" + "=" * 70)
    print("All IEEE Paper Results, Graphs, and LaTeX Tables Successfully Generated!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_evaluations()
