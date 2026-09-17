# NeuroTwin: Multi-Modal Digital Twin for Neurological Disease Progression & Clinical Inference

[![IEEE Paper Deliverables](https://img.shields.io/badge/IEEE%20Paper-Results%20Ready-success)](./artifacts/paper_results)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![React 19](https://img.shields.io/badge/React-19.2-cyan.svg)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg)](https://fastapi.tiangolo.com/)

NeuroTwin is a personalized neurological digital twin platform that integrates multi-modal clinical biometrics, cognitive assessments, and deep learning architectures to predict disease trajectory, monitor therapeutic response, and classify neurodegenerative disorders including **Parkinson's Disease** and **Alzheimer's Disease**.

---

## 🔬 Experimental Evaluation & IEEE Publication Results

The complete checklist of experimental deliverables required for IEEE conference/journal publication has been generated and validated across multiple architectural paradigms:

### Benchmark Performance Summary

| Disease Domain | Model Architecture | Accuracy | Precision | Sensitivity / Recall | Specificity | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Parkinson's Disease** | Baseline (Logistic Regression) | 76.92% | 95.45% | 72.41% | 90.00% | 82.35% | 0.9276 |
| | XGBoost Classifier | **92.31%** | 93.33% | **96.55%** | 80.00% | **94.92%** | **0.9828** |
| | **Proposed TabTransformer** | 87.18% | 90.00% | 93.10% | 70.00% | 91.53% | 0.9448 |
| **Alzheimer's Disease** | Baseline (Logistic Regression) | 80.85% | 78.95% | 75.00% | 85.19% | 76.92% | 0.9056 |
| | XGBoost Classifier | **82.98%** | 83.33% | **75.00%** | **88.89%** | **78.95%** | **0.9352** |
| | **Proposed TabTransformer** | 80.85% | 82.35% | 70.00% | **88.89%** | 75.68% | 0.9074 |

---

## 📈 Generated IEEE Publication Figures (300 DPI)

All figures are rendered at 300 DPI following IEEE publication style guidelines and stored in [`artifacts/paper_results/`](./artifacts/paper_results/):

1. **Dual-Panel Training vs. Validation Convergence Curves** (`training_val_curves_*.png`): Epoch-by-epoch loss reduction and accuracy progression for deep learning TabTransformer models.
2. **Comparative ROC Curves** (`roc_curves_*.png`): Multi-model True Positive Rate vs. False Positive Rate with annotated AUC values.
3. **Precision–Recall Curves** (`precision_recall_curves_*.png`): Classification precision vs. recall trade-offs with Average Precision (AP) scores.
4. **Diagnostic Confusion Matrices** (`confusion_matrix_*.png`): Seaborn heatmaps detailing true positives, false positives, true negatives, and false alarms.
5. **Actual vs. Predicted Distribution** (`actual_vs_predicted_*.png`): Calibrated probability density across healthy controls and patient cohorts.
6. **Literature Comparison Graph** (`literature_comparison_*.png`): Benchmarking proposed NeuroTwin architectures against existing published studies (Little et al., Sakar et al., Marcus et al., Battineni et al.).

---

## 📜 Publication LaTeX Table

Ready-to-use LaTeX table for IEEE double-column templates available at [`artifacts/paper_results/results_table.tex`](./artifacts/paper_results/results_table.tex):

```latex
\begin{table*}[htbp]
\caption{Comparative Performance of NeuroTwin Models on Clinical Neurological Benchmarks}
\label{tab:neurotwin_results}
\centering
\small
\begin{tabular}{llcccccc}
\hline
\textbf{Disease Domain} & \textbf{Model Architecture} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Sensitivity} & \textbf{Specificity} & \textbf{F1-Score} & \textbf{ROC-AUC} \\
\hline
Parkinson's Disease & Baseline (LogReg) & 76.92\% & 95.45\% & 72.41\% & 90.00\% & 82.35\% & 0.9276 \\
 & XGBoost Classifier & 92.31\% & 93.33\% & 96.55\% & 80.00\% & 94.92\% & 0.9828 \\
 & \textbf{Proposed TabTransformer} & \textbf{87.18\%} & \textbf{90.00\%} & \textbf{93.10\%} & \textbf{70.00\%} & \textbf{91.53\%} & \textbf{0.9448} \\
\hline
Alzheimer's Disease & Baseline (LogReg) & 80.85\% & 78.95\% & 75.00\% & 85.19\% & 76.92\% & 0.9056 \\
 & XGBoost Classifier & 82.98\% & 83.33\% & 75.00\% & 88.89\% & 78.95\% & 0.9352 \\
 & \textbf{Proposed TabTransformer} & \textbf{80.85\%} & \textbf{82.35\%} & \textbf{70.00\%} & \textbf{88.89\%} & \textbf{75.68\%} & \textbf{0.9074} \\
\hline
\end{tabular}
\end{table*}
```

---

## 🛠️ Project Structure

```
NeuroTwin/
├── artifacts/
│   └── paper_results/          # 300-DPI publication figures, JSON, and LaTeX tables
├── backend/
│   ├── app/
│   │   ├── evaluation/         # IEEE metrics, plotting, and LaTeX formatting
│   │   ├── models/             # Parkinson's, Alzheimer's & Swin Transformer MRI models
│   │   ├── preprocessing/      # Loaders, scalers, and data validators
│   │   └── main.py             # FastAPI backend with static plot serving & CORS
│   ├── datasets/raw/clinical/  # UCI Parkinson's and OASIS Alzheimer's datasets
│   └── scripts/                # Master generation script (generate_paper_results.py)
└── frontend/
    ├── public/plots/           # Static publication plots for web viewing
    └── src/                    # React 19 + Vite IEEE Paper Results Studio
```

---

## 🚀 Getting Started

### 1. Backend Setup
```bash
cd backend
pip install -r requirements.txt
pip install scikit-learn xgboost torch scipy matplotlib seaborn

# Run API server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Visit **http://localhost:5173** to access the interactive **IEEE Results & Research Studio**.

### 3. Regenerate Paper Deliverables
```bash
python backend/scripts/generate_paper_results.py
```
