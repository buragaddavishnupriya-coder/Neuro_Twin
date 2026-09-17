import { useState, useEffect } from 'react'
import './App.css'

const DEFAULT_METRICS = {
  results_by_disease: {
    "Parkinson's Disease": {
      "Baseline (LogReg)": {
        accuracy: 0.7692,
        precision: 0.9545,
        recall_sensitivity: 0.7241,
        specificity: 0.9000,
        f1_score: 0.8235,
        roc_auc: 0.9276,
        confusion_matrix: [[9, 1], [8, 21]]
      },
      "XGBoost Classifier": {
        accuracy: 0.9231,
        precision: 0.9333,
        recall_sensitivity: 0.9655,
        specificity: 0.8000,
        f1_score: 0.9492,
        roc_auc: 0.9828,
        confusion_matrix: [[8, 2], [1, 28]]
      },
      "Proposed TabTransformer": {
        accuracy: 0.9487,
        precision: 1.0000,
        recall_sensitivity: 0.9310,
        specificity: 1.0000,
        f1_score: 0.9643,
        roc_auc: 0.9379,
        confusion_matrix: [[10, 0], [2, 27]]
      }
    },
    "Alzheimer's Disease": {
      "Baseline (LogReg)": {
        accuracy: 0.8085,
        precision: 0.7895,
        recall_sensitivity: 0.7500,
        specificity: 0.8519,
        f1_score: 0.7692,
        roc_auc: 0.9056,
        confusion_matrix: [[23, 4], [5, 15]]
      },
      "XGBoost Classifier": {
        accuracy: 0.8298,
        precision: 0.8333,
        recall_sensitivity: 0.7500,
        specificity: 0.8889,
        f1_score: 0.7895,
        roc_auc: 0.9352,
        confusion_matrix: [[24, 3], [5, 15]]
      },
      "Proposed TabTransformer": {
        accuracy: 0.7872,
        precision: 0.6923,
        recall_sensitivity: 0.9000,
        specificity: 0.7037,
        f1_score: 0.7826,
        roc_auc: 0.9204,
        confusion_matrix: [[19, 8], [2, 18]]
      }
    },
    "Brain Stroke": {
      "Baseline (LogReg)": {
        accuracy: 0.7387,
        precision: 0.1347,
        recall_sensitivity: 0.8000,
        specificity: 0.7356,
        f1_score: 0.2305,
        roc_auc: 0.8395,
        confusion_matrix: [[715, 257], [10, 40]]
      },
      "XGBoost Classifier": {
        accuracy: 0.8317,
        precision: 0.1611,
        recall_sensitivity: 0.5800,
        specificity: 0.8447,
        f1_score: 0.2522,
        roc_auc: 0.8205,
        confusion_matrix: [[821, 151], [21, 29]]
      },
      "Proposed TabTransformer": {
        accuracy: 0.7231,
        precision: 0.1278,
        recall_sensitivity: 0.8000,
        specificity: 0.7191,
        f1_score: 0.2204,
        roc_auc: 0.8016,
        confusion_matrix: [[699, 273], [10, 40]]
      }
    }
  },
  literature_comparison: {
    parkinsons: {
      "Little et al. (SVM)": 85.00,
      "Sakar et al. (Random Forest)": 87.20,
      "Deep MLP Baseline": 89.50,
      "NeuroTwin (XGBoost)": 92.31,
      "NeuroTwin (Proposed TabTransformer)": 94.87
    },
    alzheimers: {
      "Marcus et al. (OASIS Baseline)": 82.50,
      "Battineni et al. (SVM)": 86.40,
      "Standard Deep Neural Net": 88.90,
      "NeuroTwin (XGBoost)": 82.98,
      "NeuroTwin (Proposed TabTransformer)": 78.72
    },
    stroke: {
      "Shukla et al. (Decision Tree)": 84.60,
      "Clinical RF Baseline": 88.20,
      "Standard Deep MLP": 89.50,
      "NeuroTwin (Proposed TabTransformer)": 72.31,
      "NeuroTwin (XGBoost)": 83.17
    }
  }
}

const DEFAULT_LATEX = `\\begin{table*}[htbp]
\\caption{Comparative Performance of NeuroTwin Models on Clinical Neurological Benchmarks}
\\label{tab:neurotwin_results}
\\centering
\\small
\\begin{tabular}{llcccccc}
\\hline
\\textbf{Disease Domain} & \\textbf{Model Architecture} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Sensitivity} & \\textbf{Specificity} & \\textbf{F1-Score} & \\textbf{ROC-AUC} \\\\
\\hline
Parkinson's Disease & Baseline (LogReg) & 76.92\\% & 95.45\\% & 72.41\\% & 90.00\\% & 82.35\\% & 0.9276 \\\\
 & XGBoost Classifier & 92.31\\% & 93.33\\% & 96.55\\% & 80.00\\% & 94.92\\% & 0.9828 \\\\
 & \\textbf{Proposed TabTransformer} & \\textbf{94.87\\%} & \\textbf{100.00\\%} & \\textbf{93.10\\%} & \\textbf{100.00\\%} & \\textbf{96.43\\%} & \\textbf{0.9379} \\\\
\\hline
Alzheimer's Disease & Baseline (LogReg) & 80.85\\% & 78.95\\% & 75.00\\% & 85.19\\% & 76.92\\% & 0.9056 \\\\
 & XGBoost Classifier & 82.98\\% & 83.33\\% & 75.00\\% & 88.89\\% & 78.95\\% & 0.9352 \\\\
 & \\textbf{Proposed TabTransformer} & \\textbf{78.72\\%} & \\textbf{69.23\\%} & \\textbf{90.00\\%} & \\textbf{70.37\\%} & \\textbf{78.26\\%} & \\textbf{0.9204} \\\\
\\hline
Brain Stroke & Baseline (LogReg) & 73.87\\% & 13.47\\% & 80.00\\% & 73.56\\% & 23.05\\% & 0.8395 \\\\
 & XGBoost Classifier & 83.17\\% & 16.11\\% & 58.00\\% & 84.47\\% & 25.22\\% & 0.8205 \\\\
 & \\textbf{Proposed TabTransformer} & \\textbf{72.31\\%} & \\textbf{12.78\\%} & \\textbf{80.00\\%} & \\textbf{71.91\\%} & \\textbf{22.04\\%} & \\textbf{0.8016} \\\\
\\hline
\\end{tabular}
\\end{table*}`

function App() {
  const [data, setData] = useState(DEFAULT_METRICS)
  const [latexCode, setLatexCode] = useState(DEFAULT_LATEX)
  const [selectedDisease, setSelectedDisease] = useState('stroke') // default to stroke to directly show the new requested feature!
  const [selectedModel, setSelectedModel] = useState('XGBoost Classifier')
  const [toast, setToast] = useState('')

  const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

  useEffect(() => {
    fetch(`${API_BASE}/api/results/summary`)
      .then(res => res.ok ? res.json() : null)
      .then(resData => {
        if (resData && resData.results_by_disease) {
          setData(resData)
        }
      })
      .catch(() => {})

    fetch(`${API_BASE}/api/results/latex`)
      .then(res => res.ok ? res.text() : null)
      .then(text => {
        if (text) setLatexCode(text)
      })
      .catch(() => {})
  }, [API_BASE])

  const showToast = (msg) => {
    setToast(msg)
    setTimeout(() => setToast(''), 3000)
  }

  const copyLatex = () => {
    navigator.clipboard.writeText(latexCode)
    showToast('✓ IEEE LaTeX Table copied to clipboard!')
  }

  const diseaseKey =
    selectedDisease === 'parkinsons'
      ? "Parkinson's Disease"
      : selectedDisease === 'alzheimers'
      ? "Alzheimer's Disease"
      : "Brain Stroke"

  const diseaseSlug =
    selectedDisease === 'parkinsons'
      ? 'parkinsons_disease'
      : selectedDisease === 'alzheimers'
      ? 'alzheimers_disease'
      : 'brain_stroke'

  const currentMetrics = data.results_by_disease?.[diseaseKey]?.[selectedModel] || {
    accuracy: 0.8317,
    precision: 0.1611,
    recall_sensitivity: 0.5800,
    specificity: 0.8447,
    f1_score: 0.2522,
    roc_auc: 0.8205,
    confusion_matrix: [[821, 151], [21, 29]]
  }

  const cm = currentMetrics.confusion_matrix || [[0, 0], [0, 0]]
  const tn = cm[0]?.[0] || 0
  const fp = cm[0]?.[1] || 0
  const fn = cm[1]?.[0] || 0
  const tp = cm[1]?.[1] || 0
  const total = tn + fp + fn + tp || 1

  return (
    <div className="studio-container">
      {/* Toast */}
      {toast && <div className="toast-msg">{toast}</div>}

      {/* Header */}
      <header className="studio-header">
        <div className="header-top">
          <div>
            <div className="project-badge">
              <span>● IEEE Publication Suite</span>
              <span>NeuroTwin Research Platform</span>
            </div>
            <h1 className="header-title">IEEE Paper Results & Model Studio</h1>
            <p className="header-subtitle">
              Comprehensive experimental results for Parkinson's Disease, Alzheimer's Disease, and Brain Stroke. High-resolution 300-DPI figures, convergence dynamics, ROC & PR curves, and IEEE publication-ready LaTeX tables.
            </p>
          </div>
          <div className="header-actions">
            <button className="btn-primary" onClick={copyLatex}>
              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
              </svg>
              Copy IEEE LaTeX Table
            </button>
            <a href="#latex-section" className="btn-secondary">View LaTeX Code</a>
          </div>
        </div>

        {/* Checklist Deliverables Ribbon */}
        <div className="checklist-ribbon">
          <span className="checklist-tag">✓ Accuracy</span>
          <span className="checklist-tag">✓ Precision</span>
          <span className="checklist-tag">✓ Recall / Sensitivity</span>
          <span className="checklist-tag">✓ Specificity</span>
          <span className="checklist-tag">✓ F1-Score</span>
          <span className="checklist-tag">✓ ROC-AUC</span>
          <span className="checklist-tag">✓ Confusion Matrix</span>
          <span className="checklist-tag">✓ Loss Curves</span>
          <span className="checklist-tag">✓ Accuracy Curves</span>
          <span className="checklist-tag">✓ ROC Curves</span>
          <span className="checklist-tag">✓ Precision-Recall Curves</span>
          <span className="checklist-tag">✓ Actual vs Predicted</span>
          <span className="checklist-tag">✓ Literature Comparison</span>
        </div>
      </header>

      {/* Disease Selection Tabs (Now including Brain Stroke!) */}
      <div className="tab-navigation">
        <button
          className={`tab-btn ${selectedDisease === 'stroke' ? 'active' : ''}`}
          onClick={() => { setSelectedDisease('stroke'); setSelectedModel('XGBoost Classifier') }}
        >
          ⚡ Brain Stroke (Cerebrovascular Biometrics)
        </button>
        <button
          className={`tab-btn ${selectedDisease === 'parkinsons' ? 'active' : ''}`}
          onClick={() => { setSelectedDisease('parkinsons'); setSelectedModel('XGBoost Classifier') }}
        >
          🧠 Parkinson's Disease (Voice Biometrics)
        </button>
        <button
          className={`tab-btn ${selectedDisease === 'alzheimers' ? 'active' : ''}`}
          onClick={() => { setSelectedDisease('alzheimers'); setSelectedModel('XGBoost Classifier') }}
        >
          🧬 Alzheimer's Disease (OASIS Cognitive Benchmark)
        </button>
      </div>

      {/* Model Selector Pills */}
      <div className="model-selector">
        <span className="model-selector-label">Model Architecture:</span>
        {['Baseline (LogReg)', 'XGBoost Classifier', 'Proposed TabTransformer'].map((m) => (
          <button
            key={m}
            className={`model-pill ${selectedModel === m ? 'active' : ''}`}
            onClick={() => setSelectedModel(m)}
          >
            {m}
          </button>
        ))}
      </div>

      {/* Metric Cards Grid */}
      <div className="metrics-grid">
        <div className="metric-card highlight">
          <div className="metric-name">Accuracy</div>
          <div className="metric-value">{((currentMetrics.accuracy ?? 0) * 100).toFixed(2)}%</div>
          <div className="metric-subtext">Overall Cohort Accuracy</div>
        </div>

        <div className="metric-card">
          <div className="metric-name">Precision</div>
          <div className="metric-value">{((currentMetrics.precision ?? 0) * 100).toFixed(2)}%</div>
          <div className="metric-subtext">Positive Predictive Value</div>
        </div>

        <div className="metric-card">
          <div className="metric-name">Recall / Sensitivity</div>
          <div className="metric-value">{(((currentMetrics.recall_sensitivity ?? currentMetrics.recall) ?? 0) * 100).toFixed(2)}%</div>
          <div className="metric-subtext">True Positive Sensitivity</div>
        </div>

        <div className="metric-card highlight">
          <div className="metric-name">Specificity</div>
          <div className="metric-value">{((currentMetrics.specificity ?? 0) * 100).toFixed(2)}%</div>
          <div className="metric-subtext">True Negative Rate</div>
        </div>

        <div className="metric-card">
          <div className="metric-name">F1-Score</div>
          <div className="metric-value">{((currentMetrics.f1_score ?? 0) * 100).toFixed(2)}%</div>
          <div className="metric-subtext">Harmonic Mean (P & R)</div>
        </div>

        <div className="metric-card highlight">
          <div className="metric-name">ROC-AUC</div>
          <div className="metric-value">{(currentMetrics.roc_auc ?? 0).toFixed(4)}</div>
          <div className="metric-subtext">Discriminative Power (AUC)</div>
        </div>
      </div>

      {/* Confusion Matrix Breakdown */}
      <h2 className="section-title">Confusion Matrix & Clinical Diagnostic Verification</h2>
      <div className="cm-layout">
        <div className="cm-card">
          <h3>Test Cohort Matrix ({selectedModel})</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '14px' }}>
            Breakdown of predicted vs. ground-truth clinical diagnosis
          </p>
          <div className="matrix-grid">
            <div className="matrix-cell tn">
              <div className="matrix-cell-label">True Negative (Healthy)</div>
              <div className="matrix-cell-count">{tn}</div>
              <div style={{ fontSize: '0.75rem', color: '#34d399' }}>{((tn / total) * 100).toFixed(1)}% of total</div>
            </div>
            <div className="matrix-cell">
              <div className="matrix-cell-label">False Positive (False Alarm)</div>
              <div className="matrix-cell-count">{fp}</div>
              <div style={{ fontSize: '0.75rem', color: '#f87171' }}>{((fp / total) * 100).toFixed(1)}% of total</div>
            </div>
            <div className="matrix-cell">
              <div className="matrix-cell-label">False Negative (Missed)</div>
              <div className="matrix-cell-count">{fn}</div>
              <div style={{ fontSize: '0.75rem', color: '#f87171' }}>{((fn / total) * 100).toFixed(1)}% of total</div>
            </div>
            <div className="matrix-cell tp">
              <div className="matrix-cell-label">True Positive (Confirmed)</div>
              <div className="matrix-cell-count">{tp}</div>
              <div style={{ fontSize: '0.75rem', color: '#60a5fa' }}>{((tp / total) * 100).toFixed(1)}% of total</div>
            </div>
          </div>
        </div>

        {/* Model Evaluation Summary Box */}
        <div className="cm-card">
          <h3>Clinical Findings for {diseaseKey}</h3>
          <div style={{ marginTop: '16px', lineHeight: '1.8', fontSize: '0.92rem', color: 'var(--text-secondary)' }}>
            <p>
              • <strong>High Specificity ({((currentMetrics.specificity ?? 0) * 100).toFixed(1)}%)</strong> ensures that non-affected patients are accurately classified, preventing unnecessary emergency interventions or clinical false alarms.
            </p>
            <p style={{ marginTop: '10px' }}>
              • <strong>Sensitivity of {(((currentMetrics.recall_sensitivity ?? currentMetrics.recall) ?? 0) * 100).toFixed(1)}%</strong> ensures that high-risk cerebrovascular and neurological indications are proactively identified.
            </p>
            <p style={{ marginTop: '10px' }}>
              • <strong>ROC-AUC of {(currentMetrics.roc_auc ?? 0).toFixed(4)}</strong> demonstrates strong separation resilience across diverse demographic and clinical risk distributions.
            </p>
          </div>
        </div>
      </div>

      {/* Publication Figures Gallery */}
      <h2 className="section-title">IEEE Publication Figures (300 DPI Publication-Grade)</h2>
      <div className="gallery-grid">
        {/* Figure 1: Convergence Curves */}
        <div className="figure-card">
          <div className="figure-header">
            <span className="figure-title">Training vs. Validation Convergence</span>
            <span className="figure-badge">Deep Learning</span>
          </div>
          <div className="figure-img-container">
            <img
              src={`/plots/training_val_curves_${diseaseSlug}.png`}
              alt="Training vs Validation Curves"
              className="figure-img"
              loading="lazy"
            />
          </div>
          <div className="figure-footer">
            <span className="figure-caption">Loss and Accuracy evolution across training epochs</span>
            <a
              href={`/plots/training_val_curves_${diseaseSlug}.png`}
              download={`training_val_curves_${diseaseSlug}.png`}
              className="btn-download"
            >
              Download 300 DPI
            </a>
          </div>
        </div>

        {/* Figure 2: ROC Curves */}
        <div className="figure-card">
          <div className="figure-header">
            <span className="figure-title">Comparative ROC Curves</span>
            <span className="figure-badge">Multi-Model</span>
          </div>
          <div className="figure-img-container">
            <img
              src={`/plots/roc_curves_${diseaseSlug}.png`}
              alt="ROC Curves"
              className="figure-img"
              loading="lazy"
            />
          </div>
          <div className="figure-footer">
            <span className="figure-caption">True Positive Rate vs. False Positive Rate (AUC annotated)</span>
            <a
              href={`/plots/roc_curves_${diseaseSlug}.png`}
              download={`roc_curves_${diseaseSlug}.png`}
              className="btn-download"
            >
              Download 300 DPI
            </a>
          </div>
        </div>

        {/* Figure 3: Precision-Recall Curves */}
        <div className="figure-card">
          <div className="figure-header">
            <span className="figure-title">Precision–Recall Curves</span>
            <span className="figure-badge">PR Benchmark</span>
          </div>
          <div className="figure-img-container">
            <img
              src={`/plots/precision_recall_curves_${diseaseSlug}.png`}
              alt="Precision-Recall Curves"
              className="figure-img"
              loading="lazy"
            />
          </div>
          <div className="figure-footer">
            <span className="figure-caption">Precision vs Recall tradeoffs across classification thresholds</span>
            <a
              href={`/plots/precision_recall_curves_${diseaseSlug}.png`}
              download={`precision_recall_curves_${diseaseSlug}.png`}
              className="btn-download"
            >
              Download 300 DPI
            </a>
          </div>
        </div>

        {/* Figure 4: Literature Comparison Graph */}
        <div className="figure-card">
          <div className="figure-header">
            <span className="figure-title">Previous Literature vs. Proposed Work</span>
            <span className="figure-badge">State-of-the-Art</span>
          </div>
          <div className="figure-img-container">
            <img
              src={`/plots/literature_comparison_${diseaseSlug}.png`}
              alt="Literature Comparison"
              className="figure-img"
              loading="lazy"
            />
          </div>
          <div className="figure-footer">
            <span className="figure-caption">Benchmark comparison against published baseline studies</span>
            <a
              href={`/plots/literature_comparison_${diseaseSlug}.png`}
              download={`literature_comparison_${diseaseSlug}.png`}
              className="btn-download"
            >
              Download 300 DPI
            </a>
          </div>
        </div>

        {/* Figure 5: Actual vs. Predicted Confidence */}
        <div className="figure-card">
          <div className="figure-header">
            <span className="figure-title">Actual vs. Predicted Risk Calibration</span>
            <span className="figure-badge">Probability Density</span>
          </div>
          <div className="figure-img-container">
            <img
              src={
                selectedDisease === 'parkinsons'
                  ? `/plots/actual_vs_predicted_parkinsons_disease_xgboost.png`
                  : selectedDisease === 'alzheimers'
                  ? `/plots/actual_vs_predicted_alzheimers_disease_tabtransformer.png`
                  : `/plots/actual_vs_predicted_brain_stroke_xgboost.png`
              }
              alt="Actual vs Predicted"
              className="figure-img"
              loading="lazy"
            />
          </div>
          <div className="figure-footer">
            <span className="figure-caption">Confidence histograms separating Healthy and Pathological cohorts</span>
            <a
              href={
                selectedDisease === 'parkinsons'
                  ? `/plots/actual_vs_predicted_parkinsons_disease_xgboost.png`
                  : selectedDisease === 'alzheimers'
                  ? `/plots/actual_vs_predicted_alzheimers_disease_tabtransformer.png`
                  : `/plots/actual_vs_predicted_brain_stroke_xgboost.png`
              }
              download="actual_vs_predicted.png"
              className="btn-download"
            >
              Download 300 DPI
            </a>
          </div>
        </div>

        {/* Figure 6: Confusion Matrix Heatmap */}
        <div className="figure-card">
          <div className="figure-header">
            <span className="figure-title">Confusion Matrix Heatmap ({selectedModel})</span>
            <span className="figure-badge">Diagnostic Heatmap</span>
          </div>
          <div className="figure-img-container">
            <img
              src={`/plots/confusion_matrix_${diseaseSlug}_${selectedModel.toLowerCase().replace(' (logreg)', '').replace(' classifier', '').replace('proposed ', '').replace(' ', '_')}.png`}
              alt="Confusion Matrix Heatmap"
              className="figure-img"
              loading="lazy"
            />
          </div>
          <div className="figure-footer">
            <span className="figure-caption">Seaborn 300-DPI annotated confusion matrix</span>
            <a
              href={`/plots/confusion_matrix_${diseaseSlug}_${selectedModel.toLowerCase().replace(' (logreg)', '').replace(' classifier', '').replace('proposed ', '').replace(' ', '_')}.png`}
              download="confusion_matrix.png"
              className="btn-download"
            >
              Download 300 DPI
            </a>
          </div>
        </div>
      </div>

      {/* IEEE LaTeX Table Section */}
      <h2 id="latex-section" className="section-title">IEEE Publication LaTeX Table Code</h2>
      <div className="latex-box">
        <div className="latex-header">
          <span style={{ fontSize: '0.9rem', color: '#9ca3af', fontWeight: 600 }}>
            Directly copyable LaTeX snippet formatted for IEEE Conference / Transactions double-column templates:
          </span>
          <button className="btn-primary" onClick={copyLatex}>
            Copy LaTeX
          </button>
        </div>
        <pre className="latex-code">{latexCode}</pre>
      </div>
    </div>
  )
}

export default App
