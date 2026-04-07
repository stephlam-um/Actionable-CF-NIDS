# Actionable-CF-NIDS

**From black-box alerts to minimal actionable changes: SHAP feature pruning meets counterfactual recourse for network intrusion detection.**

---

## The Problem

ML-based intrusion detection models achieve high accuracy but produce opaque predictions. An analyst sees "this flow is malicious" with no reasoning. Existing XAI approaches either stop at feature attribution (SHAP bar charts that help data scientists but not analysts triaging alerts), or pipe SHAP values into LLMs for natural language summaries — producing fluent but potentially unfaithful explanations with no reliable way to evaluate correctness.

## The Approach

1. **SHAP-based feature pruning** — use global SHAP importance to identify the minimal feature subset that preserves detection performance, reducing the raw feature space (44 features for RoEduNet, ~78 for BCCC-CIC-IDS2017) to a compact set that retains near-baseline F1.
2. **Counterfactual explanations on the pruned space** — for each flagged alert, compute "what is the smallest change to this flow that would flip the model's prediction to benign?" Fewer features means faster, sparser, more interpretable counterfactuals.
3. **Deterministic template rendering** — map counterfactual outputs to structured analyst briefs tied to MITRE ATT&CK techniques. No LLM, no hallucination, fully faithful by construction.

The result: each alert comes with a structured brief showing *what the model is keying on* (SHAP) and *what would need to change to clear it* (counterfactual), in language an analyst can act on.

## What Makes This Different

- **No LLM faithfulness problem.** Explanations are deterministic templates filled with computed values. Every claim is verifiable and reproducible.
- **Counterfactuals are actionable.** Analysts get "the model would clear this alert if X changed by Y" instead of a SHAP bar chart they can't operationalize.
- **Feature pruning makes CFs practical.** Reducing the feature space makes CF generation faster, sparser, and more interpretable — a direct synergy between the two techniques that neither achieves alone.
- **All evaluation is automated and objective.** Validity, proximity, sparsity, plausibility, F1 — no subjective "is this explanation good?" metrics.
- **SOC-grounded case studies.** Analyst briefs mapped to MITRE ATT&CK, written from operational experience.

---

## Datasets

| Dataset | Features | Attack Classes | Format | Source |
|---|---|---|---|---|
| **RoEduNet-SIMARGL2021** | 44 (NetFlow v9) | DDoS (2 variants), PortScan, Benign | CSV | [Kaggle](https://www.kaggle.com/datasets/h2020simargl/simargl2021-network-intrusion-detection-dataset) |
| **BCCC-CIC-IDS2017** | ~78 (NTLFlowLyzer) | Brute Force, DoS, DDoS, Heartbleed, Web Attack, Infiltration, Botnet, Benign | CSV | [Kaggle](https://www.kaggle.com/datasets/bcccdatasets/intrusion-detection-datasets-bccc-cic-ids-2017) |

- **RoEduNet** has real network traffic from a Romanian academic network, 44 features, and 3 attack types. Cleaner and faster for development.
- **BCCC-CIC-IDS2017** is a re-extraction of the original CIC-IDS2017 PCAPs using NTLFlowLyzer. More attack classes (7+) and richer template variety, but inherits synthetic traffic limitations.

Start with RoEduNet for a cleaner development cycle. Use BCCC for richer attack variety, or train on both and report cross-dataset results.

---

## Project Structure

```
Actionable-CF-NIDS/
│
├── README.md
├── requirements.txt
├── config.yaml                    # Dataset paths, model hyperparams, feature count thresholds
├── .gitignore
├── LICENSE
│
├── data/
│   ├── raw/                       # Original dataset files (gitignored)
│   ├── processed/                 # Cleaned, encoded, train/test splits
│   └── feature_glossary.yaml      # Per-feature: name, description, type, min/max range, unit
│
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory data analysis + class distribution
│   ├── 02_feature_selection.ipynb  # SHAP-based feature pruning walkthrough
│   ├── 03_counterfactuals.ipynb   # CF generation, analysis, metric computation
│   └── 04_case_studies.ipynb      # End-to-end analyst-facing case studies
│
├── src/
│   ├── data/
│   │   ├── loader.py              # Load and validate raw dataset
│   │   └── preprocess.py          # Clean, encode, normalize, stratified split
│   │
│   ├── model/
│   │   ├── train.py               # Train XGBoost (full and reduced feature sets)
│   │   └── evaluate.py            # Per-class and aggregate classification metrics
│   │
│   ├── explain/
│   │   ├── shap_analysis.py       # SHAP value computation + global/per-class ranking
│   │   ├── feature_selector.py    # Top-k sweep, retrain, elbow detection
│   │   ├── counterfactual.py      # DiCE-based CF generation with feature constraints
│   │   └── templates.py           # Deterministic explanation templates + MITRE mapping
│   │
│   └── evaluation/
│       ├── model_metrics.py       # F1, precision, recall, ROC curves
│       ├── cf_metrics.py          # Validity, proximity, sparsity, plausibility
│       └── case_study.py          # Render structured analyst briefs
│
├── app/
│   └── streamlit_app.py           # Interactive demo dashboard
│
├── results/
│   ├── figures/                   # SHAP plots, CF comparisons, metric charts
│   ├── tables/                    # CSV exports of all evaluation results
│   └── case_studies/              # Rendered analyst briefs (markdown or HTML)
│
└── tests/
    ├── test_preprocess.py
    ├── test_feature_selector.py
    └── test_counterfactual.py
```

---

## Setup

```bash
git clone https://github.com/<your-username>/Actionable-CF-NIDS.git
cd Actionable-CF-NIDS

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download your chosen dataset from Kaggle and place CSV(s) in data/raw/
python -m src.data.preprocess
python -m src.model.train
python -m src.explain.shap_analysis
python -m src.explain.feature_selector
python -m src.explain.counterfactual

streamlit run app/streamlit_app.py
```

## Key Dependencies

```
python >= 3.10
xgboost >= 2.0
shap >= 0.43
dice-ml >= 0.11
pandas >= 2.0
scikit-learn >= 1.3
streamlit >= 1.30
matplotlib >= 3.8
seaborn >= 0.13
pyyaml
joblib
```

---

## References

- Mothilal, R.K., Sharma, A., & Tan, C. (2020). "Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations." *ACM FAT\* 2020.*
- Lundberg, S.M. & Lee, S.I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS 2017.*
- Szczepanski, M. et al. (2021). "The Proposition and Evaluation of the RoEduNet-SIMARGL2021 Network Intrusion Detection Dataset." *Sensors, 21(13), 4319.*
- Arreche, O. et al. (2024). "XAI-based Feature Selection for Network Intrusion Detection Systems."
- Shafi, M.M. et al. (2024). "NTLFlowLyzer: Towards generating an intrusion detection dataset and intruders behavior profiling." *Computers & Security, 148.*
- Szczepanski, M. et al. (2025). "Novel Actionable Counterfactual Explanations for Intrusion Detection Using Diffusion Models."
- MITRE ATT&CK Framework: https://attack.mitre.org/
- DiCE documentation: https://interpret.ml/DiCE/

## License

MIT
