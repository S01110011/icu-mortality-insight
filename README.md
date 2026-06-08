# ICU Mortality Insight

ICU Mortality Insight is a professional Python healthtech project for estimating in-hospital mortality risk in intensive care using structured clinical data.

The project trains machine-learning models, evaluates ROC-AUC and confusion-matrix metrics, and exports explainability artifacts. It uses synthetic ICU data by default, so it is safe for a public GitHub portfolio.

## Clinical Challenge

ICU teams need early visibility into patients at elevated mortality risk so they can prioritize review, escalation and care planning. This project models structured variables commonly available in critical-care datasets:

- Age and comorbidity burden
- Vital signs
- Glasgow Coma Scale
- Oxygenation and ventilation status
- Vasopressor use
- Lactate, creatinine, bilirubin, platelets and white blood cells
- ICU length of stay context

## Features

- Synthetic ICU cohort generation
- Clinical feature engineering
- Logistic Regression and Random Forest training
- Model selection by ROC-AUC
- ROC-AUC, average precision, recall, precision and F1
- Confusion matrix export
- Feature importance and SHAP-compatible explainability workflow
- Model artifact persistence with `joblib`
- Tests, lint and GitHub Actions CI

## Project Structure

```text
icu-mortality-insight/
├── data/
│   └── README.md
├── docs/
│   └── model_card.md
├── models/
│   └── README.md
├── reports/
│   └── README.md
├── src/
│   └── icu_mortality_insight/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── explainability.py
│       ├── features.py
│       ├── inference.py
│       ├── pipeline.py
│       └── train.py
├── tests/
│   ├── test_data.py
│   └── test_training.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── pyproject.toml
└── README.md
```

## Quickstart

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Generate synthetic ICU data:

```powershell
python -m icu_mortality_insight.data --output data/icu_synthetic.csv --rows 6000
```

Train and evaluate models:

```powershell
python -m icu_mortality_insight.train --data data/icu_synthetic.csv --model-out models/icu_mortality_model.joblib --report-dir reports
```

Run tests and lint:

```powershell
python -m pytest
python -m ruff check .
```

## Outputs

The training command creates:

- `models/icu_mortality_model.joblib`
- `reports/metrics.json`
- `reports/confusion_matrix.csv`
- `reports/model_comparison.csv`
- `reports/feature_importance.csv`

## Optional SHAP Support

The project is designed for SHAP-style model explainability. To enable native SHAP exports, install the optional extra:

```powershell
pip install -e ".[explainability]"
```

If SHAP is not installed, the project still exports permutation importance as a stable explainability fallback.

## Safety Note

This is an educational portfolio project, not a medical device or clinical decision support system. Real deployment would require clinical validation, calibration, bias testing, monitoring, auditability and governance.

