# Credit Risk Modelling

This repository contains a credit-risk modelling prototype for predicting loan approval outcomes using customer credit bureau and internal product data. It combines data ingestion, exploratory analysis, feature screening, baseline model training, and a Streamlit inference application.

## Project Goals

- Understand the structure and business meaning of the available credit-risk data.
- Join internal product data with CIBIL bureau data via `PROSPECTID`.
- Clean, filter, and encode numerical and categorical features.
- Train and compare baseline classifiers for the `Approved_Flag` target.
- Expose a simple decision-support interface using a trained model.

## Data

The repository includes both raw Excel files and compressed `.zst` archives for:

- `data/internal_product.xlsx` / `data/internal_product.xlsx.zst`
- `data/cibil_score.xlsx` / `data/cibil_score.xlsx.zst`
- `data/feature_target_description.xlsx` / `data/feature_target_description.xlsx.zst`

A SQLite database artifact is also available at `data/credit_modelling.db`.

The modelling target is `Approved_Flag`. Relevant credit-risk concepts are documented in [domain_knowledge.md](domain_knowledge.md).

## Workflow

1. **Data ingestion**
   - Load the source Excel datasets with pandas.
   - Persist tables in SQLite for repeatable analysis.
   - Use [scripts/data/ingestion_db.py](scripts/data/ingestion_db.py) as the primary ingestion entry point.

2. **Data analysis and preparation**
   - Merge the internal and CIBIL datasets on `PROSPECTID`.
   - Separate categorical and numerical features.
   - Handle sentinel values such as `-99999` used for missing data.
   - Perform chi-square testing for categorical features, ANOVA for numerical features, and VIF filtering for multicollinearity.
   - Encode `EDUCATION` ordinally and one-hot encode nominal categories.
   - Use [scripts/data/data_processing.py](scripts/data/data_processing.py) for reusable preprocessing helpers.

3. **Model training**
   - Train baseline classifiers, compare evaluation metrics, and review confusion matrices.
   - Use [scripts/model_creation/train.py](scripts/model_creation/train.py) for training.
   - Use [scripts/model_creation/model_evaluation.py](scripts/model_creation/model_evaluation.py) for model assessment.

4. **Inference and application**
   - Generate predictions with the trained model in `models/predict_loan_possibility_model.pkl`.
   - Run the Streamlit UI in [app.py](app.py) for interactive loan risk assessment.
   - A simplified inference script is available at [scripts/inference/predict_loan_possibility.py](scripts/inference/predict_loan_possibility.py).

## What Has Been Achieved

- Added the core dataset files and a SQLite database artifact.
- Built a reusable ingestion script and a data-ingestion notebook.
- Merged CIBIL and internal product tables and documented feature roles.
- Implemented missing-value handling and categorical encoding helpers.
- Added statistical feature-screening utilities for categorical and numerical covariates.
- Implemented baseline model training and evaluation experiments.
- Included a Streamlit application for model inference.
- Stored the current trained model at `models/predict_loan_possibility_model.pkl`.

## Work In Progress

- Consolidate exploratory analysis and chart-driven business interpretation.
- Tune the XGBoost baseline using systematic hyperparameter search.
- Evaluate additional metrics such as precision, recall, F1-score, and class-specific performance.
- Add targeted feature engineering, scaling, and robustness checks.
- Harden the training pipeline for repeatable `train.py` execution.
- Improve the data ingestion/refactoring path for fresh repository setup.
- Expand the Streamlit app with clearer guidance and deployment-ready packaging.

## Repository Structure

```text
.
├── app.py                  # Streamlit inference app
├── assets/                 # Saved visuals and artifact images
├── data/                   # Source Excel files, compressed archives, and SQLite DB
├── models/                 # Trained model artifacts
├── notebooks/              # Exploratory and modelling notebooks
├── scripts/
│   ├── data/               # Ingestion and preprocessing helpers
│   ├── inference/          # Prediction and inference helpers
│   └── model_creation/     # Training and evaluation scripts
├── domain_knowledge.md     # Credit-risk terminology and notes
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10 or newer
- Jupyter Notebook or JupyterLab
- pandas, NumPy, SciPy, statsmodels, SQLAlchemy, scikit-learn, XGBoost, Streamlit, joblib
- `zstd` if you need to decompress `.zst` source files locally

Install the dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy statsmodels sqlalchemy scikit-learn xgboost streamlit joblib openpyxl
```

If you only have the compressed source files, decompress them into `data/`:

```bash
zstd -d data/internal_product.xlsx.zst
zstd -d data/cibil_score.xlsx.zst
zstd -d data/feature_target_description.xlsx.zst
```

### Script-driven execution

1. `python scripts/data/ingestion_db.py`
2. `python scripts/data/data_processing.py`
3. `python scripts/model_creation/train.py`
4. `python scripts/model_creation/model_evaluation.py`

To launch the loan risk assessment UI:

```bash
streamlit run app.py
```

## Current Status

Prototype complete through initial model training and a Streamlit inference interface. Next steps are a reproducible training pipeline, stronger model tuning, and a deployment-ready application.

## NOTE

This repository is an analytical and educational prototype. Its predictions and modelling decisions should not be used for real lending decisions without appropriate validation, governance, monitoring, fairness assessment, security review, and regulatory approval.
