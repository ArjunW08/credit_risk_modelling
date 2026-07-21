# Credit Risk Modelling

This project develops a credit-risk modelling workflow for predicting loan approval outcomes from customer credit and internal product information. It brings together data ingestion, exploratory and statistical analysis, feature selection, categorical encoding, baseline model training, and evaluation.

The project is currently in the research and prototyping stage. The notebooks contain the main analysis and modelling work, while reusable scripts and a user-facing application are still being developed.

## Project Goals

- Understand the structure and business meaning of the available credit-risk data.
- Combine internal product data with CIBIL credit-score data using the prospect identifier.
- Clean and select useful numerical and categorical features.
- Compare baseline classification models for the `Approved_Flag` target.
- Improve the best baseline model and make predictions available through a simple application.

## Data

The project uses the following data sources:

- `internal_product.xlsx`: internal customer and product-level information.
- `cibil_score.xlsx`: credit history and bureau-related features.
- `feature_target_description.xlsx`: feature and target descriptions.

The source files currently stored in the repository are compressed as `.xlsx.zst` archives. They need to be decompressed into `.xlsx` files before running the current ingestion workflow. Raw Excel files are ignored by Git through [.gitignore](.gitignore), while the SQLite database generated from the data is available at [data/credit_modelling.db](data/credit_modelling.db).

The modelling target is `Approved_Flag`. The project also documents relevant credit-risk concepts such as DPD, PAR, NPA, GNPA, NNPA, and credit-card utilization in [domain_knowledge.md](domain_knowledge.md).

## Workflow

1. **Data ingestion**
   - Load the Excel datasets with pandas.
   - Store the source tables in SQLite for repeatable access.
   - Initial ingestion logic is available in [notebooks/data_ingestion.ipynb](notebooks/data_ingestion.ipynb) and [scripts/data_analysis/ingestion_db.py](scripts/data_analysis/ingestion_db.py).

2. **Data analysis and preparation**
   - Merge the internal and CIBIL datasets on `PROSPECTID`.
   - Treat categorical and numerical features separately.
   - Handle the `-99999` sentinel used for missing values.
   - Use chi-square tests for categorical features, ANOVA for numerical features against the categorical target, and sequential VIF filtering for multicollinearity.
   - Encode `EDUCATION` ordinally and apply one-hot encoding to nominal features.
   - The analysis is documented in [notebooks/dataset_analysis.ipynb](notebooks/dataset_analysis.ipynb), with reusable functions in [scripts/data_analysis/data_processing.py](scripts/data_analysis/data_processing.py).

3. **Baseline modelling**
   - Train and compare Decision Tree, Random Forest, and XGBoost classifiers.
   - Review accuracy and the confusion matrix while considering the imbalance in the target classes.
   - The current experiments are in [notebooks/model_training.ipynb](notebooks/model_training.ipynb).

## What Has Been Achieved

- Added the initial project datasets and a SQLite database artifact.
- Created a data-ingestion notebook and a reusable ingestion script.
- Merged the available data sources and documented the feature categories.
- Investigated missing-value markers and implemented data-cleaning helpers.
- Implemented statistical feature-screening utilities:
  - chi-square testing for categorical variables,
  - ANOVA for numerical variables,
  - sequential VIF-based multicollinearity filtering.
- Implemented categorical feature encoding, including the current ordinal mapping for `EDUCATION` and one-hot encoding for selected nominal columns.
- Trained and compared initial classification models.
- Recorded the current baseline results in the modelling notebook:
  - Decision Tree: approximately 71% accuracy,
  - Random Forest: approximately 76% accuracy,
  - XGBoost: approximately 77.5% accuracy.
- Identified XGBoost as the current baseline model for further improvement.
- Added a baseline confusion-matrix artifact at [assets/confusion_matrix.png](assets/confusion_matrix.png).

## Work In Progress

- Write up and consolidate the exploratory data analysis, including charts and business interpretation.
- Tune the XGBoost model using methods such as GridSearchCV, RandomizedSearchCV, or Bayesian optimization.
- Evaluate metrics beyond accuracy, especially class-wise precision, recall, F1-score, and other metrics appropriate for an imbalanced target.
- Add feature engineering and scaling experiments where they improve the selected models.
- Create a production-oriented `train.py` entry point for repeatable training and evaluation.
- Resolve the ingestion path and compressed-file handling so a fresh checkout can be prepared without manual adjustments.
- Build a Streamlit interface for interactive predictions and model results.
- Package and deploy the application.

## Repository Structure

```text
.
├── assets/                 # Saved modelling artifacts and visualizations
├── data/                   # Compressed source data and SQLite database
├── notebooks/              # Ingestion, analysis, and modelling experiments
├── scripts/
│   └── data_analysis/      # Reusable ingestion and preprocessing helpers
├── domain_knowledge.md     # Credit-risk and banking terminology
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10 or newer
- Jupyter Notebook or JupyterLab
- pandas, NumPy, SciPy, statsmodels, SQLAlchemy, scikit-learn, and XGBoost
- `zstd` if the compressed source files need to be decompressed locally

Install the core Python dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy statsmodels sqlalchemy scikit-learn xgboost jupyter openpyxl
```

Decompress the source files into the `data/` directory, then run the notebooks in this order:

1. `notebooks/data_ingestion.ipynb`
2. `notebooks/dataset_analysis.ipynb`
3. `notebooks/model_training.ipynb`

The current scripts use relative paths, so run them from the directory expected by each script or update the paths as part of the ingestion refactor.

## Current Status

**Prototype complete through initial model training.** The next major milestone is a reproducible, tuned training pipeline followed by a Streamlit interface and deployment.

## NOTE

This repository is an analytical and educational prototype. Its predictions and modelling decisions should not be used for real lending decisions without appropriate validation, governance, monitoring, fairness assessment, security review, and regulatory approval.
