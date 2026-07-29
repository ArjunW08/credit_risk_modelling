# Credit Risk Modelling

A credit-risk modelling prototype that predicts loan approval outcomes (P1–P4 risk bands) using customer credit bureau and internal product data. The project covers end-to-end data ingestion, feature engineering, model training, and a Streamlit inference application — all containerised with Docker.

## Project Goals

- Join internal product data with CIBIL bureau data via `PROSPECTID`.
- Clean, filter, and encode numerical and categorical features.
- Train and compare baseline classifiers for the `Approved_Flag` target.
- Expose a decision-support interface using a trained XGBoost model.

## Data

The source datasets are stored as compressed `.zst` archives :

- `data/internal_product.xlsx.zst`
- `data/cibil_score.xlsx.zst`
- `data/feature_target_description.xlsx.zst`

A pre-built SQLite database is available at `data/credit_modelling.db`.

The modelling target is `Approved_Flag`. Credit-risk terminology is documented in [domain_knowledge.md](domain_knowledge.md). The full exploratory analysis is documented in [eda.md](eda.md).

## Workflow

1. **Data ingestion** — Load source Excel datasets and persist to SQLite via [scripts/data/ingestion_db.py](scripts/data/ingestion_db.py).
2. **Data analysis and preparation** — Merge internal and CIBIL tables, handle `-99999` sentinel values, apply chi-square / VIF / ANOVA feature screening, and encode categoricals via [scripts/data/data_processing.py](scripts/data/data_processing.py).
3. **Model training** — Train XGBoost with RandomizedSearchCV and evaluate via [scripts/model_creation/train.py](scripts/model_creation/train.py) and [scripts/model_creation/model_evaluation.py](scripts/model_creation/model_evaluation.py).
4. **Inference** — Generate predictions with the trained model at `models/predict_loan_possibility_model.pkl` using the Streamlit UI in [app.py](app.py) or the CLI script at [scripts/inference/predict_loan_possibility.py](scripts/inference/predict_loan_possibility.py).

## Repository Structure

```text
.
├── app.py                  # Streamlit inference app
├── Dockerfile              # Docker build (Miniconda + uv)
├── docker-compose.yml      # Compose services (app + train)
├── .dockerignore           # Docker build context exclusions
├── pyproject.toml          # Project metadata and dependencies (uv)
├── data/                   # Compressed archives (.zst), SQLite DB
├── models/                 # Trained model artifacts (.pkl)
├── notebooks/              # Exploratory and modelling notebooks
├── scripts/
│   ├── entrypoint_train.py # Docker training pipeline entrypoint
│   ├── data/               # Ingestion and preprocessing helpers
│   ├── inference/          # Prediction and inference helpers
│   └── model_creation/     # Training and evaluation scripts
├── domain_knowledge.md     # Credit-risk terminology and notes
├── eda.md                  # Exploratory data analysis details and insights
└── README.md
```

## Getting Started

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- [uv](https://docs.astral.sh/uv/) package installer
- Jupyter Notebook or JupyterLab (for notebooks)
- `zstd` if you need to decompress `.zst` source files

### Setup

Create the conda environment and install all dependencies from `pyproject.toml`:

```bash
conda create -n crm python=3.11 -y
conda activate crm
pip install uv
uv pip install .
```

If you only have the compressed archives (fresh clone), decompress them first:

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

### Docker

Requires [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/).

**Build the image:**

```bash
docker compose build
```

**Run the Streamlit inference app:**

```bash
docker compose up app
```

The UI will be available at [http://localhost:8501](http://localhost:8501). The `data/`, `models/`, and `logs/` directories are bind-mounted from the host so the container uses your local data and trained model automatically.

**Run the training pipeline** (ingestion → processing → training):

```bash
docker compose --profile train run train
```

**Stop and remove containers:**

```bash
docker compose down
```

## NOTE

This repository is an analytical and educational prototype. Its predictions and modelling decisions should not be used for real lending decisions without appropriate validation, governance, monitoring, fairness assessment, security review, and regulatory approval.
