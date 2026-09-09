# ML_Project 

# Telco Customer Churn E2E MLOps Pipeline

This repository implements a production-grade, end-to-end MLOps architecture for customer churn prediction using the IBM Telco Churn dataset. The system integrates local development workflows in VS Code with cloud-based experiment tracking via Databricks Community Edition, CI/CD automation via GitHub Actions, and production data drift monitoring using Evidently AI.

## Architecture Overview

The system architecture bridges isolated local training scripts with cloud monitoring frameworks across six key stages:

* **Data Ingestion:** Authenticated file transfer streaming the raw data files out of Databricks Unity Catalog Volumes down to local memory via the Databricks Python SDK.
* **Feature Engineering:** Data preprocessing modules automated to resolve data formatting quirks, convert unstructured blank objects, assign label matrix arrays, and encode text vectors into one-hot binary structures.
* **Experimentation and Optimization:** Automated model selection hyperparameter sweeps comparing XGBoost and Random Forest frameworks using Optuna, logging metrics, parameters, and loss attributes directly to Databricks MLflow tracking instances.
* **Model Packaging:** Packaging raw algorithm configurations into an integrated Custom MLflow PyFunc wrapper class ensuring model endpoints ingest raw telemetry features without external transformations.
* **CI/CD Automation:** GitHub Actions workflows executing programmatic verification unit tests via Pytest, and automatically executing the core training file routines on every system codebase push.
* **Production Scoring and Drift Monitoring:** Batch inference modules processing incoming datasets, executing statistical drift validation checks with Evidently AI, and triggering automated REST API dispatches to launch remote cloud retraining loops if data drift limits are breached.

## Directory Structure

```text
telco_mlops_project/
├── .github/
│   └── workflows/
│       └── mlops_pipeline.yml
├── config/
│   └── config.yaml
├── src/
│   ├── __init__.py
│   ├── custom_model.py
│   ├── data_ingestion.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   └── train.py
├── tests/
│   ├── __init__.py
│   └── test_features.py
├── batch_scoring_monitoring.py
├── main.py
├── requirements.txt

```

## Environment Prerequisites

Install all required baseline software libraries via pip using the terminal execution commands:

```bash
pip install -r requirements.txt
```

Create a secret file named `.env` in the root workspace folder to secure account credentials:

```ini
DATABRICKS_HOST="https://databricks.com"
DATABRICKS_TOKEN="your_databricks_personal_access_token"
DATABRICKS_USER_EMAIL="your_login_email@example.com"
GITHUB_TOKEN="your_github_classic_access_token"
```

## Core Pipeline Execution

To trigger the data ingestion, preprocessing, multi-model optimization search, and artifact logging processes locally, execute the primary workspace entrypoint:

```bash
python main.py
```

## Batch Scoring and Drift Automation

To run batch inferences on new incoming telemetry datasets and compute feature drift report summaries, execute the scoring script:

```bash
python batch_scoring_monitoring.py
```
