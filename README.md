# Fraud Detection System

A Streamlit dashboard for detecting fraudulent credit card transactions. Upload a
transaction dataset and get live KPIs, class-balancing with SMOTE, model training
and comparison, and an anomaly-detection view — all rendered on the fly, no
pre-trained model required to get started.

## Features

- **CSV/Excel upload** — or falls back to the bundled sample dataset
  (`fraud_detection_dataset_500.csv`)
- **Auto target detection** — recognizes common label columns (`Class`, `Fraud`,
  `label`, `target`)
- **KPI cards** — total transactions, fraud count, legitimate count, fraud rate
- **Class balancing** — SMOTE oversampling, with before/after distribution charts
- **Model comparison** — Logistic Regression, Decision Tree, and Neural Network
  (MLP), each with accuracy scores and confusion matrices
- **ROC curve & AUC score** for the Logistic Regression model
- **Anomaly detection** — Isolation Forest flags suspicious transactions
  independent of the labeled target
- **Excel export** — download per-transaction fraud predictions and probabilities

## Tech stack

Python, Streamlit, Pandas, NumPy, Scikit-learn, imbalanced-learn (SMOTE),
Matplotlib, Seaborn

## Getting started

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser. Upload your own CSV/Excel file, or leave it
empty to explore the bundled sample dataset.

## Dataset

`fraud_detection_dataset_500.csv` — 500 sample transactions with a binary
fraud label, used as the default dataset when no file is uploaded.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard — the entire pipeline runs on each upload |
| `fraud_detection_system.ipynb` | Exploratory notebook version of the analysis |
| `fraud_model.pkl` / `scaler.pkl` | Saved model/scaler artifacts from the notebook |
| `fraud_detection_dataset_500.csv` | Sample/default dataset |
