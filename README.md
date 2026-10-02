# 📡 Telecom Customer Churn Analytics (Python + Streamlit)

An end-to-end data science project: wrangling → EDA → ML model comparison → interactive dashboard.

**Problem:** Telecom companies lose revenue when customers cancel ("churn"). This project finds *why*
customers leave and *predicts who is likely to leave next*, so retention offers can be targeted.

## Dataset
`data/telecom_churn_raw.csv` – **5,035 raw records, 23 columns** (5,000 after de-duplication).
Synthetic data modelled on the public IBM/Kaggle *Telco Customer Churn* dataset (demographics, services,
contract, billing, support calls, data usage, churn label). Messiness is intentionally included:
missing values, duplicates, inconsistent text, wrong data types and outliers.
Regenerate it any time with `python generate_data.py`.
To use your own data (e.g. the real Telco CSV), replace the file and keep the column names.

## Quick start
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501. The first launch trains the models (~20 s), then results are cached.

## Project structure
```
churn_project/
├── app.py                  # Streamlit dashboard (5 pages)
├── generate_data.py        # Creates the dataset
├── data/telecom_churn_raw.csv
├── src/
│   ├── preprocessing.py    # Cleaning, imputation, feature engineering, quality report
│   └── models.py           # Pipelines, training, evaluation, comparison
├── requirements.txt
└── .streamlit/config.toml
```

## What each dashboard page shows
| Page | Content |
|---|---|
| 🏠 Dashboard | KPIs, churn split, churn by contract, plain-English insights, best model |
| 🧹 Data & Wrangling | Raw data, problems found, missing-value before/after chart, cleaning log, clean data download |
| 🔍 Exploratory Analysis | Filterable histograms, churn-driver bars, heatmap, correlation matrix, box/scatter plots |
| 🤖 Model Comparison | Metrics table, grouped bars, ROC curves, 5-fold CV, confusion matrix, feature importance |
| 🎯 Predict Churn | Form → probability gauge, risk level, suggested retention actions, all-model comparison |

## Methods
- **Preprocessing:** duplicate removal, text standardisation, type conversion, outlier handling,
  group-wise median/mode imputation, feature engineering (`NumAddOns`, `ChargePerTenureMonth`, `TenureGroup`).
- **Encoding/scaling:** `StandardScaler` for numeric, `OneHotEncoder` for categorical (in a scikit-learn `Pipeline`, so no data leakage).
- **Models compared:** Logistic Regression, Random Forest, Gradient Boosting.
- **Evaluation:** stratified 80/20 hold-out + 5-fold CV; Accuracy, Precision, Recall, F1, ROC-AUC.

## Typical results (hold-out set)
| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.704 | 0.765 | 0.596 | 0.817 |
| Random Forest | 0.746 | 0.674 | 0.602 | 0.814 |
| Gradient Boosting | 0.749 | 0.677 | 0.606 | 0.810 |
