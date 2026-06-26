# Customer Churn Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red)
![Streamlit](https://img.shields.io/badge/Streamlit-app-ff4b4b?logo=streamlit)
![CI](https://github.com/yinytan0104/churn-predictor/actions/workflows/ci.yml/badge.svg)

An end-to-end ML pipeline that predicts which telecom customers are likely to cancel their service, explains *why* using SHAP, and quantifies the **dollar value** of acting on those predictions.

Built on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 26.5% churn rate.

---

## Results

| Model | ROC-AUC | PR-AUC | Recall | Precision |
|---|---|---|---|---|
| **Logistic Regression** ⭐ | **0.843** | **0.671** | **0.944** | 0.398 |
| Random Forest | 0.826 | 0.616 | 0.912 | 0.419 |
| HistGradient Boosting | 0.818 | 0.600 | 0.925 | 0.393 |
| XGBoost | 0.806 | 0.584 | 0.877 | 0.420 |

All models use F2-optimal threshold calibration on a held-out validation set. Winner chosen by ROC-AUC.

### Business impact (at trained threshold, 1,409-customer test set)

| | Value |
|---|---|
| Churners caught (TP) | 353 / 374 — **94.4% recall** |
| Revenue saved | **$176,500** |
| Churners missed | 21 customers × $500 LTV = $10,500 lost |
| False alarms | 535 × $50 offer = $26,750 wasted |
| **Net campaign value** | **$139,250** |

---

## Why accuracy is a trap here

Only 26.5% of customers churn. A model that predicts *"nobody ever churns"* is 73.5% accurate — and completely useless. This project is evaluated on:

- **ROC-AUC** — how well the model ranks churners above stayers regardless of threshold (~0.84)
- **PR-AUC** — precision-recall tradeoff focused entirely on the minority churn class
- **Recall** — what fraction of real churners are caught (missing one = full LTV lost)
- **Calibration** — are predicted probabilities trustworthy? (a 70% predicted churn ≈ 70% actual churn rate)
- **Decision threshold** — lowering it catches more churners at the cost of more false alarms; the app lets you slide this and see the business impact change in real time

---

## How it works

```
data/telco.csv
      │
      ▼
  prepare.py          — fix TotalCharges, engineer 5 features, one-hot encode
      │
      ▼
data/processed.csv
      │
      ▼
   train.py           — SMOTE, 4 models, F2-threshold calibration, RandomizedSearchCV
      │                  → ROC curve, calibration curve, SHAP summary plots
      ▼
 models/              — model.joblib  metrics.json  *.png
      │
      ▼
   app.py             — Streamlit: predict + explain + business impact dashboard
```

---

## How to run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python prepare.py                # clean data → data/processed.csv
python train.py                  # train & evaluate → models/

streamlit run app.py             # open http://localhost:8501
```

### Deploy to Streamlit Community Cloud (free)

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
2. Select this repo, branch `main`, file `app.py`
3. Click **Deploy** — `model.joblib` is committed so the app loads immediately

---

## Project structure

```
churn-predictor/
├── 01_eda.ipynb          # exploratory analysis: why these features?
├── prepare.py            # data cleaning + feature engineering
├── train.py              # model training, evaluation, plots
├── app.py                # Streamlit dashboard
├── get_data.py           # download raw CSV from Kaggle
├── requirements.txt      # loose dependencies
├── requirements-lock.txt # pinned versions for reproducibility
├── data/
│   ├── telco.csv         # raw dataset (7,043 customers, 21 columns)
│   └── processed.csv     # engineered features (38 columns)
└── models/
    ├── model.joblib       # trained model + metadata
    ├── metrics.json       # ROC-AUC, recall, business impact
    ├── roc_curve.png      # saved by train.py
    ├── calibration_curve.png
    └── shap_summary.png
```

---

## Feature engineering

| Feature | Rationale |
|---|---|
| `NumServices` | More add-ons → more locked-in → less likely to leave |
| `TenureGroup` | 0-1yr customers churn at ~47% vs ~6% for 4yr+ |
| `AvgChargesPerMonth` | Overpaying relative to usage history → price sensitivity |
| `IsMonthToMonth` | Month-to-month churn rate is ~4× annual contracts |
| `HighValueNewCustomer` | Short tenure + high bill = price-shocked, not yet invested |

---

## Resume

> Built an end-to-end churn classification pipeline (ROC-AUC 0.843, 94% recall) on 7,000+ telecom customers; handled class imbalance with SMOTE and F2-optimal threshold calibration; added SHAP explainability, calibration curves, and a live Streamlit app that translates model output into estimated campaign revenue.
