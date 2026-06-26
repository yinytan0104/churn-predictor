# 📉 Customer Churn Predictor

A classification model that predicts which telecom customers are likely to cancel,
served as an interactive app that shows each customer's churn probability, lets you
tune the decision threshold, and explains the prediction with SHAP.

Built on the **Telco Customer Churn dataset** (7,043 customers, 26.5% churn rate).

**What it demonstrates:** data cleaning, feature engineering, classification, and —
the part that makes it a *data science* project — evaluating an **imbalanced** problem
the right way (ROC-AUC, precision, recall, confusion matrix) instead of relying on
accuracy, plus choosing a decision threshold to match a business goal.

---

## Files

| File | What it does |
|------|--------------|
| `get_data.py` | Downloads the Telco dataset into `data/telco.csv` |
| `prepare.py`  | Cleans the data (fixes the `TotalCharges` quirk), engineers features → `data/processed.csv` |
| `train.py`    | Trains 3 classifiers, evaluates with ROC-AUC + confusion matrix, saves the best |
| `app.py`      | Live app: churn probability, adjustable threshold, SHAP explanation |

---

## How to run it

Needs **Python 3.10+**. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python get_data.py
python prepare.py
python train.py

streamlit run app.py
```

---

## The key idea: why accuracy is a trap here

Only 26.5% of customers churn. A lazy model that predicts *"nobody churns"* would be
73% accurate and completely useless — it never catches the customers you actually want
to save. That's why this project is judged on:

- **ROC-AUC** — how well the model ranks churners above non-churners (~0.84 here)
- **Recall** — what fraction of real churners it catches
- **Precision** — of the customers it flags, how many actually churn
- **The decision threshold** — lowering it catches more churners but raises more false
  alarms. The right setting is a *business* decision: how costly is a lost customer
  versus a wasted retention offer? The app lets you slide this and see the effect.

The models use `class_weight="balanced"` so they pay attention to the rarer churn class
instead of ignoring it.

---

## Make it yours

Open `prepare.py`, find the **YOUR TURN** block, and add 2-3 features you can explain
(e.g. average charges per month, a month-to-month contract flag). Re-run `prepare.py`
and `train.py` and watch ROC-AUC and recall move.

---

## Resume bullet this produces (fill in YOUR numbers)

> Built a customer-churn classification model (ROC-AUC ~0.84) on 7,000+ telecom
> customers, handling class imbalance with balanced class weights and evaluating with
> precision/recall and a confusion matrix; deployed an interactive Streamlit app with
> an adjustable decision threshold and SHAP explanations.

---

## Optional next steps
- Deploy publicly on Streamlit Community Cloud for a shareable link.
- Add a ROC curve and precision-recall curve to the app.
- Try `scale_pos_weight` with XGBoost and compare.
