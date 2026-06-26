"""
app.py
-------
Step 3: the live app. Enter a customer's details, get their churn PROBABILITY,
choose a decision threshold (the business tradeoff), and see what's driving it.

Run it with:   streamlit run app.py
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap

bundle = joblib.load("models/model.joblib")
model = bundle["model"]
FEATURES = bundle["features"]
background = bundle["background"]
X_test, y_test = bundle["X_test"], bundle["y_test"]
with open("models/metrics.json") as f:
    metrics = json.load(f)

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")
st.title("📉 Customer Churn Predictor")
best = metrics["best_model"]
auc = metrics["results"][best]["roc_auc"]
st.caption(f"Model: {best}  •  ROC-AUC {auc:.3f}  •  trained on the Telco dataset (26.5% churn)")

# ---- Customer inputs --------------------------------------------------------
st.sidebar.header("Customer details")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly = st.sidebar.slider("Monthly charges ($)", 18, 120, 70)
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
payment = st.sidebar.selectbox("Payment method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
techsupport = st.sidebar.selectbox("Tech support", ["No", "Yes", "No internet service"])
online_sec = st.sidebar.selectbox("Online security", ["No", "Yes", "No internet service"])
senior = st.sidebar.checkbox("Senior citizen")
paperless = st.sidebar.checkbox("Paperless billing", value=True)

# ---- The business lever: decision threshold ---------------------------------
st.sidebar.header("Decision threshold")
threshold = st.sidebar.slider(
    "Flag as churn risk if probability ≥", 0.1, 0.9, 0.5, 0.05,
    help="Lower = catch more churners but more false alarms. This is a business choice.")


def build_row():
    """Build one input row matching the trained feature columns."""
    row = {f: 0 for f in FEATURES}
    # numeric
    if "tenure" in row: row["tenure"] = tenure
    if "MonthlyCharges" in row: row["MonthlyCharges"] = monthly
    if "TotalCharges" in row: row["TotalCharges"] = monthly * tenure
    if "SeniorCitizen" in row: row["SeniorCitizen"] = int(senior)
    if "NumServices" in row:
        row["NumServices"] = int(techsupport == "Yes") + int(online_sec == "Yes")
    # one-hot style flags (drop_first means some categories are the baseline = all zeros)
    for col, val in {
        f"Contract_{contract}": 1,
        f"InternetService_{internet}": 1,
        f"PaymentMethod_{payment}": 1,
        f"TechSupport_{techsupport}": 1,
        f"OnlineSecurity_{online_sec}": 1,
        f"PaperlessBilling_Yes": int(paperless),
        f"TenureGroup_{'0-1yr' if tenure<=12 else '1-2yr' if tenure<=24 else '2-4yr' if tenure<=48 else '4yr+'}": 1,
    }.items():
        if col in row:
            row[col] = val
    return pd.DataFrame([row])[FEATURES]


if st.sidebar.button("Predict", type="primary"):
    X_one = build_row()
    prob = model.predict_proba(X_one)[0, 1]
    flagged = prob >= threshold

    c1, c2 = st.columns(2)
    c1.metric("Churn probability", f"{prob:.0%}")
    c2.metric("Decision", "⚠️ Churn risk" if flagged else "✅ Likely to stay")

    st.progress(float(prob))

    # ---- Explain with SHAP ---------------------------------------------------
    st.subheader("What's driving this")
    try:
        explainer = shap.Explainer(model.predict_proba, background)
        sv = explainer(X_one)
        vals = sv.values[0, :, 1] if sv.values.ndim == 3 else sv.values[0]
        impact = (pd.DataFrame({"feature": FEATURES, "impact": vals})
                  .assign(a=lambda d: d["impact"].abs())
                  .sort_values("a", ascending=False).head(7))
        for _, r in impact.iterrows():
            arrow = "▲ raises churn risk" if r["impact"] > 0 else "▼ lowers churn risk"
            st.write(f"**{r['feature']}** — {arrow}")
    except Exception as e:
        st.info(f"(Explanation unavailable: {e})")
else:
    st.info("Set the customer's details in the sidebar, then click **Predict**.")

# ---- Show how the threshold affects the whole test set ----------------------
with st.expander("How the threshold changes results across all test customers"):
    proba_all = model.predict_proba(X_test)[:, 1]
    preds_all = (proba_all >= threshold).astype(int)
    tp = int(((preds_all == 1) & (y_test == 1)).sum())
    fp = int(((preds_all == 1) & (y_test == 0)).sum())
    fn = int(((preds_all == 0) & (y_test == 1)).sum())
    caught = tp / (tp + fn) if (tp + fn) else 0
    st.write(f"At threshold **{threshold:.2f}**, the model catches **{caught:.0%}** of real churners "
             f"({tp} of {tp+fn}) and raises **{fp}** false alarms.")
    st.caption("Lower the threshold to catch more churners at the cost of more false alarms — "
               "the right setting depends on how expensive a lost customer is vs. a wasted retention offer.")
