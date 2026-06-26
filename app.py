"""
app.py
-------
Step 3: the live Streamlit app. Two tabs:
  • Predict  — enter a customer's details, get churn probability + business impact
  • Model performance — ROC curve, calibration curve, SHAP summary

Run it with:   streamlit run app.py
"""

import json
import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap

bundle = joblib.load("models/model.joblib")
model           = bundle["model"]
FEATURES        = bundle["features"]
background      = bundle["background"]
X_test, y_test  = bundle["X_test"], bundle["y_test"]
SAVED_THRESHOLD = float(bundle.get("threshold", 0.5))

with open("models/metrics.json") as f:
    metrics = json.load(f)

CUSTOMER_LTV   = 500
RETENTION_COST = 50

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")
st.title("📉 Customer Churn Predictor")

best = metrics["best_model"]
if "tuned_metrics" in metrics:
    m = metrics["tuned_metrics"]
    st.caption(
        f"Model: **{best} (tuned)** · ROC-AUC **{m['roc_auc']:.3f}** · "
        f"PR-AUC **{m['pr_auc']:.3f}** · Recall **{m['recall']:.3f}** · "
        f"Telco dataset · 7,043 customers · 26.5% churn"
    )
else:
    auc = metrics["results"][best]["roc_auc"]
    st.caption(f"Model: {best} · ROC-AUC {auc:.3f} · Telco dataset · 26.5% churn")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Customer details")
tenure      = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly     = st.sidebar.slider("Monthly charges ($)", 18, 120, 70)
contract    = st.sidebar.selectbox("Contract",
                ["Month-to-month", "One year", "Two year"])
internet    = st.sidebar.selectbox("Internet service",
                ["Fiber optic", "DSL", "No"])
payment     = st.sidebar.selectbox("Payment method",
                ["Electronic check", "Mailed check",
                 "Bank transfer (automatic)", "Credit card (automatic)"])
techsupport = st.sidebar.selectbox("Tech support",
                ["No", "Yes", "No internet service"])
online_sec  = st.sidebar.selectbox("Online security",
                ["No", "Yes", "No internet service"])
senior      = st.sidebar.checkbox("Senior citizen")
paperless   = st.sidebar.checkbox("Paperless billing", value=True)

st.sidebar.header("Decision threshold")
threshold = st.sidebar.slider(
    "Flag as churn risk if probability ≥",
    0.05, 0.95, SAVED_THRESHOLD, 0.05,
    help="Lower = catch more churners but more false alarms. This is a business decision.",
)


def build_row():
    row = {f: 0 for f in FEATURES}
    if "tenure"               in row: row["tenure"]               = tenure
    if "MonthlyCharges"       in row: row["MonthlyCharges"]       = monthly
    if "TotalCharges"         in row: row["TotalCharges"]         = monthly * tenure
    if "SeniorCitizen"        in row: row["SeniorCitizen"]        = int(senior)
    if "NumServices"          in row:
        row["NumServices"] = int(techsupport == "Yes") + int(online_sec == "Yes")
    if "AvgChargesPerMonth"   in row:
        row["AvgChargesPerMonth"]   = (monthly * tenure) / (tenure + 1)
    if "IsMonthToMonth"       in row:
        row["IsMonthToMonth"]       = int(contract == "Month-to-month")
    if "HighValueNewCustomer" in row:
        row["HighValueNewCustomer"] = int(tenure < 12 and monthly > 64)

    tenure_grp = ("0-1yr" if tenure <= 12 else
                  "1-2yr" if tenure <= 24 else
                  "2-4yr" if tenure <= 48 else "4yr+")
    for col, val in {
        f"Contract_{contract}":         1,
        f"InternetService_{internet}":  1,
        f"PaymentMethod_{payment}":     1,
        f"TechSupport_{techsupport}":   1,
        f"OnlineSecurity_{online_sec}": 1,
        f"PaperlessBilling_Yes":        int(paperless),
        f"TenureGroup_{tenure_grp}":    1,
    }.items():
        if col in row:
            row[col] = val
    return pd.DataFrame([row])[FEATURES]


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_perf = st.tabs(["Predict", "Model performance"])

# ──────────────────────────────────────────────────────────────────────────────
with tab_predict:
    if st.button("Predict", type="primary"):
        X_one = build_row()
        prob   = model.predict_proba(X_one)[0, 1]
        flagged = prob >= threshold

        c1, c2 = st.columns(2)
        c1.metric("Churn probability", f"{prob:.0%}")
        c2.metric("Decision", "⚠️ Churn risk" if flagged else "✅ Likely to stay")
        st.progress(float(prob))

        # SHAP explanation
        st.subheader("What's driving this prediction")
        try:
            explainer = shap.Explainer(model.predict_proba, background)
            sv = explainer(X_one)
            vals = sv.values[0, :, 1] if sv.values.ndim == 3 else sv.values[0]
            impact_df = (
                pd.DataFrame({"feature": FEATURES, "impact": vals})
                .assign(abs_impact=lambda d: d["impact"].abs())
                .sort_values("abs_impact", ascending=False)
                .head(7)
            )
            for _, r in impact_df.iterrows():
                direction = "▲ raises churn risk" if r["impact"] > 0 else "▼ lowers churn risk"
                st.write(f"**{r['feature']}** — {direction}")
        except Exception as e:
            st.info(f"Explanation unavailable: {e}")

        # Business impact for this customer + full test set at this threshold
        st.subheader("Business impact at this threshold")
        proba_all = model.predict_proba(X_test)[:, 1]
        preds_all = (proba_all >= threshold).astype(int)
        tp = int(((preds_all == 1) & (y_test == 1)).sum())
        fp = int(((preds_all == 1) & (y_test == 0)).sum())
        fn = int(((preds_all == 0) & (y_test == 1)).sum())
        total_churners = tp + fn
        caught_pct = tp / total_churners if total_churners else 0
        net = tp * CUSTOMER_LTV - fn * CUSTOMER_LTV - fp * RETENTION_COST

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Churners caught",   f"{tp} / {total_churners}", f"{caught_pct:.0%} recall")
        b2.metric("False alarms",      str(fp),                    "wasted offers")
        b3.metric("Revenue saved",     f"${tp * CUSTOMER_LTV:,}")
        b4.metric("Net campaign value", f"${net:,}")
        st.caption(
            f"Assumes ${CUSTOMER_LTV} lifetime value per saved customer "
            f"and ${RETENTION_COST} per retention offer sent."
        )
    else:
        st.info("Set the customer's details in the sidebar, then click **Predict**.")

    with st.expander("How the threshold changes results across all test customers"):
        proba_all = model.predict_proba(X_test)[:, 1]
        preds_all = (proba_all >= threshold).astype(int)
        tp_ = int(((preds_all == 1) & (y_test == 1)).sum())
        fp_ = int(((preds_all == 1) & (y_test == 0)).sum())
        fn_ = int(((preds_all == 0) & (y_test == 1)).sum())
        caught_ = tp_ / (tp_ + fn_) if (tp_ + fn_) else 0
        st.write(
            f"At threshold **{threshold:.2f}** the model catches **{caught_:.0%}** of real churners "
            f"({tp_} of {tp_+fn_}) and raises **{fp_}** false alarms."
        )
        st.caption(
            "Lower the threshold to catch more churners at the cost of more false alarms. "
            "The right setting depends on how expensive a lost customer is vs. a wasted offer."
        )

# ──────────────────────────────────────────────────────────────────────────────
with tab_perf:
    st.subheader("Evaluation on held-out test set (20% of data, never seen during training)")

    if "tuned_metrics" in metrics:
        m = metrics["tuned_metrics"]
        cols = st.columns(5)
        for col, (label, key) in zip(cols, [
            ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"),
            ("Recall",  "recall"),  ("Precision", "precision"), ("F1", "f1"),
        ]):
            col.metric(label, f"{m[key]:.3f}")
        st.caption(
            "ROC-AUC and PR-AUC don't depend on the threshold — they measure the model's "
            "raw discriminating power. Recall, precision, and F1 reflect the chosen threshold above."
        )

    col_roc, col_cal = st.columns(2)
    for path, caption, col in [
        ("models/roc_curve.png",        "ROC Curve",         col_roc),
        ("models/calibration_curve.png", "Calibration Curve", col_cal),
    ]:
        if os.path.exists(path):
            col.image(path, caption=caption, use_container_width=True)

    st.markdown("---")
    st.subheader("SHAP — what drives churn predictions globally")
    if os.path.exists("models/shap_summary.png"):
        st.image("models/shap_summary.png", use_container_width=True)
        st.caption(
            "Each dot is one test customer. Position on the x-axis shows how much that feature "
            "pushed the prediction toward churn (right) or away (left). "
            "Color = feature value: red = high, blue = low."
        )
    else:
        st.info("Run train.py to generate the SHAP plot.")

    st.markdown("---")
    if "business_impact" in metrics:
        bi = metrics["business_impact"]
        st.subheader("Business impact (at trained threshold on test set)")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Revenue saved",      f"${bi['saved']:,}")
        b2.metric("Revenue lost",       f"-${bi['lost']:,}")
        b3.metric("Wasted offers",      f"-${bi['wasted']:,}")
        b4.metric("Net campaign value", f"${bi['net']:,}")
        st.caption(
            f"TP={bi['tp']} churners caught · FN={bi['fn']} missed · "
            f"FP={bi['fp']} false alarms · "
            f"LTV=${CUSTOMER_LTV} · retention offer=${RETENTION_COST}"
        )
