"""
train.py
---------
Step 2: train classifiers with SMOTE, threshold calibration, hyperparameter
tuning, and output ROC / calibration / SHAP plots for the Streamlit app.

Run it with:   python train.py   (after prepare.py)
Reads  data/processed.csv  ->  writes  models/  (model, metrics, plots)

Set SEARCH_ITER=3 for a fast CI run; default is 20.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, average_precision_score,
    precision_recall_curve, roc_curve,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import shap
import joblib

DATA_PATH      = "data/processed.csv"
CUSTOMER_LTV   = 500   # revenue lost when a customer churns ($)
RETENTION_COST = 50    # cost of one proactive retention offer ($)
N_ITER         = int(os.getenv("SEARCH_ITER", "20"))
os.makedirs("models", exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def optimal_threshold(y_val, proba_val):
    """Threshold that maximises F2 score (recall-weighted) on a validation set."""
    p, r, t = precision_recall_curve(y_val, proba_val)
    with np.errstate(invalid="ignore"):
        f2 = np.where((4 * p + r) > 0, 5 * p * r / (4 * p + r), 0)
    return float(t[np.argmax(f2[:-1])])


def evaluate(name, model, X_test, y_test, threshold):
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    m = {
        "roc_auc":   roc_auc_score(y_test, proba),
        "pr_auc":    average_precision_score(y_test, proba),
        "accuracy":  accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall":    recall_score(y_test, preds),
        "f1":        f1_score(y_test, preds),
    }
    cm = confusion_matrix(y_test, preds)
    print(f"\n{'─' * 58}")
    print(f"  {name}  (threshold={threshold:.2f})")
    print(f"{'─' * 58}")
    print(f"  ROC-AUC {m['roc_auc']:.3f}  |  PR-AUC   {m['pr_auc']:.3f}")
    print(f"  Acc     {m['accuracy']:.3f}  |  Precision {m['precision']:.3f}  "
          f"|  Recall {m['recall']:.3f}  |  F1 {m['f1']:.3f}")
    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    print(f"                  pred stay   pred churn")
    print(f"  actual stay     {cm[0, 0]:>9}   {cm[0, 1]:>10}")
    print(f"  actual churn    {cm[1, 0]:>9}   {cm[1, 1]:>10}")
    return m, cm.tolist()


def print_business_impact(cm):
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    saved  = tp * CUSTOMER_LTV
    lost   = fn * CUSTOMER_LTV
    wasted = fp * RETENTION_COST
    net    = saved - lost - wasted
    print(f"\n── Business impact (LTV=${CUSTOMER_LTV}, retention offer=${RETENTION_COST}) ──")
    print(f"  Churners caught (TP={tp}):  ${saved:>8,}  revenue saved")
    print(f"  Churners missed (FN={fn}):  -${lost:>7,}  lost forever")
    print(f"  False alarms    (FP={fp}): -${wasted:>7,}  wasted offers")
    print(f"  {'─'*40}")
    print(f"  Net campaign value:          ${net:>8,}")
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "saved": saved, "lost": lost, "wasted": wasted, "net": net}


def save_plots(model, X_test, y_test, feature_names):
    proba = model.predict_proba(X_test)[:, 1]

    # ── ROC curve ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, color="#2563eb", label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("models/roc_curve.png", dpi=150)
    plt.close(fig)
    print("Saved models/roc_curve.png")

    # ── Calibration curve ─────────────────────────────────────────────────────
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, frac_pos, "s-", color="#16a34a", label="Model")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig("models/calibration_curve.png", dpi=150)
    plt.close(fig)
    print("Saved models/calibration_curve.png")

    # ── SHAP summary ──────────────────────────────────────────────────────────
    print("Computing SHAP values …")
    try:
        sample = X_test.sample(min(200, len(X_test)), random_state=42)
        fn = list(feature_names)

        # Unwrap ImbPipeline: apply preprocessing steps (skip SMOTE), extract clf
        if hasattr(model, "named_steps"):
            clf = model.named_steps["clf"]
            X_proc = sample.copy()
            for step_name, step in model.named_steps.items():
                if step_name == "smote":
                    continue
                if step_name == "clf":
                    break
                if hasattr(step, "transform"):
                    X_proc = step.transform(X_proc)
        else:
            clf = model
            X_proc = sample.values

        if isinstance(clf, LogisticRegression):
            explainer = shap.LinearExplainer(clf, X_proc)
            sv = explainer.shap_values(X_proc)
        elif isinstance(clf, xgb.XGBClassifier):
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(X_proc)
        else:  # RF, HistGBM
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(X_proc)
            if isinstance(sv, list):
                sv = sv[1]

        shap.summary_plot(sv, X_proc, feature_names=fn, show=False, plot_size=(9, 7))
        plt.title("SHAP — what drives each churn prediction")
        plt.tight_layout()
        plt.savefig("models/shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved models/shap_summary.png")
    except Exception as e:
        print(f"SHAP plot skipped: {e}")


# ── Tuning setup ──────────────────────────────────────────────────────────────

def build_tune_estimator(name, spw):
    if name == "Logistic Regression":
        est = ImbPipeline([
            ("smote",  SMOTE(random_state=42)),
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
        grid = {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100]}

    elif name == "Random Forest":
        est = RandomForestClassifier(
            random_state=42, class_weight="balanced", n_jobs=1
        )
        grid = {
            "n_estimators":     [200, 400, 600],
            "max_depth":        [None, 8, 15],
            "min_samples_leaf": [1, 2, 4],
            "max_features":     ["sqrt", 0.3],
        }

    elif name == "HistGradient Boosting":
        est = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("clf",   HistGradientBoostingClassifier(random_state=42)),
        ])
        grid = {
            "clf__learning_rate":     [0.02, 0.05, 0.1, 0.15],
            "clf__max_depth":         [3, 5, 7, None],
            "clf__min_samples_leaf":  [10, 20, 30],
            "clf__l2_regularization": [0, 0.1, 1.0],
        }

    else:  # XGBoost
        est = xgb.XGBClassifier(
            random_state=42, eval_metric="logloss",
            scale_pos_weight=spw, verbosity=0, n_jobs=1,
        )
        grid = {
            "n_estimators":     [200, 400],
            "max_depth":        [3, 5, 7],
            "learning_rate":    [0.02, 0.05, 0.1],
            "subsample":        [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "min_child_weight": [1, 3, 5],
        }

    return est, grid


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(DATA_PATH)
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    # Three-way split: train 60% | val 20% (threshold calibration) | test 20% (final report)
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=42, stratify=y_tv
    )
    print(f"Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")
    print(f"Churn — train: {y_train.mean():.1%}  val: {y_val.mean():.1%}  "
          f"test: {y_test.mean():.1%}\n")

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / pos  # XGBoost scale_pos_weight

    models = {
        "Logistic Regression": ImbPipeline([
            ("smote",  SMOTE(random_state=42)),
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "HistGradient Boosting": ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("clf",   HistGradientBoostingClassifier(random_state=42, max_iter=300)),
        ]),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, random_state=42, eval_metric="logloss",
            scale_pos_weight=spw, n_jobs=-1, verbosity=0,
        ),
    }

    print("── Base models ─────────────────────────────────────────────")
    results, fitted, thresholds = {}, {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        val_proba = model.predict_proba(X_val)[:, 1]
        thresh = optimal_threshold(y_val, val_proba)
        thresholds[name] = thresh
        results[name], _ = evaluate(name, model, X_test, y_test, threshold=thresh)

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    print(f"\n>>> Best base model: {best_name}  "
          f"(ROC-AUC {results[best_name]['roc_auc']:.3f}  "
          f"PR-AUC {results[best_name]['pr_auc']:.3f})\n")

    print(f"── Tuning {best_name} (n_iter={N_ITER}, cv=5) ──────────────────")
    tune_est, param_grid = build_tune_estimator(best_name, spw)
    search = RandomizedSearchCV(
        tune_est,
        param_distributions=param_grid,
        n_iter=N_ITER,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"Best CV ROC-AUC: {search.best_score_:.3f}")
    print(f"Best params:     {search.best_params_}")

    tuned = search.best_estimator_
    val_proba_tuned = tuned.predict_proba(X_val)[:, 1]
    tuned_thresh = optimal_threshold(y_val, val_proba_tuned)

    print(f"\n── Tuned {best_name} ────────────────────────────────────────")
    tuned_metrics, tuned_cm = evaluate(
        f"{best_name} (tuned)", tuned, X_test, y_test, threshold=tuned_thresh
    )
    impact = print_business_impact(tuned_cm)

    save_plots(tuned, X_test, y_test, list(X.columns))

    joblib.dump(
        {
            "model":      tuned,
            "features":   list(X.columns),
            "threshold":  tuned_thresh,
            "background": X_train.sample(min(100, len(X_train)), random_state=42),
            "X_test":     X_test,
            "y_test":     y_test.values,
        },
        "models/model.joblib",
    )
    with open("models/metrics.json", "w") as f:
        json.dump(
            {
                "best_model":             best_name,
                "threshold":              tuned_thresh,
                "results":                results,
                "tuned_metrics":          tuned_metrics,
                "tuned_confusion_matrix": tuned_cm,
                "business_impact":        impact,
            },
            f, indent=2,
        )
    print("\nSaved models/model.joblib and models/metrics.json")


if __name__ == "__main__":
    main()
