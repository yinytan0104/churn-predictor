"""
train.py
---------
Step 2: train several classifiers with proper imbalance handling, threshold
calibration, and hyperparameter tuning.

Run it with:   python train.py   (after prepare.py)
Reads  data/processed.csv  ->  writes  models/model.joblib  and  models/metrics.json
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, average_precision_score, precision_recall_curve,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import joblib

DATA_PATH = "data/processed.csv"
os.makedirs("models", exist_ok=True)


def optimal_threshold(y_val, proba_val):
    """Threshold that maximises F2 (weights recall 2× over precision) on a val set."""
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


def build_tune_estimator(name, spw):
    """Return a fresh (unfitted) estimator + param grid for RandomizedSearchCV."""
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
            "n_estimators":    [200, 400, 600],
            "max_depth":       [None, 8, 15],
            "min_samples_leaf": [1, 2, 4],
            "max_features":    ["sqrt", 0.3],
        }

    elif name == "HistGradient Boosting":
        est = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("clf",   HistGradientBoostingClassifier(random_state=42)),
        ])
        grid = {
            "clf__learning_rate":    [0.02, 0.05, 0.1, 0.15],
            "clf__max_depth":        [3, 5, 7, None],
            "clf__min_samples_leaf": [10, 20, 30],
            "clf__l2_regularization": [0, 0.1, 1.0],
        }

    else:  # XGBoost
        est = xgb.XGBClassifier(
            random_state=42, eval_metric="logloss",
            scale_pos_weight=spw, verbosity=0, n_jobs=1,
        )
        grid = {
            "n_estimators":    [200, 400],
            "max_depth":       [3, 5, 7],
            "learning_rate":   [0.02, 0.05, 0.1],
            "subsample":       [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "min_child_weight": [1, 3, 5],
        }

    return est, grid


def main():
    df = pd.read_csv(DATA_PATH)
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    # Three-way split so the test set is never touched until final evaluation.
    # train 60% | val 20% (threshold calibration) | test 20% (final report)
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

    # SMOTE is inside ImbPipeline so it only ever sees training data (never val/test).
    # LR and RF use class_weight="balanced" instead; XGBoost uses scale_pos_weight.
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
        # Calibrate threshold on val set (not seen during training)
        val_proba = model.predict_proba(X_val)[:, 1]
        thresh = optimal_threshold(y_val, val_proba)
        thresholds[name] = thresh
        results[name], _ = evaluate(name, model, X_test, y_test, threshold=thresh)

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    print(f"\n>>> Best base model: {best_name}  "
          f"(ROC-AUC {results[best_name]['roc_auc']:.3f}  "
          f"PR-AUC {results[best_name]['pr_auc']:.3f})\n")

    # Hyperparameter tuning — train only on X_train so X_val stays clean for
    # threshold calibration on the winning tuned model.
    print(f"── Tuning {best_name} (RandomizedSearchCV n_iter=20, cv=5) ─────")
    tune_est, param_grid = build_tune_estimator(best_name, spw)
    search = RandomizedSearchCV(
        tune_est,
        param_distributions=param_grid,
        n_iter=20,
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

    print(f"\n── Tuned {best_name} ───────────────────────────────────────────")
    tuned_metrics, tuned_cm = evaluate(
        f"{best_name} (tuned)", tuned, X_test, y_test, threshold=tuned_thresh
    )

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
                "best_model":           best_name,
                "threshold":            tuned_thresh,
                "base_results":         results,
                "tuned_metrics":        tuned_metrics,
                "tuned_confusion_matrix": tuned_cm,
            },
            f, indent=2,
        )
    print("\nSaved models/model.joblib and models/metrics.json")


if __name__ == "__main__":
    main()
