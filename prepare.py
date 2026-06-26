"""
prepare.py
-----------
Cleans the raw Telco Customer Churn dataset and engineers predictive features.

Reads  data/telco.csv  ->  writes  data/processed.csv
"""

import pandas as pd
import numpy as np

RAW_PATH = "data/telco.csv"
OUT_PATH = "data/processed.csv"
TARGET = "Churn"


def main():
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded raw data: {df.shape[0]} customers, {df.shape[1]} columns")

    df = df.drop(columns=["customerID"])

    # TotalCharges is stored as text; 11 new customers (tenure=0) have blank values
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_blank = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    print(f"Fixed {n_blank} blank TotalCharges values (new customers, tenure 0)")

    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    churn_rate = df[TARGET].mean()
    print(f"Churn rate: {churn_rate:.1%}")

    # Feature engineering
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["NumServices"] = (df[service_cols] == "Yes").sum(axis=1)

    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 1000],
        labels=["0-1yr", "1-2yr", "2-4yr", "4yr+"],
    )

    # Captures price sensitivity relative to usage history
    df["AvgChargesPerMonth"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Explicit flag outperforms letting the model infer it from three dummy columns
    df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)

    # High bill + short tenure = not yet invested in the service
    monthly_median = df["MonthlyCharges"].median()
    df["HighValueNewCustomer"] = (
        (df["tenure"] < 12) & (df["MonthlyCharges"] > monthly_median)
    ).astype(int)

    df = pd.get_dummies(df, drop_first=True)
    # Recent pandas returns bool dtype from get_dummies; cast to int for model compatibility
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved clean data: {df.shape[0]} rows, {df.shape[1]} columns -> {OUT_PATH}")


if __name__ == "__main__":
    main()
