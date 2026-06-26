"""
prepare.py
-----------
Step 1: clean the raw Telco data and turn it into a numeric table a model can learn from.

Run it with:   python prepare.py
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

    # ---- 1. Drop the ID column (it's just a label, not a predictor) -----------
    df = df.drop(columns=["customerID"])

    # ---- 2. Fix the TotalCharges quirk ----------------------------------------
    # TotalCharges loaded as text because 11 rows are blank — these are brand-new
    # customers (tenure = 0) who haven't been billed yet. Convert to a number;
    # the blanks become NaN, then fill with 0 (they've paid nothing so far).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_blank = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    print(f"Fixed {n_blank} blank TotalCharges values (new customers, tenure 0)")

    # ---- 3. Turn the target into 0/1 ------------------------------------------
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    churn_rate = df[TARGET].mean()
    print(f"Churn rate: {churn_rate:.1%}  (this is imbalanced — accuracy will mislead)")

    # ---- 4. Feature engineering -----------------------------------------------
    # Count how many add-on services a customer has — more services often means
    # more "locked in" and less likely to leave.
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["NumServices"] = (df[service_cols] == "Yes").sum(axis=1)

    # New customers churn far more than long-tenured ones — bucket tenure.
    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 1000],
        labels=["0-1yr", "1-2yr", "2-4yr", "4yr+"],
    )

    # Customers overpaying relative to their tenure have higher price sensitivity
    df["AvgChargesPerMonth"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Month-to-month customers churn at ~3x the rate of annual contracts; explicit
    # flag beats relying on the model to infer it from three dummy columns
    df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)

    # Short-tenure + high monthly bill = price-shocked, not yet invested in service
    monthly_median = df["MonthlyCharges"].median()
    df["HighValueNewCustomer"] = (
        (df["tenure"] < 12) & (df["MonthlyCharges"] > monthly_median)
    ).astype(int)

    # ---- 5. Encode all text columns into numbers ------------------------------
    df = pd.get_dummies(df, drop_first=True)
    # get_dummies makes boolean columns in recent pandas; cast to 0/1 ints so the
    # whole table is purely numeric (cleaner, and required by some ML tooling).
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved clean data: {df.shape[0]} rows, {df.shape[1]} columns -> {OUT_PATH}")


if __name__ == "__main__":
    main()
