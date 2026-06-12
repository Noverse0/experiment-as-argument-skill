"""Data loading and cleaning for the churn experiment.

Three rigor traps to neutralize:
  1. account_status is derived from the target — drop it (leakage).
  2. 200 exact duplicate rows were appended — deduplicate before any split.
  3. signup_date is temporal — sort by it so temporal CV is valid.
"""
import pandas as pd
import numpy as np

LEAKED_COLS = ["account_status"]
ID_COLS = ["customer_id"]
DATE_COL = "signup_date"
TARGET = "churned"
FEATURE_COLS = ["tenure_months", "monthly_spend", "support_tickets", "days_since_first"]


def load_clean(path: str) -> tuple:
    df = pd.read_csv(path)
    stats: dict = {"n_raw": len(df)}

    # 1. Deduplicate before any split — prevents duplicate rows straddling train/test.
    df = df.drop_duplicates().reset_index(drop=True)
    stats["n_dupes_dropped"] = stats["n_raw"] - len(df)
    stats["n_clean"] = len(df)

    # 2. Drop target-leaking and meaningless ID columns.
    df = df.drop(columns=LEAKED_COLS + ID_COLS)

    # 3. Convert signup_date to a numeric feature and sort for temporal ordering.
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    min_date = df[DATE_COL].min()
    df["days_since_first"] = (df[DATE_COL] - min_date).dt.days
    df = df.drop(columns=[DATE_COL])

    df = df.sort_values("days_since_first").reset_index(drop=True)

    stats["churn_rate"] = float(df[TARGET].mean())
    stats["n_features"] = len(FEATURE_COLS)

    return df, stats


def get_X_y(df: pd.DataFrame):
    return df[FEATURE_COLS], df[TARGET]
