import pandas as pd
import numpy as np

TARGET = "churned"
FEATURE_COLS = ["tenure_months", "monthly_spend", "support_tickets"]
# Excluded columns and reasons:
# - customer_id: row identifier, not predictive
# - signup_date: used only for temporal ordering of the split
# - days_since_last_login: TARGET LEAK — churned customers have stopped logging in
#   by definition; this value is recorded after the churn outcome, not before it.
#   A model using it learns a consequence of churn, not a cause.


def load_and_prepare(csv_path: str):
    """Load CSV, remove exact duplicates, sort chronologically."""
    df = pd.read_csv(csv_path)
    n_before = len(df)
    df = df.drop_duplicates()
    n_duplicates = n_before - len(df)
    df = df.sort_values("signup_date").reset_index(drop=True)
    return df, n_duplicates


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Temporal train/test split: first (1-test_frac) by signup_date → train."""
    n = len(df)
    split_idx = int(n * (1 - test_frac))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def get_Xy(df: pd.DataFrame):
    """Extract feature matrix and target vector from prepared DataFrame."""
    X = df[FEATURE_COLS].values.astype(float)
    y = df[TARGET].values.astype(int)
    return X, y
