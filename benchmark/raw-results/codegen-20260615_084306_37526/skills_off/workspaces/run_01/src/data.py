"""Data loading, preprocessing, and splitting for the churn experiment."""
import pandas as pd
import numpy as np

REFERENCE_DATE = pd.Timestamp("2023-01-01")

# days_since_last_login is a target leak: it is recorded *after* the churn
# outcome because a churned customer has, by definition, stopped logging in.
# The value is causally derived from the label, not a predictor of it.
LEAKY_FEATURES = {"days_since_last_login"}

_DROP = {"customer_id", "signup_date", "churned"} | LEAKY_FEATURES
TARGET = "churned"


def load_and_preprocess(csv_path: str) -> tuple:
    """
    Load churn CSV, remove duplicates, engineer features.

    Returns (X, y, meta) where X is a DataFrame of clean features,
    y is the binary target Series, and meta is a dict of audit info.
    The rows are sorted by signup_date (ascending) to support temporal splits.
    """
    df = pd.read_csv(csv_path, parse_dates=["signup_date"])

    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)

    df = df.sort_values("signup_date").reset_index(drop=True)

    # Encode signup cohort as numeric (days since the dataset start date)
    df["days_since_signup"] = (df["signup_date"] - REFERENCE_DATE).dt.days

    feature_cols = [c for c in df.columns if c not in _DROP]

    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    meta = {
        "n_duplicates_removed": n_removed,
        "n_total": len(df),
        "features": feature_cols,
        "churn_rate": float(y.mean()),
    }

    return X, y, meta


def temporal_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> tuple:
    """
    Split preserving temporal order (assumes rows sorted by signup_date ascending).
    Earlier rows go to train; later rows go to test.
    """
    n = len(X)
    split_idx = int(n * (1 - test_size))
    X_train = X.iloc[:split_idx].reset_index(drop=True)
    X_test = X.iloc[split_idx:].reset_index(drop=True)
    y_train = y.iloc[:split_idx].reset_index(drop=True)
    y_test = y.iloc[split_idx:].reset_index(drop=True)
    return X_train, X_test, y_train, y_test
