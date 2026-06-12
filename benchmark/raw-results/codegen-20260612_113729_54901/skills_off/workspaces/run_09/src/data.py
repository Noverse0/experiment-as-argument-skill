"""Data loading, cleaning, and split logic.

Rigor notes:
- account_status is a perfect target leak (closed iff churned=1) — dropped.
- 200 exact duplicate rows are appended in make_dataset.py — deduplicated before splitting.
- signup_date is temporal — rows are sorted by date and TimeSeriesSplit is used so
  training always precedes test chronologically.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


FEATURE_COLS = ["tenure_months", "monthly_spend", "support_tickets", "signup_days"]
TARGET_COL = "churned"

# Columns that must be dropped before any modelling
LEAK_COLS = ["account_status"]  # encodes target perfectly
META_COLS = ["customer_id", "signup_date"]


def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV, remove duplicates, drop leaky/meta columns, engineer features."""
    df = pd.read_csv(path, parse_dates=["signup_date"])

    n_before = len(df)
    df = df.drop_duplicates()
    n_dropped = n_before - len(df)

    # Convert temporal column to numeric (days since dataset start)
    reference_date = df["signup_date"].min()
    df["signup_days"] = (df["signup_date"] - reference_date).dt.days

    # Sort chronologically for time-based splitting
    df = df.sort_values("signup_date").reset_index(drop=True)

    # Drop leak and meta columns
    df = df.drop(columns=LEAK_COLS + META_COLS)

    return df, n_dropped


def get_features_target(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    return X, y


def get_time_splits(df: pd.DataFrame, n_splits: int = 5):
    """Return TimeSeriesSplit indices (train, test) over the sorted dataframe."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X, y = get_features_target(df)
    return list(tscv.split(X, y)), X, y
