import pandas as pd
import numpy as np


def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def dedup(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicate rows. Returns cleaned df and count removed."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, before - len(df)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for modeling.

    account_status is dropped: it encodes the target directly ("closed" iff
    churned=1), which would be perfect label leakage.
    customer_id is dropped: opaque identifier with no predictive content.
    signup_date is converted to days_since_earliest (numeric temporal feature).
    """
    df = df.copy()
    df = df.drop(columns=["customer_id", "account_status"])
    dates = pd.to_datetime(df["signup_date"])
    df["days_since_start"] = (dates - dates.min()).dt.days
    df = df.drop(columns=["signup_date"])
    return df


def time_split(df: pd.DataFrame, train_frac: float = 0.8):
    """
    Temporal split: customers with earlier signup dates form the training set.
    Simulates real deployment where we train on historical customers and
    predict for newly acquired ones.
    """
    df_sorted = df.sort_values("days_since_start").reset_index(drop=True)
    cutoff = int(len(df_sorted) * train_frac)
    return df_sorted.iloc[:cutoff].copy(), df_sorted.iloc[cutoff:].copy()
