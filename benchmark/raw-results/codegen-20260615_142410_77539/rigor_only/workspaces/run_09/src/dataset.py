"""Data loading and preprocessing: dedup, time-based split."""
import pandas as pd
import numpy as np
from typing import Tuple


def load_churn_data(path: str = "churn.csv") -> pd.DataFrame:
    """Load churn dataset."""
    df = pd.read_csv(path)
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    return df


def deduplicate(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Remove exact duplicates and return clean df + count removed."""
    initial_len = len(df)
    df_clean = df.drop_duplicates()
    removed = initial_len - len(df_clean)
    return df_clean, removed


def split_by_time(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split on signup_date (respects temporal order).
    Within each split, shuffle with the given seed to ensure reproducibility.
    """
    df = df.sort_values("signup_date").reset_index(drop=True)
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].sample(frac=1, random_state=seed).reset_index(drop=True)
    test = df.iloc[split_idx:].sample(frac=1, random_state=seed).reset_index(drop=True)
    return train, test


def get_features_and_target(
    df: pd.DataFrame,
    drop_leakage: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract features and target.
    If drop_leakage=True, drop 'days_since_last_login' (target leak).
    Always drop non-predictive columns: customer_id, signup_date.
    """
    X = df.drop(columns=["churned", "customer_id", "signup_date"])

    if drop_leakage:
        X = X.drop(columns=["days_since_last_login"])

    y = df["churned"]
    return X, y
