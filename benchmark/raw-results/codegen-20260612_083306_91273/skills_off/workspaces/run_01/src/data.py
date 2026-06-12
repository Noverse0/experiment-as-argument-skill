"""Data loading, cleaning, deduplication, feature engineering, and splitting."""
import pandas as pd
import numpy as np


LEAK_COLS = ["account_status"]
ID_COLS = ["customer_id"]
DATE_COL = "signup_date"
TARGET = "churned"
REFERENCE_DATE = pd.Timestamp("2023-01-01")


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop target-leaking and identifier columns."""
    drop = [c for c in LEAK_COLS + ID_COLS if c in df.columns]
    return df.drop(columns=drop)


def dedup_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows. Must run before splitting."""
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    print(f"[dedup] removed {n_removed} exact duplicates ({n_before} → {len(df)} rows)")
    return df.reset_index(drop=True)


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Convert signup_date to numeric days, drop the original string column."""
    df = df.copy()
    df["signup_day"] = (pd.to_datetime(df[DATE_COL]) - REFERENCE_DATE).dt.days
    df = df.drop(columns=[DATE_COL])
    return df


def time_split(df: pd.DataFrame, test_frac: float = 0.20):
    """Sort by signup_day, split at quantile boundary (respects temporal ordering)."""
    df_sorted = df.sort_values("signup_day").reset_index(drop=True)
    cutoff = int(len(df_sorted) * (1 - test_frac))
    train = df_sorted.iloc[:cutoff]
    test = df_sorted.iloc[cutoff:]

    # Sanity: no duplicate rows across the boundary
    train_idx = set(train.index)
    test_idx = set(test.index)
    assert train_idx.isdisjoint(test_idx), "Train/test index overlap detected"

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET]
    return X_train, X_test, y_train, y_test


def prepare(path: str, test_frac: float = 0.20):
    """Full data preparation pipeline."""
    df = load_data(path)
    df = clean_data(df)
    df = dedup_data(df)
    df = feature_engineer(df)
    return time_split(df, test_frac=test_frac)
