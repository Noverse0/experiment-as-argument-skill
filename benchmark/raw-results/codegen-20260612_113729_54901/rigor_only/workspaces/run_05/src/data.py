"""Data loading, deduplication, splitting, and preprocessing.

Design decisions that guard rigor:
- Dedup BEFORE split: 200 planted duplicate rows must not straddle train/test.
- Drop account_status: it is derived from the target (closed ↔ churned=1), so
  including it would be a direct label leak.
- Drop customer_id: identifier with no predictive signal.
- Time-based split on signup_date: the dataset spans 2023–2025; a random split
  on temporal data constitutes leakage via future information.
- Fit scaler on train only, apply to val/test: prevents distribution leakage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Features that are kept after dropping leaks and identifiers
NUMERIC_FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]
TARGET = "churned"

# Leaky / identifier columns that must be removed before any ML step
LEAKED_COLS = ["customer_id", "account_status", "signup_date"]


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    return df


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicate rows. Returns cleaned df and count removed."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    return df.reset_index(drop=True), removed


def time_split(
    df: pd.DataFrame,
    train_frac: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically by signup_date, no random shuffling."""
    df_sorted = df.sort_values("signup_date").reset_index(drop=True)
    cutoff = int(len(df_sorted) * train_frac)
    train = df_sorted.iloc[:cutoff].copy()
    test = df_sorted.iloc[cutoff:].copy()
    return train, test


def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Drop leaks, scale numerics (fit on train only)."""
    X_train_raw = train[NUMERIC_FEATURES].values.astype(float)
    X_test_raw = test[NUMERIC_FEATURES].values.astype(float)
    y_train = train[TARGET].values
    y_test = test[TARGET].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, X_test, y_train, y_test, scaler


def prepare(
    path: str | Path,
    train_frac: float = 0.80,
) -> dict:
    """Full pipeline: load → dedup → split → features."""
    df_raw = load_raw(path)
    df, n_dupes_removed = deduplicate(df_raw)
    train_df, test_df = time_split(df, train_frac=train_frac)
    X_train, X_test, y_train, y_test, scaler = build_features(train_df, test_df)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "train_df": train_df,
        "test_df": test_df,
        "n_dupes_removed": n_dupes_removed,
        "feature_names": NUMERIC_FEATURES,
    }
