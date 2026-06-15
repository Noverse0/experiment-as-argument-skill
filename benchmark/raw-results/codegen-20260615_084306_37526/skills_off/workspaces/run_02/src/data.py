"""Data loading, deduplication, feature engineering, and time-based splitting.

Key design decisions:
- Deduplicate on all columns (including customer_id) before splitting to prevent
  duplicate rows from straddling train/test.
- Exclude days_since_last_login: target leak. This value is recorded at/after the
  churn event — a churned customer has stopped logging in, so a high value directly
  encodes the outcome. Including it would inflate performance in a way that does not
  transfer to production, where churn has not yet occurred.
- Use a time-based split (sort by signup_date) so that training customers predate
  test customers. Random splits on temporal data are a form of leakage.
- Fit the StandardScaler on the training fold only; apply to test.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

REFERENCE_DATE = pd.Timestamp("2023-01-01")

FEATURE_COLS = ["tenure_months", "monthly_spend", "support_tickets", "signup_days"]
TARGET_COL = "churned"
LEAK_COLS = ["days_since_last_login"]
ID_COLS = ["customer_id"]


def load_and_prepare(
    path: str,
    train_frac: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, dict]:
    """Load, clean, split, and scale the churn dataset.

    Returns (X_train, X_test, y_train, y_test, fitted_scaler, metadata).
    """
    df = pd.read_csv(path)
    original_size = len(df)

    # Step 1: remove exact duplicates before splitting
    df = df.drop_duplicates()
    deduped_size = len(df)

    # Step 2: drop identifier — not a feature
    df = df.drop(columns=ID_COLS)

    # Step 3: exclude target leak
    df = df.drop(columns=LEAK_COLS)

    # Step 4: convert signup_date to a numeric feature
    df["signup_days"] = (
        pd.to_datetime(df["signup_date"]) - REFERENCE_DATE
    ).dt.days
    df = df.drop(columns=["signup_date"])

    # Step 5: time-based split — earlier signups → train, later → test
    df = df.sort_values("signup_days").reset_index(drop=True)
    split_idx = int(len(df) * train_frac)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train_raw = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values
    X_test_raw = test_df[FEATURE_COLS].values
    y_test = test_df[TARGET_COL].values

    # Step 6: scale — fit on train only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    meta = {
        "feature_names": FEATURE_COLS,
        "original_size": original_size,
        "deduped_size": deduped_size,
        "duplicates_removed": original_size - deduped_size,
        "total_size": deduped_size,
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "train_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
        "train_frac": train_frac,
    }
    return X_train, X_test, y_train, y_test, scaler, meta
