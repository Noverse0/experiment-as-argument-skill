"""Data loading, cleaning, and splitting for the churn experiment.

Rigor rules applied:
- account_status is dropped: it encodes the target (closed iff churned) and would
  be unavailable at real prediction time — keeping it is perfect label leakage.
- Duplicates are removed before splitting so identical rows cannot straddle
  train and test, inflating test performance.
- signup_date drives a temporal split (earlier customers → train, later → test);
  it is not used as a model feature because tenure_months already captures
  time-in-service and no seasonal pattern exists in the synthetic DGP.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


TARGET = "churned"
LEAKY_COLS = ["account_status"]   # derived from target — perfect leak
ID_COLS = ["customer_id"]          # identifier, not predictive
DATE_COL = "signup_date"
FEATURE_COLS = ["tenure_months", "monthly_spend", "support_tickets"]


def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV, remove leaky columns, deduplicate, and sort chronologically."""
    df = pd.read_csv(path)

    # Drop leaky column before anything else
    df = df.drop(columns=LEAKY_COLS)

    # Deduplicate on all non-ID columns before splitting.
    # The 200 appended rows share all values except customer_id (re-indexed).
    dup_subset = [c for c in df.columns if c not in ID_COLS]
    before = len(df)
    df = df.drop_duplicates(subset=dup_subset, keep="first").reset_index(drop=True)
    n_removed = before - len(df)

    # Drop ID — not predictive
    df = df.drop(columns=ID_COLS)

    # Parse date and sort chronologically so we can do a temporal split
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    return df, n_removed


def time_split(
    df: pd.DataFrame, train_frac: float = 0.80
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological 80/20 split — earlier customers train, later customers test."""
    n_train = int(len(df) * train_frac)
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()


def build_matrices(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Return (X_train, y_train, X_test, y_test, fitted_scaler).

    Scaler is fitted on train only, then applied to test — no leakage.
    """
    X_train = train[FEATURE_COLS].to_numpy(dtype=float)
    y_train = train[TARGET].to_numpy()
    X_test = test[FEATURE_COLS].to_numpy(dtype=float)
    y_test = test[TARGET].to_numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, scaler
