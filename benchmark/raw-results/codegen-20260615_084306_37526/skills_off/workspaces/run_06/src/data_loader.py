"""Load, deduplicate, and clean the churn dataset.

Key decisions:
- Drop days_since_last_login: recorded after the outcome (churned customers stop
  logging in by definition), so it encodes the target. Using it would be target leakage.
- Deduplicate on customer_id: 200 exact-duplicate rows were appended; they must be
  removed before splitting to prevent train/test contamination.
- Sort by signup_date for temporal splits: random splits on temporal data are leakage.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


LEAK_COLS = ["days_since_last_login"]
ID_COLS = ["customer_id"]


def load_and_clean(path: str) -> tuple[pd.DataFrame, dict]:
    """Return cleaned DataFrame (sorted by signup time) and a stats dict."""
    df = pd.read_csv(path)
    n_raw = len(df)

    # Remove exact duplicate rows (appended at generation time).
    df = df.drop_duplicates(subset="customer_id").reset_index(drop=True)
    n_deduped = n_raw - len(df)

    # Exclude target-leaking column.
    df = df.drop(columns=LEAK_COLS)

    # Convert signup_date to an integer (days since earliest signup) for modeling.
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["signup_days"] = (df["signup_date"] - df["signup_date"].min()).dt.days
    df = df.drop(columns=["signup_date"])

    # Drop identifier.
    df = df.drop(columns=ID_COLS)

    # Sort ascending by signup time; required for TimeSeriesSplit.
    df = df.sort_values("signup_days").reset_index(drop=True)

    stats = {
        "n_raw": n_raw,
        "n_deduped": n_deduped,
        "n_clean": len(df),
        "churn_rate": float(df["churned"].mean()),
        "feature_cols": [c for c in df.columns if c != "churned"],
    }
    return df, stats


def get_X_y(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    feature_cols = [c for c in df.columns if c != "churned"]
    return df[feature_cols].values, df["churned"].values
