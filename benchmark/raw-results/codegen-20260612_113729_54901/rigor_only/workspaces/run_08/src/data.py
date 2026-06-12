"""Data loading, deduplication, feature engineering, and temporal splitting."""
import pandas as pd
import numpy as np
from typing import Tuple, Dict


def load_and_clean(path: str) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Load dataset, remove leakage features, deduplicate, engineer temporal features.

    Leakage decisions:
    - account_status: dropped — derived directly from target ("closed" iff churned==1)
    - customer_id: dropped — identifier with no predictive signal
    - signup_date: converted to year/month/day_of_year numeric features, then dropped

    Deduplication is done BEFORE splitting to prevent the same row appearing in
    both train and test sets (200 exact duplicates are planted in this dataset).

    Data is sorted by signup_date so temporal_split() can cut chronologically.
    """
    df = pd.read_csv(path)

    # Drop direct label leak: account_status == "closed" iff churned == 1
    df = df.drop(columns=["account_status"])

    # Deduplicate before any split to avoid cross-split contamination
    n_before = len(df)
    df = df.drop_duplicates()
    n_dupes = n_before - len(df)

    # Parse temporal column; sort ascending so iloc-based split is chronological
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df = df.sort_values("signup_date").reset_index(drop=True)

    # Numeric proxies for signup date — no future information encoded
    df["signup_year"] = df["signup_date"].dt.year
    df["signup_month"] = df["signup_date"].dt.month
    df["signup_dayofyear"] = df["signup_date"].dt.dayofyear

    df = df.drop(columns=["customer_id", "signup_date"])

    y = df.pop("churned")
    X = df

    metadata: Dict = {
        "n_rows": len(X),
        "n_dupes_removed": n_dupes,
        "churn_rate": float(y.mean()),
        "features": list(X.columns),
    }

    return X, y, metadata


def temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_frac: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Chronological holdout split: first (1-test_frac) rows → train, last test_frac → test.
    Requires data to be pre-sorted by signup_date (load_and_clean does this).
    """
    n = len(X)
    cutoff = int(n * (1 - test_frac))
    return (
        X.iloc[:cutoff].copy(),
        X.iloc[cutoff:].copy(),
        y.iloc[:cutoff].copy(),
        y.iloc[cutoff:].copy(),
    )
