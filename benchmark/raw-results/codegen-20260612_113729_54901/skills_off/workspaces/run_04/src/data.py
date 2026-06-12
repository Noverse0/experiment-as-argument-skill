"""Data loading and preprocessing with rigor traps addressed."""
import pandas as pd
import numpy as np
from typing import Tuple

FEATURE_COLS = ["tenure_months", "monthly_spend", "support_tickets"]
TARGET_COL = "churned"
# account_status is dropped: it is derived from the target (perfect leakage).
# signup_date is used only for time-based splitting, not as a model feature.
# customer_id is an identifier, not a feature.
LEAKY_COLS = ["account_status"]
ID_COLS = ["customer_id", "signup_date"]
DROP_COLS = LEAKY_COLS + ID_COLS


def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV, deduplicate, and drop leaky/identifier columns."""
    df = pd.read_csv(path, parse_dates=["signup_date"])

    n_before = len(df)
    # Deduplicate on all feature + target columns (not customer_id, which isn't
    # meaningful for exact-duplicate detection across the planted dup rows).
    df = df.drop_duplicates(subset=FEATURE_COLS + [TARGET_COL])
    n_after = len(df)
    dedup_removed = n_before - n_after

    # Drop leaky columns (account_status is derived from the target).
    # Keep signup_date for time-based splitting; it gets excluded in get_X_y.
    cols_to_drop = [c for c in LEAKY_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    return df, dedup_removed


def time_split(df: pd.DataFrame, test_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by signup_date so no future data leaks into training."""
    df_sorted = df.sort_values("signup_date").reset_index(drop=True)
    cutoff = int(len(df_sorted) * (1 - test_frac))
    train = df_sorted.iloc[:cutoff].copy()
    test = df_sorted.iloc[cutoff:].copy()
    return train, test


def get_X_y(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[TARGET_COL].to_numpy(dtype=int)
    return X, y
