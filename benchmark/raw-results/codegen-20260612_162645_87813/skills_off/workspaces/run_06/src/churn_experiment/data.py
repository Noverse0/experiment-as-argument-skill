"""Data loading, auditing, and splitting.

Discipline enforced here:
- Audit the raw data (target rate, duplicates) before any modeling.
- Deduplicate BEFORE splitting so identical rows cannot straddle train/test.
- Split chronologically (the task is forward-looking) rather than randomly.
- Expose only non-leaking numeric features to the model.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config


@dataclass
class DataAudit:
    n_rows_raw: int
    n_duplicate_rows: int
    n_rows_deduped: int
    churn_rate: float
    majority_class_rate: float
    time_min: str
    time_max: str
    account_status_is_leak: bool


def load_raw(path: str) -> pd.DataFrame:
    """Load the CSV with signup_date parsed as a real timestamp."""
    df = pd.read_csv(path, parse_dates=[config.TIME_COL])
    return df


def audit(df: pd.DataFrame) -> DataAudit:
    """Characterize the raw data before touching a model.

    Also confirms the account_status leak empirically rather than trusting a
    comment: if account_status partitions the target perfectly, it is a leak.
    """
    dup = int(df.duplicated().sum())
    # account_status leak check: does each status map to a single churn value?
    leak = False
    if "account_status" in df.columns:
        purity = df.groupby("account_status")[config.TARGET].nunique()
        leak = bool((purity == 1).all())
    return DataAudit(
        n_rows_raw=len(df),
        n_duplicate_rows=dup,
        n_rows_deduped=len(df) - dup,
        churn_rate=float(df[config.TARGET].mean()),
        majority_class_rate=float(df[config.TARGET].value_counts(normalize=True).max()),
        time_min=str(df[config.TIME_COL].min().date()),
        time_max=str(df[config.TIME_COL].max().date()),
        account_status_is_leak=leak,
    )


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows. Done before the split so duplicates cannot
    land on both sides of the train/test boundary (a classic leak)."""
    return df.drop_duplicates().reset_index(drop=True)


def chronological_split(df: pd.DataFrame, test_fraction: float = config.TEST_FRACTION):
    """Split by time: earliest rows -> train, latest -> test.

    A random split would leak future information into training for a
    forward-looking task. Sorting by signup_date and slicing keeps all test
    customers strictly later (or equal date) than train customers.
    Returns (train_df, test_df), each sorted ascending by time.
    """
    ordered = df.sort_values(config.TIME_COL, kind="mergesort").reset_index(drop=True)
    cutoff = int(round(len(ordered) * (1 - test_fraction)))
    train_df = ordered.iloc[:cutoff].reset_index(drop=True)
    test_df = ordered.iloc[cutoff:].reset_index(drop=True)
    return train_df, test_df


def to_xy(df: pd.DataFrame):
    """Project to the allowed feature matrix X and target y.

    Only config.FEATURES are exposed. customer_id, account_status (leak), and
    signup_date (used only for ordering) are intentionally excluded.
    """
    X = df[config.FEATURES].copy()
    y = df[config.TARGET].astype(int).copy()
    return X, y
