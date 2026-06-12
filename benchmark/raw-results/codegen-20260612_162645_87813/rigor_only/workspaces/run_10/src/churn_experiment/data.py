"""Data loading, leak handling, deduplication, and the temporal split.

Leak surface for this dataset (justified, not assumed):

* ``account_status``  — DROPPED. It is "closed" iff churned == 1, i.e. it is a
  recoded copy of the target recorded at/after the outcome. Keeping it gives a
  trivially perfect classifier and proves nothing about churn prediction.
* ``customer_id``     — DROPPED. A row identifier carries no generalizable signal.
* ``signup_date``     — NOT used as a feature, but USED to order the split.
  Churn prediction is forward-looking, so we train on earlier customers and test
  on later ones. A random split would let the model peek at the future.
* duplicate rows      — REMOVED before splitting. Exact duplicates that straddle
  the train/test boundary leak memorized rows into the test set.

Predictive features are the behavioural columns only:
``tenure_months``, ``monthly_spend``, ``support_tickets``.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TARGET = "churned"
DATE_COL = "signup_date"
FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]
LEAK_COLS = ["account_status"]  # target-derived; see module docstring
ID_COLS = ["customer_id"]


@dataclass
class SplitData:
    """A temporal dev/test split. ``test`` is touched exactly once, at the end."""

    X_dev: pd.DataFrame
    y_dev: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    n_duplicates_removed: int
    n_rows_after_dedup: int


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows. Returns (clean_df, n_removed)."""
    before = len(df)
    clean = df.drop_duplicates().reset_index(drop=True)
    return clean, before - len(clean)


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select only the justified predictive features (no leak/id columns)."""
    return df[FEATURES].copy()


def temporal_split(df: pd.DataFrame, test_frac: float = 0.25) -> SplitData:
    """Dedup, order by signup_date, hold out the most recent ``test_frac`` as test.

    Splitting after sorting by time means we never train on customers who signed
    up later than the ones we evaluate on.
    """
    clean, n_dup = deduplicate(df)
    ordered = clean.sort_values(DATE_COL, kind="mergesort").reset_index(drop=True)

    n_test = int(round(len(ordered) * test_frac))
    n_dev = len(ordered) - n_test
    dev = ordered.iloc[:n_dev]
    test = ordered.iloc[n_dev:]

    return SplitData(
        X_dev=feature_matrix(dev),
        y_dev=dev[TARGET].reset_index(drop=True),
        X_test=feature_matrix(test),
        y_test=test[TARGET].reset_index(drop=True),
        n_duplicates_removed=n_dup,
        n_rows_after_dedup=len(clean),
    )
