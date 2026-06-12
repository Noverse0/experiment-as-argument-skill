"""Data loading and cleaning for the churn experiment.

All cleaning decisions here are defensive against the leak surface identified
before coding (see tasks/todo.md and REPORT.md):

- ``account_status`` is a perfect target leak (it is "closed" iff churned). DROP.
- ``customer_id`` is an identifier. DROP.
- Exact duplicate rows exist; they must be removed BEFORE any split so they
  cannot straddle train/test.
- ``signup_date`` is temporal; the task (churn prediction) is forward-looking,
  so it is used to ORDER rows for a time-based split, never as a model feature.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TARGET = "churned"
# Columns deliberately excluded from the feature matrix, with reasons.
LEAK_COLUMN = "account_status"  # derived from the target -> perfect leak
ID_COLUMN = "customer_id"  # identifier, no predictive meaning
TIME_COLUMN = "signup_date"  # temporal ordering only, not a feature
FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]


@dataclass
class CleanData:
    """Cleaned dataset, ordered by time, ready for a time-based split."""

    X: pd.DataFrame  # feature matrix (FEATURES only), time-ordered
    y: pd.Series  # target, aligned to X
    n_raw: int  # rows before dedup
    n_duplicates_removed: int  # exact duplicate rows dropped
    churn_rate: float  # positive class fraction after dedup


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw CSV with no transformation."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> CleanData:
    """Clean raw data: dedup, drop leak/id, order by time, select features.

    Dedup happens before anything else so duplicates cannot leak across a
    later split. We sort by ``signup_date`` so a TimeSeriesSplit trains on the
    past and validates on the future.
    """
    n_raw = len(df)

    # Remove exact duplicate rows BEFORE splitting (anti-leakage).
    deduped = df.drop_duplicates().reset_index(drop=True)
    n_duplicates_removed = n_raw - len(deduped)

    # Time order for forward-looking evaluation.
    ordered = deduped.sort_values(TIME_COLUMN, kind="mergesort").reset_index(
        drop=True
    )

    X = ordered[FEATURES].copy()
    y = ordered[TARGET].astype(int).copy()
    churn_rate = float(y.mean())

    return CleanData(
        X=X,
        y=y,
        n_raw=n_raw,
        n_duplicates_removed=n_duplicates_removed,
        churn_rate=churn_rate,
    )


def leaky_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix that INCLUDES the leak column, for the leakage demo only.

    Never used for the real comparison. Exists so the experiment can *show*
    (not just assert) that account_status drives AUC to ~1.0.
    """
    deduped = df.drop_duplicates().reset_index(drop=True)
    ordered = deduped.sort_values(TIME_COLUMN, kind="mergesort").reset_index(
        drop=True
    )
    X = ordered[FEATURES].copy()
    # Encode the leak column as a 0/1 indicator.
    X[LEAK_COLUMN] = (ordered[LEAK_COLUMN] == "closed").astype(int)
    return X
