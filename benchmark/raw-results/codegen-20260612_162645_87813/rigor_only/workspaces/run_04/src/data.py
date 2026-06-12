"""Data loading and leakage-aware preparation.

Design decisions (the argument depends on these):

- ``account_status`` is DROPPED. It is "closed" iff ``churned == 1`` (a perfect
  target leak: it is recorded as a consequence of the outcome we predict).
  Keeping it makes any model score ~1.0 AUC and proves nothing about churn.
- ``customer_id`` is DROPPED. A pure identifier; with duplicate rows present it
  would let a model memorize specific customers.
- ``signup_date`` is NOT used as a raw feature. The churn task is forward
  looking, so the date is used ONLY to order rows for a time-based split.
- Exact duplicate rows are removed BEFORE any split so duplicates cannot
  straddle the train/test boundary (which would inflate scores).

Feature columns left to the models: tenure_months, monthly_spend,
support_tickets -- none of which can be derived from the label.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

LEAK_COLUMNS = ["account_status"]          # derived from the target
ID_COLUMNS = ["customer_id"]               # identifier, not predictive
TIME_COLUMN = "signup_date"                # used for ordering, not as a feature
TARGET = "churned"
FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]


@dataclass
class PreparedData:
    """Time-ordered, deduplicated, leak-free features and labels."""

    X: pd.DataFrame
    y: pd.Series
    n_raw: int
    n_duplicates_dropped: int
    churn_rate: float


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare(path: str) -> PreparedData:
    """Load the CSV and return leak-free, deduplicated, time-ordered data."""
    df = load_raw(path)
    n_raw = len(df)

    # Dedup BEFORE splitting so identical rows cannot straddle train/test.
    df_dedup = df.drop_duplicates().reset_index(drop=True)
    n_duplicates_dropped = n_raw - len(df_dedup)

    # Order by time: a forward-looking task must train on the past only.
    df_sorted = df_dedup.sort_values(TIME_COLUMN, kind="mergesort").reset_index(
        drop=True
    )

    y = df_sorted[TARGET].astype(int)
    X = df_sorted[FEATURES].copy()
    return PreparedData(
        X=X,
        y=y,
        n_raw=n_raw,
        n_duplicates_dropped=n_duplicates_dropped,
        churn_rate=float(y.mean()),
    )


def load_leaky_features(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Features INCLUDING the leak, for the leakage-ceiling sanity check only.

    Never used by the real experiment -- it exists to demonstrate that the
    dropped column is in fact a perfect leak.
    """
    df = load_raw(path).drop_duplicates().reset_index(drop=True)
    y = df[TARGET].astype(int)
    leak = (df["account_status"] == "closed").astype(int).rename("status_closed")
    X = pd.concat([df[FEATURES], leak], axis=1)
    return X, y
