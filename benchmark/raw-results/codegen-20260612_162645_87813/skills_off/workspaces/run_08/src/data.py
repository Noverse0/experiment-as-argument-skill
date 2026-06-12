"""Data loading and leak-free preparation for the churn experiment.

Every decision here is a data-discipline decision, justified inline:

- account_status is dropped because it is a PERFECT TARGET LEAK: in this dataset
  it equals "closed" if and only if churned == 1 (verified: 3067 active rows are
  all churn=0, 1133 closed rows are all churn=1). Keeping it makes any model
  trivially perfect and proves nothing about churn prediction.
- customer_id is dropped: a bare identifier carries no generalizable signal.
- signup_date is NOT used as a feature. It is temporal and the task is
  forward-looking, so it is used ONLY to order rows for a time-based split.
  A random split on temporal data is leakage (the model would peek at the future).
- Exact duplicate rows are removed BEFORE any split, otherwise the 200 planted
  duplicates would straddle train/test and inflate scores.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# The only columns the model is allowed to see at fit time.
FEATURE_COLUMNS = ["tenure_months", "monthly_spend", "support_tickets"]
TARGET_COLUMN = "churned"

# Columns deliberately excluded from features, with the reason.
EXCLUDED_COLUMNS = {
    "customer_id": "bare identifier, no generalizable signal",
    "account_status": "perfect target leak (closed <=> churned)",
    "signup_date": "temporal; used only to order the time-based split",
}


@dataclass
class PreparedData:
    """Output of `prepare`. `X`/`y` are time-ordered (oldest signup first)."""

    X: pd.DataFrame
    y: pd.Series
    n_raw: int
    n_duplicates_removed: int
    churn_rate: float
    order_column: str = "signup_date"


def load_raw(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV with no transformation. Parses signup_date for ordering."""
    df = pd.read_csv(csv_path, parse_dates=["signup_date"])
    return df


def count_exact_duplicates(df: pd.DataFrame) -> int:
    """Number of rows that are exact duplicates of an earlier row."""
    return int(df.duplicated().sum())


def prepare(csv_path: str) -> PreparedData:
    """Load, dedup, drop leaks/identifiers, and time-order the data.

    Returns features restricted to FEATURE_COLUMNS only. Rows are sorted by
    signup_date ascending so a downstream TimeSeriesSplit trains on the past
    and tests on the future.
    """
    raw = load_raw(csv_path)
    n_raw = len(raw)

    n_dupes = count_exact_duplicates(raw)
    deduped = raw.drop_duplicates().reset_index(drop=True)

    # Time order: oldest signups first. Stable sort keeps determinism.
    ordered = deduped.sort_values("signup_date", kind="stable").reset_index(drop=True)

    X = ordered[FEATURE_COLUMNS].copy()
    y = ordered[TARGET_COLUMN].astype(int).copy()

    return PreparedData(
        X=X,
        y=y,
        n_raw=n_raw,
        n_duplicates_removed=n_dupes,
        churn_rate=float(y.mean()),
    )


def leaky_features(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Build a feature matrix that INCLUDES the leaky account_status column.

    Used ONLY by the leakage-ceiling sanity check to demonstrate that the
    dropped column would have produced a near-perfect (and meaningless) score.
    Never used in the real comparison.
    """
    raw = load_raw(csv_path).drop_duplicates().reset_index(drop=True)
    X = raw[FEATURE_COLUMNS].copy()
    X["account_status_closed"] = (raw["account_status"] == "closed").astype(int)
    y = raw[TARGET_COLUMN].astype(int).copy()
    return X, y
