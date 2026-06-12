"""Data loading and preparation for the churn experiment.

Three properties of this dataset drive every decision here:

1. ``account_status`` is a perfect target leak: it equals ``"closed"`` if and
   only if ``churned == 1``. A model handed this column "predicts" churn at
   AUC 1.0 while learning nothing. We drop it. (Verified by ``leakage_audit``.)
2. The raw file contains exact duplicate rows. A random split lets the same
   customer land in both train and test, inflating scores. We deduplicate
   *before* any split.
3. ``signup_date`` is temporal and churn prediction is forward-looking, so a
   random split would let the model train on customers who signed up *after*
   the ones it is tested on. We sort by signup date and use a time-based split
   (see ``evaluate.time_series_cv``).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TARGET = "churned"
TIME_COLUMN = "signup_date"
# Identifier; carries no generalizable signal and must never be a feature.
ID_COLUMNS = ["customer_id"]
# Perfect target leak (account_status == "closed" iff churned == 1).
LEAK_COLUMNS = ["account_status"]
# The only columns the model is allowed to see at fit time.
FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]


@dataclass
class PreparedData:
    """Features, target, and provenance facts about the preparation."""

    X: pd.DataFrame
    y: pd.Series
    n_rows_raw: int
    n_duplicates_dropped: int
    churn_rate: float
    time: pd.Series  # signup_date, time-sorted, parallel to X/y (split key, not a feature)


def load_raw(path: str) -> pd.DataFrame:
    """Load the CSV with the time column parsed as a datetime."""
    return pd.read_csv(path, parse_dates=[TIME_COLUMN])


def leakage_audit(df: pd.DataFrame) -> dict:
    """Quantify the planted leak so the report can state it as a measured fact.

    Returns the fraction of rows where account_status is perfectly determined
    by the target. ~1.0 means the column is a perfect leak and must be dropped.
    """
    if "account_status" not in df.columns:
        return {"account_status_leak_fraction": 0.0}
    closed_is_churned = ((df["account_status"] == "closed") == (df[TARGET] == 1)).mean()
    return {"account_status_leak_fraction": float(closed_is_churned)}


def prepare(df: pd.DataFrame) -> PreparedData:
    """Dedup, drop leak/id columns, sort by time, and split off the target.

    Order matters: deduplicate *before* anything else so duplicates cannot
    straddle a later split, then time-sort so the forward-looking CV is honest.
    """
    n_rows_raw = len(df)

    # 1. Drop exact duplicate rows before any split.
    deduped = df.drop_duplicates().reset_index(drop=True)
    n_duplicates_dropped = n_rows_raw - len(deduped)

    # 2. Time-sort for the forward-looking split.
    deduped = deduped.sort_values(TIME_COLUMN, kind="mergesort").reset_index(drop=True)

    # 3. Select only the allowed feature columns (drops id + leak by omission).
    X = deduped[FEATURES].copy()
    y = deduped[TARGET].astype(int).copy()
    time = deduped[TIME_COLUMN].copy()

    return PreparedData(
        X=X,
        y=y,
        n_rows_raw=n_rows_raw,
        n_duplicates_dropped=n_duplicates_dropped,
        churn_rate=float(y.mean()),
        time=time,
    )
