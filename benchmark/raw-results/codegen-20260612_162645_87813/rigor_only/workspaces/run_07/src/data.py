"""Data loading and cleaning for the churn experiment.

Leak surface (documented decisions, not hidden in our heads):

- ``account_status``: in this dataset it is "closed" exactly when ``churned == 1``
  and "active" otherwise. It is a deterministic function of the target -- a
  recorded-after-the-outcome status flag. Keeping it would hand the model the
  answer (see the leakage-ceiling sanity check). DROPPED.
- ``customer_id``: a row identifier with no predictive content; it also happens
  to be duplicated by the planted duplicate rows. DROPPED.
- ``signup_date``: a temporal column. Churn is forward-looking, so a random
  split would let the model train on the future and test on the past. We use it
  to ORDER rows for a time-based split, and do NOT feed the raw calendar date in
  as a feature (it would encode the split/era, not customer behaviour).

Duplicate handling: the generator appends 200 exact duplicate rows. Exact
duplicates that straddle a train/test boundary are leakage (the model is tested
on rows it trained on). We drop exact duplicates BEFORE any split.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Columns the model is allowed to learn from.
FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]
TARGET = "churned"
# Columns deliberately excluded from the feature set, with the reason.
DROPPED = {
    "account_status": "target leak (closed iff churned)",
    "customer_id": "row identifier, no signal",
    "signup_date": "temporal ordering only, not a feature",
}
TIME_COL = "signup_date"


@dataclass
class CleanData:
    """Cleaned, time-ordered dataset ready for a time-based split."""

    X: pd.DataFrame  # features only, rows ordered oldest -> newest
    y: pd.Series  # target aligned to X
    n_raw: int  # rows as loaded
    n_duplicates: int  # exact duplicate rows removed
    n_clean: int  # rows after dedup
    churn_rate: float  # positive rate after dedup


def load_clean(path: str) -> CleanData:
    """Load the CSV, drop the leak/id columns, dedup, and time-order.

    Steps, in order, so the boundary discipline is auditable:
      1. read raw
      2. drop EXACT duplicate rows (before any split)
      3. sort by signup_date (oldest first) for a time-based split
      4. select the allowed feature columns + target
    """
    raw = pd.read_csv(path)
    n_raw = len(raw)

    deduped = raw.drop_duplicates()
    n_duplicates = n_raw - len(deduped)

    ordered = deduped.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)

    X = ordered[FEATURES].copy()
    y = ordered[TARGET].astype(int).copy()
    return CleanData(
        X=X,
        y=y,
        n_raw=n_raw,
        n_duplicates=n_duplicates,
        n_clean=len(ordered),
        churn_rate=float(y.mean()),
    )


def load_with_leak(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Cleaned data PLUS the leaked ``account_status`` column, encoded 0/1.

    Used only by the leakage-ceiling sanity check to demonstrate why we drop it.
    """
    raw = pd.read_csv(path).drop_duplicates()
    ordered = raw.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)
    X = ordered[FEATURES].copy()
    X["account_status_closed"] = (ordered["account_status"] == "closed").astype(int)
    y = ordered[TARGET].astype(int).copy()
    return X, y
