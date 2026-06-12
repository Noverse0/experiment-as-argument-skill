"""Data loading and leak-aware preparation for the churn experiment.

Leak surface in the raw CSV (documented, then handled here):

- ``account_status``: in the generated data this column equals ``"closed"``
  exactly when ``churned == 1`` and ``"active"`` otherwise -- it is a perfect
  function of the target, recorded *after* the outcome. Keeping it would leak
  the label. DROPPED. (A dedicated leakage-ceiling check in ``experiment.py``
  encodes it on purpose to show it drives AUC to ~1.0.)
- ``customer_id``: a row identifier with no predictive content. DROPPED.
- ``signup_date``: a temporal column. Churn prediction is forward-looking, so
  the evaluation splits on this column (train on earlier signups, evaluate on
  later ones). It is NOT used as a model feature -- it carries no causal signal
  in the generator and feeding raw dates to the models would only add noise.

Duplicate rows: the generator appends 200 exact duplicates. They are removed
*before* any split so identical rows cannot straddle the train/test boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET = "churned"
LEAK_COLUMNS = ("account_status",)  # derived from the target -> dropped
ID_COLUMNS = ("customer_id",)
TIME_COLUMN = "signup_date"
FEATURE_COLUMNS = ("tenure_months", "monthly_spend", "support_tickets")


@dataclass
class PreparedData:
    """Time-ordered, deduplicated, leak-free data ready for splitting.

    ``X`` and ``y`` are sorted by signup date (ascending) so that a
    ``TimeSeriesSplit`` over the row index respects chronology.
    """

    X: pd.DataFrame
    y: pd.Series
    n_raw: int
    n_duplicates_removed: int
    churn_rate: float
    feature_columns: list[str]


def load_raw(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV with no transformation."""
    return pd.read_csv(csv_path)


def prepare(csv_path: str) -> PreparedData:
    """Load, deduplicate, drop leak/id columns, and time-sort the data.

    Returns features that are safe to fit on. No scaling/encoding happens here;
    that is fit per-fold on training data only (see ``experiment.py``).
    """
    raw = load_raw(csv_path)
    n_raw = len(raw)

    # Deduplicate exact rows BEFORE any split so duplicates cannot straddle
    # the train/test boundary.
    deduped = raw.drop_duplicates().reset_index(drop=True)
    n_duplicates_removed = n_raw - len(deduped)

    # Respect time: sort ascending by signup date so an index-based
    # TimeSeriesSplit trains on the past and evaluates on the future.
    deduped[TIME_COLUMN] = pd.to_datetime(deduped[TIME_COLUMN])
    ordered = deduped.sort_values(TIME_COLUMN, kind="mergesort").reset_index(drop=True)

    y = ordered[TARGET].astype(int)
    X = ordered[list(FEATURE_COLUMNS)].copy()

    return PreparedData(
        X=X,
        y=y,
        n_raw=n_raw,
        n_duplicates_removed=n_duplicates_removed,
        churn_rate=float(y.mean()),
        feature_columns=list(FEATURE_COLUMNS),
    )


def with_leak_feature(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Same prep as :func:`prepare` but KEEP the leaky ``account_status``.

    Used only by the leakage-ceiling sanity check to demonstrate that the
    dropped column trivially solves the task. Never used for the real claim.
    """
    raw = load_raw(csv_path)
    deduped = raw.drop_duplicates().reset_index(drop=True)
    deduped[TIME_COLUMN] = pd.to_datetime(deduped[TIME_COLUMN])
    ordered = deduped.sort_values(TIME_COLUMN, kind="mergesort").reset_index(drop=True)

    y = ordered[TARGET].astype(int)
    leak = (ordered["account_status"] == "closed").astype(int)
    X = ordered[list(FEATURE_COLUMNS)].copy()
    X["account_status_closed"] = leak.to_numpy()
    return X, y
