"""Data loading and leakage-aware preparation for the churn experiment.

The raw CSV contains three deliberate rigor hazards that this module neutralises:

1. ``account_status`` is derived from the target ("closed" iff churned). It is a
   perfect target leak and is dropped. Keeping it would let any model score a
   near-perfect AUC that evaporates in production where status is unknown at
   prediction time. See :func:`drop_leaky_columns`.
2. The CSV appends 200 exact duplicate rows. A random split would let copies of
   the same customer straddle train and test, inflating scores. We deduplicate
   *before* any split. See :func:`load_churn`.
3. ``signup_date`` is a temporal column and churn is a forward-looking task, so
   the experiment uses a time-ordered split (handled in ``evaluate``); this
   module exposes the sort key.

``customer_id`` is a row identifier with no predictive signal and is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TARGET = "churned"
# Columns removed before modelling and *why*. Justification lives in code, not
# in someone's head (per the data-discipline rules).
LEAK_COLUMNS = {
    "account_status": "target leak: 'closed' iff churned; unknown at predict time",
    "customer_id": "row identifier; no predictive signal",
}
TIME_COLUMN = "signup_date"
FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]


@dataclass
class ChurnData:
    """Prepared, leak-free dataset, time-ordered ascending by signup_date."""

    X: pd.DataFrame  # feature matrix, FEATURES columns only
    y: pd.Series  # binary target
    n_raw: int  # rows in the raw CSV
    n_duplicates: int  # exact duplicate rows removed
    churn_rate: float  # positive-class prevalence after dedup


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that leak the target or carry no signal."""
    present = [c for c in LEAK_COLUMNS if c in df.columns]
    return df.drop(columns=present)


def load_churn(csv_path: str) -> ChurnData:
    """Load the CSV, deduplicate, drop leaks, and time-order the rows.

    Deduplication happens before any split so identical rows cannot straddle the
    train/test boundary. Rows are sorted by ``signup_date`` (stable) so the
    downstream time-ordered split is honest.
    """
    raw = pd.read_csv(csv_path)
    n_raw = len(raw)

    # Dedup across the full row (the planted duplicates copy every column,
    # customer_id included), before splitting.
    deduped = raw.drop_duplicates().reset_index(drop=True)
    n_duplicates = n_raw - len(deduped)

    # Respect time: sort ascending so the split trains on the past, tests on the
    # future. Stable sort keeps within-day order deterministic.
    ordered = deduped.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)

    clean = drop_leaky_columns(ordered)
    X = clean[FEATURES].copy()
    y = clean[TARGET].astype(int).copy()
    return ChurnData(
        X=X,
        y=y,
        n_raw=n_raw,
        n_duplicates=n_duplicates,
        churn_rate=float(y.mean()),
    )
