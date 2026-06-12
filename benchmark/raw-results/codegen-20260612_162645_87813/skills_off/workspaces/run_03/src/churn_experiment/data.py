"""Data loading and the data-discipline decisions for the churn experiment.

Every column-level decision here is a guard against leakage, made *before* any
model sees the data and justified in comments rather than in someone's head.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TARGET = "churned"

# Time column. churn is forward-looking, so the split must respect signup order
# (random splits on temporal data are leakage). Used for ORDERING ONLY, never as
# a model feature.
TIME_COLUMN = "signup_date"

# Columns the model may NOT see at fit time, with the reason each is excluded.
LEAK_COLUMNS = {
    # account_status == "closed" iff churned == 1. It is a deterministic function
    # of the label (set after the outcome), so it is a perfect target leak.
    "account_status": "deterministic function of the label (closed iff churned)",
    # customer_id is an arbitrary identifier with no predictive content; keeping
    # it invites the model to memorize rows.
    "customer_id": "row identifier, no predictive signal",
}

# The features the model is actually allowed to use.
FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]


@dataclass
class Dataset:
    """A loaded, de-duplicated, leak-free dataset, ordered by signup time."""

    frame: pd.DataFrame  # cleaned rows, sorted by TIME_COLUMN
    n_raw: int  # rows before dedup
    n_duplicates_removed: int
    target_rate: float

    @property
    def X(self) -> pd.DataFrame:
        return self.frame[FEATURES]

    @property
    def y(self) -> pd.Series:
        return self.frame[TARGET]


def load_dataset(path: str) -> Dataset:
    """Load the churn CSV and apply the data-discipline rules.

    Steps (order matters):
      1. Read raw rows.
      2. Drop EXACT duplicate rows before any split, so duplicates cannot
         straddle the train/test boundary and inflate scores.
      3. Sort by signup_date so a time-based split trains on the past and tests
         on the future.
      4. Expose only the approved feature columns (leak/id columns are kept in
         the frame for auditing but never returned by .X).
    """
    raw = pd.read_csv(path)
    n_raw = len(raw)

    # 2. Deduplicate across the (eventual) split boundary.
    deduped = raw.drop_duplicates().reset_index(drop=True)
    n_dupes = n_raw - len(deduped)

    # 3. Respect time: parse and sort. Stable sort keeps ties deterministic.
    deduped[TIME_COLUMN] = pd.to_datetime(deduped[TIME_COLUMN])
    ordered = deduped.sort_values(TIME_COLUMN, kind="mergesort").reset_index(
        drop=True
    )

    return Dataset(
        frame=ordered,
        n_raw=n_raw,
        n_duplicates_removed=n_dupes,
        target_rate=float(ordered[TARGET].mean()),
    )


def assert_no_leak_columns(X: pd.DataFrame) -> None:
    """Fail loudly if any known leak/id column reached the feature matrix."""
    present = set(X.columns) & set(LEAK_COLUMNS)
    if present:
        reasons = "; ".join(f"{c}: {LEAK_COLUMNS[c]}" for c in sorted(present))
        raise AssertionError(f"leak columns reached features: {reasons}")
