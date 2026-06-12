"""Data loading and leak defenses.

The dataset carries three documented hazards (see REPORT.md). This module
removes them *before any model sees the data*:

  1. ``account_status`` is dropped. It is "closed" iff the customer churned,
     so it is a perfect copy of the target recorded after the outcome.
  2. ``customer_id`` is dropped. It is a row identifier with no signal.
  3. Exact duplicate rows are removed once, on the full frame, so that no
     duplicate can land in both train and test.

``signup_date`` is kept only to order rows for a time-based split; it is not
fed to the models as a feature.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Columns the models are allowed to learn from.
FEATURE_COLUMNS = ["tenure_months", "monthly_spend", "support_tickets"]
TARGET_COLUMN = "churned"
DATE_COLUMN = "signup_date"

# Dropped on purpose; kept here so the reasons are auditable, not implicit.
LEAK_COLUMNS = ["account_status"]  # encodes the target (closed iff churned)
ID_COLUMNS = ["customer_id"]  # identifier, no predictive content


@dataclass
class LoadedData:
    """Cleaned data plus the provenance numbers the report must cite."""

    X: pd.DataFrame  # FEATURE_COLUMNS only, sorted by signup_date
    y: pd.Series
    dates: pd.Series  # signup_date, aligned to X, used for the time split
    n_raw: int
    n_duplicates_removed: int
    churn_rate: float


def load_clean(csv_path: str) -> LoadedData:
    """Load the CSV and apply every leak defense, returning audit counts."""
    raw = pd.read_csv(csv_path)
    n_raw = len(raw)

    # Dedup across the whole frame BEFORE splitting so duplicates cannot
    # straddle the train/test boundary.
    deduped = raw.drop_duplicates().reset_index(drop=True)
    n_duplicates_removed = n_raw - len(deduped)

    # Order by signup date so a time-based split trains on the past and tests
    # on the future. Stable sort keeps ties deterministic.
    deduped[DATE_COLUMN] = pd.to_datetime(deduped[DATE_COLUMN])
    ordered = deduped.sort_values(DATE_COLUMN, kind="stable").reset_index(drop=True)

    X = ordered[FEATURE_COLUMNS].copy()
    y = ordered[TARGET_COLUMN].astype(int).copy()
    dates = ordered[DATE_COLUMN].copy()

    return LoadedData(
        X=X,
        y=y,
        dates=dates,
        n_raw=n_raw,
        n_duplicates_removed=n_duplicates_removed,
        churn_rate=float(y.mean()),
    )
