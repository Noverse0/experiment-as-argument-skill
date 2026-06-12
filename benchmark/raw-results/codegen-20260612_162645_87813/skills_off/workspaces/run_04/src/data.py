"""Data loading and cleaning for the churn experiment.

The cleaning here encodes the data-contact policy. Three known hazards in this
dataset are handled explicitly (see comments). Each decision is justified in
code, not just in the report.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

TARGET = "churned"

# Features the model is ALLOWED to see at fit time.
#   - tenure_months, monthly_spend, support_tickets: legitimate, known at
#     prediction time.
#   - signup_date is NOT a feature: it is the time axis used to order the
#     forward-looking split. Using it as a raw feature would let the model key
#     off cohort identity rather than behaviour.
#   - customer_id is an identifier, never a feature.
#   - account_status is DROPPED as a target leak (see clean_churn).
NUMERIC_FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]

# Documented target leak. In this dataset account_status == "closed" iff the
# customer churned, so it is a post-outcome record, not a predictor. Including
# it gives ~perfect AUC (see the leakage-ceiling sanity check) and proves
# nothing about churn prediction. We drop it and justify the drop here.
LEAK_COLUMNS = ["account_status"]

# Identifier columns: carry no signal, must not be features.
ID_COLUMNS = ["customer_id"]

TIME_COLUMN = "signup_date"


@dataclass
class CleanResult:
    """Cleaned data plus an audit trail of what cleaning did."""

    frame: pd.DataFrame
    n_raw: int
    n_duplicates_dropped: int
    leak_columns_dropped: list[str] = field(default_factory=list)
    target_rate: float = 0.0


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw CSV exactly as written by make_dataset.py."""
    return pd.read_csv(path, parse_dates=[TIME_COLUMN])


def clean_churn(df: pd.DataFrame) -> CleanResult:
    """Apply the data-contact policy.

    Steps, in order:
      1. Drop documented target-leak columns (account_status).
      2. Drop exact duplicate rows (ignoring customer_id) BEFORE any split so
         no row can straddle the train/test boundary. The dataset ships 200
         exact duplicates appended on top of 4000 unique rows.
      3. Sort chronologically by signup_date so a forward-looking split is
         possible downstream.
    """
    n_raw = len(df)

    leak_present = [c for c in LEAK_COLUMNS if c in df.columns]
    df = df.drop(columns=leak_present)

    # Deduplicate on everything except the identifier: two rows with identical
    # features+target+date are the same observation regardless of id.
    dedup_subset = [c for c in df.columns if c not in ID_COLUMNS]
    before = len(df)
    df = df.drop_duplicates(subset=dedup_subset).reset_index(drop=True)
    n_dupes = before - len(df)

    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)

    return CleanResult(
        frame=df,
        n_raw=n_raw,
        n_duplicates_dropped=n_dupes,
        leak_columns_dropped=leak_present,
        target_rate=float(df[TARGET].mean()),
    )


def features_and_target(df: pd.DataFrame):
    """Split a cleaned frame into the allowed feature matrix and target."""
    X = df[NUMERIC_FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    return X, y
