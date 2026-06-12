"""Data loading and leakage-aware preparation for the churn experiment.

The single source of truth for which columns the model may see at fit time.
Every drop below is a deliberate decision, justified inline -- not an accident.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TARGET = "churned"
TIME_COL = "signup_date"

# Numeric predictors the model is allowed to use.
FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]

# Columns deliberately excluded from features, with reasons:
#   customer_id    -> row identifier, carries no signal, risks memorization.
#   account_status -> TARGET LEAK: it is "closed" iff churned == 1 (perfect proxy
#                     for the label, recorded after the outcome). Including it
#                     makes the task trivially perfect; see the leakage-ceiling
#                     sanity check in experiment.py.
#   signup_date    -> temporal column used to ORDER the time-based split, not a
#                     feature (it carries no churn signal in this dataset).
LEAK_COLS = ["account_status"]
ID_COLS = ["customer_id"]


@dataclass
class PreparedData:
    """Cleaned, deduplicated, time-sorted data ready for evaluation."""

    X: pd.DataFrame  # FEATURES only, sorted by signup_date
    y: pd.Series  # TARGET, aligned to X
    order_dates: pd.Series  # signup_date (datetime), aligned to X
    n_raw: int
    n_after_dedup: int
    n_duplicates_removed: int
    churn_rate: float


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw CSV exactly as generated."""
    return pd.read_csv(path)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Dedup full records, then sort chronologically. Shared by prepare() and the
    leaky-feature builder so their rows stay aligned to the same labels.

    Deduplication happens BEFORE any split so the 200 appended exact-duplicate
    rows cannot straddle the train/test boundary (which would leak labels).
    Sorting by signup_date puts rows in chronological order for the time split.
    """
    work = df.copy()
    work[TIME_COL] = pd.to_datetime(work[TIME_COL])
    work = work.drop_duplicates().reset_index(drop=True)
    work = work.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)
    return work


def prepare(df: pd.DataFrame) -> PreparedData:
    """Clean -> dedup -> time-sort. All fit-like work happens downstream, per fold."""
    n_raw = len(df)

    work = _clean(df)
    n_after_dedup = len(work)

    X = work[FEATURES].copy()
    y = work[TARGET].astype(int).copy()
    order_dates = work[TIME_COL].copy()

    return PreparedData(
        X=X,
        y=y,
        order_dates=order_dates,
        n_raw=n_raw,
        n_after_dedup=n_after_dedup,
        n_duplicates_removed=n_raw - n_after_dedup,
        churn_rate=float(y.mean()),
    )


def build_leaky_features(df: pd.DataFrame) -> pd.DataFrame:
    """FEATURES plus the leaked account_status, encoded numerically.

    Used ONLY by the leakage-ceiling sanity check to demonstrate that including
    account_status yields near-perfect performance. Never used for the real run.
    Goes through the same _clean() path so its rows align with prepare()'s labels.
    """
    work = _clean(df)
    out = work[FEATURES].copy()
    out["account_status_closed"] = (work["account_status"] == "closed").astype(int)
    return out
