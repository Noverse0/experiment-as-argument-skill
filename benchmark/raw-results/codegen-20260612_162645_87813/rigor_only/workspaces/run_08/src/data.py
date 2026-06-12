"""Data loading, leak audit, and cleaning for the churn dataset.

Leak surface (decided BEFORE modeling, see REPORT.md):

- ``account_status``  -> PERFECT TARGET LEAK. It is "closed" iff churned==1 and
  "active" iff churned==0. It is recorded *after* the churn outcome, so a model
  that sees it is cheating. We DROP it. (A demo in src.sanity quantifies the leak.)
- ``customer_id``     -> row identifier, carries no generalizable signal. DROP.
- ``signup_date``     -> temporal column. Not used as a predictive feature; it is
  used only to construct a time-ordered split for a robustness check.
- 200 exact duplicate rows are appended by the generator. We deduplicate BEFORE
  splitting so identical rows cannot straddle the train/test boundary (which would
  leak test rows into training and inflate metrics).

Features actually used: tenure_months, monthly_spend, support_tickets.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TARGET = "churned"
LEAK_COLUMNS = ["account_status"]          # derived from the label
ID_COLUMNS = ["customer_id"]               # identifier, no signal
TIME_COLUMN = "signup_date"                # used only for the time-based split
FEATURE_COLUMNS = ["tenure_months", "monthly_spend", "support_tickets"]


@dataclass
class DatasetAudit:
    """Facts about the raw dataset, reported (not hidden) per rigor rules."""

    n_raw: int
    n_duplicates: int
    n_after_dedup: int
    base_rate: float
    leak_columns_dropped: list[str]
    id_columns_dropped: list[str]
    time_span: tuple[str, str]


def load_raw(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def audit(df: pd.DataFrame) -> DatasetAudit:
    """Compute and return dataset facts without mutating the frame."""
    n_dupes = int(df.duplicated().sum())
    deduped = df.drop_duplicates()
    times = pd.to_datetime(df[TIME_COLUMN])
    return DatasetAudit(
        n_raw=int(len(df)),
        n_duplicates=n_dupes,
        n_after_dedup=int(len(deduped)),
        base_rate=float(df[TARGET].mean()),
        leak_columns_dropped=list(LEAK_COLUMNS),
        id_columns_dropped=list(ID_COLUMNS),
        time_span=(str(times.min().date()), str(times.max().date())),
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate and sort by signup time. Keeps TIME_COLUMN for splitting.

    Deduplication happens here, on the FULL frame, on purpose: it removes exact
    copies so they cannot end up on both sides of a later split. This is not a
    fit-like transform (it learns nothing from the data), so it does not leak.
    """
    deduped = df.drop_duplicates().reset_index(drop=True)
    deduped = deduped.sort_values(TIME_COLUMN, kind="mergesort").reset_index(drop=True)
    return deduped


def features_and_target(
    df: pd.DataFrame, *, include_leak: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) using only the approved feature columns.

    ``include_leak=True`` additionally injects ``account_status`` (encoded 0/1).
    It exists ONLY for the leakage demonstration in src.sanity; the real
    experiment never sets it.
    """
    cols = list(FEATURE_COLUMNS)
    X = df[cols].copy()
    if include_leak:
        X["account_status_closed"] = (df["account_status"] == "closed").astype(int)
    y = df[TARGET].astype(int)
    return X, y
