"""Data loading and leakage-aware cleaning for the churn experiment.

The argument this experiment supports is "GBM vs LogReg on churn". Every
decision in this module exists to keep that comparison honest, so the
reasoning is written down here rather than left implicit.

Leak surface identified before coding (confirmed empirically, see REPORT.md):

* ``account_status`` is "closed" iff ``churned == 1`` and "active" otherwise.
  It is recorded *as a consequence of* the outcome, so it encodes the target
  perfectly. Including it is target leakage -> DROPPED.
* ``customer_id`` is a row identifier with no predictive meaning -> DROPPED.
* ``signup_date`` is temporal. The task (predict churn) is forward-looking, so
  it is used to order rows for a time-based split, not fed as a raw feature.
* The raw file contains exact duplicate rows. They must be removed *before*
  splitting, otherwise the same customer can land in both train and test.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TARGET = "churned"
TIME_COL = "signup_date"
LEAK_COLS = ("account_status",)  # derived from the label
ID_COLS = ("customer_id",)
FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]


@dataclass
class LoadStats:
    """Auditable facts about what the cleaning step did."""

    n_raw: int
    n_exact_duplicates: int
    n_after_dedup: int
    churn_rate: float


def load_clean(csv_path: str) -> tuple[pd.DataFrame, LoadStats]:
    """Load the churn CSV, drop leakage/ID columns, and dedup before any split.

    Returns the cleaned frame (sorted by signup time, ready for a time split)
    and a :class:`LoadStats` record so the report can cite exact numbers
    instead of adjectives.
    """
    raw = pd.read_csv(csv_path)
    n_raw = len(raw)

    # Dedup across the WHOLE frame before splitting. Exact duplicates that
    # straddle a train/test boundary inflate test scores for free.
    n_dupes = int(raw.duplicated().sum())
    deduped = raw.drop_duplicates().reset_index(drop=True)

    # Parse time and sort ascending so a forward-chaining split is "train on
    # the past, test on the future".
    deduped[TIME_COL] = pd.to_datetime(deduped[TIME_COL])
    ordered = deduped.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)

    stats = LoadStats(
        n_raw=n_raw,
        n_exact_duplicates=n_dupes,
        n_after_dedup=len(ordered),
        churn_rate=float(ordered[TARGET].mean()),
    )
    return ordered, stats


def split_xy(df: pd.DataFrame, include_leak: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y). ``include_leak`` is for the leakage-probe sanity check only."""
    cols = list(FEATURES)
    if include_leak:
        # Encode account_status numerically purely to *demonstrate* the leak in
        # a sanity check. Never used by the real comparison.
        df = df.copy()
        df["account_status_leak"] = (df["account_status"] == "closed").astype(int)
        cols = cols + ["account_status_leak"]
    return df[cols].copy(), df[TARGET].copy()
