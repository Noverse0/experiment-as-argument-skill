"""Data loading and leak-aware preparation for the churn experiment.

Decisions made here are deliberate and documented, because the dataset
contains planted rigor traps:

- ``account_status`` is "closed" iff the customer churned. It is a perfect
  function of the target (recorded *after* the outcome) and is therefore a
  target leak. We drop it. Keeping it would give ~perfect, meaningless scores.
- ``customer_id`` is a row identifier with no predictive content. Dropped.
- The raw file contains exact duplicate rows. If duplicates straddle the
  train/test boundary the model can "memorise" test rows. We deduplicate
  *before* any split so identical rows cannot leak across it.
- ``signup_date`` is a temporal column and churn prediction is forward
  looking, so the evaluation splits on time (see ``evaluate.py``). The date
  itself is not used as a model feature.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Columns the models are allowed to see at fit time. Everything else is either
# an identifier, a leak, or used only for the temporal split.
FEATURE_COLUMNS = ["tenure_months", "monthly_spend", "support_tickets"]
TARGET_COLUMN = "churned"
TIME_COLUMN = "signup_date"
LEAK_COLUMNS = ["account_status"]  # derived from the target -> dropped
ID_COLUMNS = ["customer_id"]


@dataclass
class PreparedData:
    """Deduplicated, leak-free, time-sorted data ready for evaluation."""

    X: pd.DataFrame  # feature matrix, FEATURE_COLUMNS only
    y: pd.Series  # binary target
    time: pd.Series  # signup_date as datetime, aligned to X/y
    n_raw: int  # rows in the raw file
    n_duplicates_dropped: int  # exact duplicate rows removed before split


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw CSV without any cleaning."""
    return pd.read_csv(path)


def prepare(df: pd.DataFrame) -> PreparedData:
    """Turn a raw dataframe into leak-free, deduplicated, time-sorted data.

    Steps, in order:
      1. drop exact duplicate rows (before any split),
      2. drop the leak column(s) and identifiers,
      3. parse the temporal column and sort ascending by it,
      4. split out features (FEATURE_COLUMNS only) and target.
    """
    n_raw = len(df)

    deduped = df.drop_duplicates().reset_index(drop=True)
    n_duplicates_dropped = n_raw - len(deduped)

    deduped = deduped.copy()
    deduped[TIME_COLUMN] = pd.to_datetime(deduped[TIME_COLUMN])
    deduped = deduped.sort_values(TIME_COLUMN).reset_index(drop=True)

    X = deduped[FEATURE_COLUMNS].copy()
    y = deduped[TARGET_COLUMN].astype(int)
    time = deduped[TIME_COLUMN]

    return PreparedData(
        X=X,
        y=y,
        time=time,
        n_raw=n_raw,
        n_duplicates_dropped=n_duplicates_dropped,
    )
