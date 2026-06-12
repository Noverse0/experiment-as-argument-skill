"""Data loading and the leak-prevention decisions, in one place.

Every drop/dedup decision here is a *data contact policy* choice. The reasons are
in code comments (not just in someone's head) so the experiment's argument survives review.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Columns we deliberately exclude from the model, and WHY.
#   account_status : perfect TARGET LEAK. In this dataset "closed" occurs iff churned==1
#                    (verified by crosstab). Including it makes any model score ~1.0 AUC,
#                    which measures the leak, not predictive skill. Dropped.
#   customer_id    : a unique identifier. No generalizable signal; only memorization risk.
#   signup_date    : a temporal column. Used ONLY to order rows for the time-based split.
#                    Not handed to the model as a feature.
#   churned        : the target.
LEAK_COLUMNS = ["account_status"]
ID_COLUMNS = ["customer_id"]
TIME_COLUMN = "signup_date"
TARGET = "churned"

# The only columns the model is allowed to see at fit time.
FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]


@dataclass
class LoadedData:
    """Result of loading + cleaning, with the audit numbers a report must cite."""

    df: pd.DataFrame  # deduped, time-sorted, cleaned (still has TIME_COLUMN for split ordering)
    n_raw: int
    n_duplicates_dropped: int
    positive_rate: float

    @property
    def n_clean(self) -> int:
        return len(self.df)


def load(csv_path: str) -> LoadedData:
    """Load the churn CSV and apply leak-prevention discipline.

    Steps (order matters):
      1. Read raw rows.
      2. Drop EXACT duplicate rows. The generator appends 200 verbatim copies; if they
         straddle a train/test split the model is tested on rows it trained on.
         We dedup BEFORE any split so a row lives on exactly one side.
      3. Sort by signup_date so a time-based split trains on the past, tests on the future.
    """
    raw = pd.read_csv(csv_path)
    n_raw = len(raw)

    deduped = raw.drop_duplicates().reset_index(drop=True)
    n_duplicates_dropped = n_raw - len(deduped)

    # Parse the time column for ordering; keep it for the splitter, exclude it from FEATURES.
    deduped[TIME_COLUMN] = pd.to_datetime(deduped[TIME_COLUMN])
    ordered = deduped.sort_values(TIME_COLUMN).reset_index(drop=True)

    return LoadedData(
        df=ordered,
        n_raw=n_raw,
        n_duplicates_dropped=n_duplicates_dropped,
        positive_rate=float(ordered[TARGET].mean()),
    )


def features_target(df: pd.DataFrame):
    """Return (X, y) using only the allowed FEATURES. Leak/id/time columns never enter X."""
    return df[FEATURES].copy(), df[TARGET].astype(int).copy()
