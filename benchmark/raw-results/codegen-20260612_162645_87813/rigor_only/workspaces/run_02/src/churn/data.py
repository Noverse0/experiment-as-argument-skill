"""Data loading, leakage-aware cleaning, and time-ordered preparation.

Leak surface audited before any modeling (see docstrings + the report):

- ``account_status``  -> DROPPED. It is recorded *after* the outcome:
  the value is "closed" iff the customer churned, so it is a perfect proxy
  for the target. Keeping it would leak the label (verified by the
  leakage-ceiling sanity check, which gets ~1.0 AUC with it included).
- ``customer_id``     -> DROPPED. Pure row identifier, carries no signal and
  can only memorize.
- ``signup_date``     -> NOT a model feature. It is a temporal column; we use
  it only to order rows so evaluation respects time (no future -> past leak).

Legitimate predictors kept: tenure_months, monthly_spend, support_tickets.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TARGET = "churned"
DATE_COL = "signup_date"
LEAK_COLS = ["account_status"]   # derived from the label (post-outcome)
ID_COLS = ["customer_id"]        # identifier, no signal
FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]


@dataclass
class LoadResult:
    """Cleaned, time-ordered data plus the audit facts the report needs."""

    X: pd.DataFrame          # features only, ordered by signup_date
    y: pd.Series             # target, aligned to X
    n_raw: int               # rows as read from disk
    n_duplicates: int        # exact duplicate rows removed
    n_clean: int             # rows after dedup
    churn_rate: float        # P(churned == 1) after dedup
    dropped_columns: dict     # column -> reason


def load_clean(csv_path: str) -> LoadResult:
    """Load the churn CSV and return leakage-free, time-ordered features/target.

    Steps (order matters for rigor):
      1. Read raw CSV.
      2. Drop *exact* duplicate rows BEFORE any split so copies cannot
         straddle a train/test boundary (the dataset plants 200 of them).
      3. Order by signup_date so downstream time-based CV never trains on
         the future to predict the past.
      4. Drop leak + id columns; keep only justified predictors.
    """
    raw = pd.read_csv(csv_path)
    n_raw = len(raw)

    # 2) Deduplicate across the full row before splitting.
    deduped = raw.drop_duplicates().reset_index(drop=True)
    n_duplicates = n_raw - len(deduped)

    # 3) Respect time: order rows by signup date (forward-looking task).
    deduped[DATE_COL] = pd.to_datetime(deduped[DATE_COL])
    ordered = deduped.sort_values(DATE_COL, kind="mergesort").reset_index(drop=True)

    y = ordered[TARGET].astype(int)
    X = ordered[FEATURES].copy()

    dropped = {c: "target leakage (recorded after outcome)" for c in LEAK_COLS}
    dropped.update({c: "row identifier (no signal)" for c in ID_COLS})
    dropped[DATE_COL] = "temporal column used only to order folds, not as a feature"

    return LoadResult(
        X=X,
        y=y,
        n_raw=n_raw,
        n_duplicates=n_duplicates,
        n_clean=len(ordered),
        churn_rate=float(y.mean()),
        dropped_columns=dropped,
    )


def load_with_leak(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return features INCLUDING the leaky account_status (one-hot encoded).

    Used ONLY by the leakage-ceiling sanity check to demonstrate that the
    leaked column produces near-perfect, untrustworthy scores. Never used by
    the real experiment.
    """
    raw = pd.read_csv(csv_path).drop_duplicates().reset_index(drop=True)
    y = raw[TARGET].astype(int)
    X = raw[FEATURES].copy()
    X["account_status_closed"] = (raw["account_status"] == "closed").astype(int)
    return X, y
