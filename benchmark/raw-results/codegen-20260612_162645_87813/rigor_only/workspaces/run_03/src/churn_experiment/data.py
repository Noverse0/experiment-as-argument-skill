"""Data loading and leakage-aware preparation for the churn experiment.

Leak surface audit (decided BEFORE modeling, see REPORT.md):

- ``account_status``  : DROPPED. It is a perfect function of the target
  ("closed" iff churned). Keeping it leaks the label and yields ~1.0 AUC on a
  task that is genuinely noisy. Demonstrated by the leak-ceiling sanity check.
- ``customer_id``     : DROPPED. Row identifier, carries no generalizable signal
  and could act as a memorization handle.
- ``signup_date``     : NOT a model feature. Used only to order rows for a
  time-based split, because churn prediction is forward-looking and a random
  split on temporal data is leakage.
- 200 exact duplicate rows are removed BEFORE any split so identical rows cannot
  straddle the train/test boundary (a duplicate seen in train and scored in test
  inflates metrics).

Model features are therefore the three behavioural columns:
``tenure_months``, ``monthly_spend``, ``support_tickets``.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TARGET = "churned"
LEAK_COLUMNS = ("account_status",)  # perfect target leak
ID_COLUMNS = ("customer_id",)       # identifier, no signal
TIME_COLUMN = "signup_date"         # used for ordering only, not a feature
FEATURE_COLUMNS = ("tenure_months", "monthly_spend", "support_tickets")


@dataclass
class PreparedData:
    """Result of preparing the raw CSV for the experiment.

    Rows are sorted by ``signup_date`` ascending so a forward-chaining
    (time-based) cross-validator trains on the past and tests on the future.
    """

    X: pd.DataFrame          # features only, time-ordered
    y: pd.Series             # target, aligned with X
    n_raw: int               # rows in the raw file
    n_duplicates: int        # exact duplicate rows removed
    n_final: int             # rows after dedup
    positive_rate: float     # churn base rate after dedup


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw CSV exactly as produced by ``make_dataset.py``."""
    return pd.read_csv(path)


def prepare(df: pd.DataFrame) -> PreparedData:
    """Apply the leakage-aware preparation policy.

    Steps (order matters):
    1. count and drop exact duplicate rows (before any split);
    2. sort by signup_date for a time-based split;
    3. drop leak + id columns; keep behavioural features only.
    """
    n_raw = len(df)

    deduped = df.drop_duplicates().reset_index(drop=True)
    n_duplicates = n_raw - len(deduped)

    # Time order for forward-looking evaluation.
    deduped = deduped.sort_values(TIME_COLUMN, kind="mergesort").reset_index(drop=True)

    y = deduped[TARGET].astype(int)
    drop_cols = [TARGET, TIME_COLUMN, *LEAK_COLUMNS, *ID_COLUMNS]
    X = deduped.drop(columns=drop_cols)

    # Guard the contract: only the intended behavioural features survive.
    assert list(X.columns) == list(FEATURE_COLUMNS), (
        f"unexpected feature set {list(X.columns)}; expected {list(FEATURE_COLUMNS)}"
    )

    return PreparedData(
        X=X,
        y=y,
        n_raw=n_raw,
        n_duplicates=n_duplicates,
        n_final=len(deduped),
        positive_rate=float(y.mean()),
    )


def load_prepared(path: str) -> PreparedData:
    """Convenience: load then prepare."""
    return prepare(load_raw(path))
