"""Data loading and leakage-aware preparation for the churn experiment.

Every cleaning decision here is a *data contact policy* choice, documented inline
so the argument survives review. The raw generator plants three traps; this module
neutralises each one before any model sees the data.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Features the models are ALLOWED to see at fit time.
# Deliberately excludes:
#   - customer_id      : a bare identifier, no signal, pure overfit risk.
#   - signup_date      : temporal; used ONLY to order the time-based split, not as a feature.
#   - account_status   : TARGET LEAK. The generator sets it to "closed" iff churned==1,
#                        so it encodes the label perfectly. Including it yields ~1.0 AUC
#                        (see the leakage-ceiling sanity check) and proves nothing.
FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]
TARGET = "churned"
TIME_COL = "signup_date"
LEAK_COLS = ["account_status"]
ID_COLS = ["customer_id"]


@dataclass
class PreparedData:
    """Time-ordered, deduplicated, leak-free data ready for evaluation."""

    X: pd.DataFrame  # FEATURES only, sorted by signup_date ascending
    y: pd.Series
    n_raw: int
    n_duplicates_dropped: int
    churn_rate: float
    feature_names: list[str]


def load_raw(path: str) -> pd.DataFrame:
    """Load the CSV exactly as written, parsing the temporal column."""
    df = pd.read_csv(path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    return df


def prepare(path: str) -> PreparedData:
    """Apply the data contact policy and return model-ready data.

    Order matters and follows the rigor rules:
      1. Deduplicate BEFORE any split so identical rows can't straddle train/test.
      2. Sort by signup_date so a forward-looking split trains on the past only.
      3. Keep only the allowed feature columns (drops id, time, and the leak).
    """
    raw = load_raw(path)
    n_raw = len(raw)

    # (1) Dedup across the whole frame, before splitting. The generator appends
    # 200 exact duplicates; a random split would leak them across the boundary.
    deduped = raw.drop_duplicates().reset_index(drop=True)
    n_duplicates_dropped = n_raw - len(deduped)

    # (2) Respect time: churn is forward-looking, so order by signup_date and let
    # the evaluator split chronologically (past -> future).
    ordered = deduped.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)

    # (3) Restrict to the allowed feature surface. Dropping account_status here is
    # the single most important line in the experiment.
    X = ordered[FEATURES].copy()
    y = ordered[TARGET].astype(int).copy()

    return PreparedData(
        X=X,
        y=y,
        n_raw=n_raw,
        n_duplicates_dropped=n_duplicates_dropped,
        churn_rate=float(y.mean()),
        feature_names=list(FEATURES),
    )
