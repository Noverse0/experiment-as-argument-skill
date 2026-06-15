"""Data loading, deduplication, and temporal splitting."""
from __future__ import annotations

import pandas as pd

TARGET = "churned"
# Legitimate causal features only.
FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]
# days_since_last_login is derived from the outcome (churned customers stop
# logging in), so it is recorded at/after the event — a temporal leak.
LEAKY_FEATURES = ["days_since_last_login"]


def load_and_clean(path: str) -> tuple[pd.DataFrame, dict]:
    """Load CSV, remove exact duplicates, report audit info."""
    df = pd.read_csv(path, parse_dates=["signup_date"])
    n_raw = len(df)
    df = df.drop_duplicates()
    n_deduped = len(df)
    audit = {
        "n_raw": n_raw,
        "n_duplicates_removed": n_raw - n_deduped,
        "n_clean": n_deduped,
        "target_rate": float(df[TARGET].mean()),
    }
    return df, audit


def temporal_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by signup_date and split so test customers signed up later.

    Using time ordering avoids duplicate rows straddling the boundary and
    respects the forward-looking nature of churn prediction.
    """
    df = df.sort_values("signup_date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_frac))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test
