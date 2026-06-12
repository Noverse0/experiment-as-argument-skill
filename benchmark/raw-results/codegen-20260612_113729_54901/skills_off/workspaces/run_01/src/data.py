"""Data loading and cleaning for the churn experiment."""
import pandas as pd
import numpy as np

# account_status is derived from churned ("closed" iff churned=1) — perfect label leakage.
_LEAK_COLS = ["account_status"]
_ID_COLS = ["customer_id"]
_REFERENCE_DATE = pd.Timestamp("2023-01-01")


def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV, remove leaky columns, deduplicate, encode dates, sort by time."""
    df = pd.read_csv(path)

    # 1. Deduplicate before any split to prevent train/test contamination.
    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    if n_before != n_after:
        print(f"Deduplication: removed {n_before - n_after} exact duplicate rows "
              f"({n_before} → {n_after})")

    # 2. Drop leaky column — account_status encodes the label directly.
    df = df.drop(columns=_LEAK_COLS, errors="ignore")

    # 3. Drop identifier — carries no signal.
    df = df.drop(columns=_ID_COLS, errors="ignore")

    # 4. Convert signup_date to numeric (days from reference) and sort.
    #    Sorting ensures TimeSeriesSplit respects temporal ordering.
    df["signup_days"] = (
        pd.to_datetime(df["signup_date"]) - _REFERENCE_DATE
    ).dt.days
    df = df.drop(columns=["signup_date"])
    df = df.sort_values("signup_days").reset_index(drop=True)

    return df


def get_features_and_target(df: pd.DataFrame):
    """Split into feature matrix X and target series y."""
    target = "churned"
    features = [c for c in df.columns if c != target]
    return df[features], df[target]


def class_balance_report(y: pd.Series) -> dict:
    positive_rate = float(y.mean())
    return {
        "n_samples": int(len(y)),
        "positive_rate": round(positive_rate, 4),
        "class_ratio": f"{y.sum()}:{(y == 0).sum()} (churn:active)",
    }
