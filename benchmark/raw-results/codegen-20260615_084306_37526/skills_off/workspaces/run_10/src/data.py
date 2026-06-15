"""Data loading and preparation for the churn experiment."""
import pandas as pd
import numpy as np

# Only the three causally honest features; everything else is dropped:
#   customer_id   — identifier, not a predictor
#   signup_date   — used for temporal ordering / split, not as a raw feature
#   days_since_last_login — TARGET LEAK: recorded after the churn outcome
#                           (churned customers stop logging in by definition),
#                           so including it inflates AUC without predictive validity.
FEATURE_COLS = ["tenure_months", "monthly_spend", "support_tickets"]
TARGET_COL = "churned"


def load_and_prepare(path: str) -> pd.DataFrame:
    """Load CSV, remove planted duplicates, sort by signup_date.

    The generator appends 200 exact-duplicate rows; a random split lets them
    straddle train/test, inflating held-out performance.  Drop them first.
    """
    df = pd.read_csv(path, parse_dates=["signup_date"])
    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    n_removed = n_before - n_after
    print(f"  dedup: {n_before} → {n_after} rows ({n_removed} duplicates removed)")

    # Sort by signup date so temporal CV folds respect time order.
    df = df.sort_values("signup_date").reset_index(drop=True)
    return df


def get_X_y(df: pd.DataFrame):
    return df[FEATURE_COLS].copy(), df[TARGET_COL].copy()


def class_balance(y: pd.Series) -> dict:
    counts = y.value_counts().to_dict()
    rate = y.mean()
    return {"n_positive": int(counts.get(1, 0)), "n_negative": int(counts.get(0, 0)),
            "churn_rate": float(round(rate, 4))}
