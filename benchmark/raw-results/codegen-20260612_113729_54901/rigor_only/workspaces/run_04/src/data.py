"""Data loading, cleaning, and splitting for the churn experiment."""
import pandas as pd
import numpy as np

FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]
TARGET = "churned"
# Columns dropped with justification:
#   customer_id   - row identifier, no predictive signal
#   signup_date   - used only for time-based splitting, not as a feature
#   account_status - LEAKAGE: "closed" iff churned==1, derived from the target
DROP_COLS = ["customer_id", "signup_date", "account_status"]


def load_and_clean(path: str) -> tuple[pd.DataFrame, dict]:
    """Load CSV, deduplicate, drop leaky/non-feature columns.

    Returns (clean_df, audit_dict) where audit_dict records what was removed.
    """
    df = pd.read_csv(path, parse_dates=["signup_date"])
    n_raw = len(df)

    # Deduplication must happen before any split to prevent train/test bleed.
    df = df.drop_duplicates()
    n_deduped = len(df)
    n_dupes_removed = n_raw - n_deduped

    # Verify that account_status is perfectly correlated with the target.
    # This makes it a label-derived feature; we must exclude it.
    leak_check = (df["account_status"] == "closed") == (df[TARGET] == 1)
    assert leak_check.all(), "account_status / churned correlation broken — re-check leak logic"

    audit = {
        "n_raw": n_raw,
        "n_after_dedup": n_deduped,
        "n_dupes_removed": n_dupes_removed,
        "target_rate": float(df[TARGET].mean()),
        "dropped_columns": DROP_COLS,
        "leak_column_confirmed": "account_status",
    }
    df = df.drop(columns=DROP_COLS)
    return df, audit


def time_based_split(
    path: str, test_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load raw CSV (before dropping signup_date), deduplicate, then split by time.

    signup_date is a temporal column. A random split would allow the model to
    learn from future information, which constitutes leakage for any
    forward-looking prediction task. We use the latest `test_frac` of
    customers (by signup_date) as the held-out test set.
    """
    df_raw = pd.read_csv(path, parse_dates=["signup_date"])
    df_raw = df_raw.drop_duplicates()

    df_sorted = df_raw.sort_values("signup_date").reset_index(drop=True)
    cutoff = int(len(df_sorted) * (1 - test_frac))
    train_raw = df_sorted.iloc[:cutoff].copy()
    test_raw = df_sorted.iloc[cutoff:].copy()

    split_info = {
        "n_train": len(train_raw),
        "n_test": len(test_raw),
        "train_signup_date_range": [
            str(train_raw["signup_date"].min().date()),
            str(train_raw["signup_date"].max().date()),
        ],
        "test_signup_date_range": [
            str(test_raw["signup_date"].min().date()),
            str(test_raw["signup_date"].max().date()),
        ],
        "train_target_rate": float(train_raw[TARGET].mean()),
        "test_target_rate": float(test_raw[TARGET].mean()),
    }

    train = train_raw.drop(columns=DROP_COLS)
    test = test_raw.drop(columns=DROP_COLS)
    return train, test, split_info


def get_Xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return df[FEATURES].values.astype(float), df[TARGET].values.astype(int)
