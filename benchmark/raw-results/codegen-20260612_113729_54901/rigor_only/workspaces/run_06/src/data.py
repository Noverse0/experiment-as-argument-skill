"""Load, clean, deduplicate, and prepare the churn dataset."""
import numpy as np
import pandas as pd

FEATURE_COLS = ["tenure_months", "monthly_spend", "support_tickets", "signup_days"]
TARGET_COL = "churned"

# account_status encodes the target directly ("closed" iff churned==1) — perfect leakage.
# customer_id is a row identifier with no predictive content.
_DROP_COLS = ["account_status", "customer_id"]


def load_and_clean(csv_path: str):
    """
    Load CSV, remove leaky/identifier columns, engineer features, deduplicate.

    Drops account_status (leakage: derived from churned) and customer_id (identifier).
    Converts signup_date to signup_days (days since earliest signup in dataset).
    Removes the 200 injected exact-duplicate rows to prevent train/test contamination.

    Returns:
        df: cleaned DataFrame with columns FEATURE_COLS + [TARGET_COL]
        n_removed: number of duplicate rows removed
    """
    df = pd.read_csv(csv_path)
    df = df.drop(columns=_DROP_COLS)

    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["signup_days"] = (df["signup_date"] - df["signup_date"].min()).dt.days
    df = df.drop(columns=["signup_date"])

    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)

    return df, n_removed


def prepare_arrays(df: pd.DataFrame):
    """
    Sort by signup_days and return numpy arrays ready for CV.

    Sorting by the temporal column ensures TimeSeriesSplit folds respect time order.

    Returns:
        X: (n, len(FEATURE_COLS)) float array
        y: (n,) int array
        metadata: dict with dataset statistics
    """
    df = df.sort_values("signup_days").reset_index(drop=True)
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[TARGET_COL].to_numpy(dtype=int)
    metadata = {
        "n_total": int(len(df)),
        "target_rate": float(y.mean()),
        "feature_cols": FEATURE_COLS,
    }
    return X, y, metadata


def train_test_split_temporal(X: np.ndarray, y: np.ndarray, test_frac: float = 0.20):
    """Simple 80/20 temporal split. Assumes X is already sorted by time."""
    split_idx = int(len(X) * (1 - test_frac))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
