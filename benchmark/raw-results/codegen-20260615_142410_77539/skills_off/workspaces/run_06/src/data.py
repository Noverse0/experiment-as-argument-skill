"""Data loading, deduplication, and splitting for churn experiment."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


LEAK_FEATURES = ["days_since_last_login"]
NON_PREDICTIVE_FEATURES = ["customer_id", "signup_date"]
FEATURE_COLS = ["tenure_months", "monthly_spend", "support_tickets"]
TARGET_COL = "churned"


def load_and_deduplicate(path: str) -> pd.DataFrame:
    """Load CSV and remove exact duplicate rows, keeping first occurrence."""
    df = pd.read_csv(path)

    # Identify duplicates (excluding customer_id since it's unique by design)
    dup_cols = [c for c in df.columns if c not in ["customer_id"]]
    dup_mask = df[dup_cols].duplicated(keep="first")
    n_dup = dup_mask.sum()

    df = df[~dup_mask].reset_index(drop=True)

    return df, n_dup


def time_based_split(
    df: pd.DataFrame, train_frac: float = 0.7, seed: int = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by signup_date (temporal ordering) to prevent future leakage.
    Earlier signups → train, later signups → test.
    """
    df = df.sort_values("signup_date").reset_index(drop=True)
    split_idx = int(len(df) * train_frac)
    train = df[:split_idx].copy()
    test = df[split_idx:].copy()
    return train, test


def prepare_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract features and fit scaler on train only.
    Returns: X_train, X_test, y_train, y_test (all as arrays).
    """
    X_train = train[FEATURE_COLS].values.astype(float)
    y_train = train[TARGET_COL].values.astype(int)

    X_test = test[FEATURE_COLS].values.astype(float)
    y_test = test[TARGET_COL].values.astype(int)

    # Fit scaler on train only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def validate_no_leak(df: pd.DataFrame) -> None:
    """Warn if leak features are present."""
    leak_present = [f for f in LEAK_FEATURES if f in df.columns]
    if leak_present:
        print(f"WARNING: Leak features present (will be excluded): {leak_present}")
