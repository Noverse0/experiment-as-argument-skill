"""Data loading, deduplication, splitting, and preprocessing."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load CSV and drop the leaked feature (account_status)."""
    df = pd.read_csv(csv_path)
    # account_status is derived from churned (closed iff churned=1).
    # This is perfect leakage — drop it.
    df = df.drop(columns=["account_status"])
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows (before splitting)."""
    # Dataset has 200 exact duplicates appended; remove them.
    duplicates_before = len(df)
    df = df.drop_duplicates()
    duplicates_after = len(df)
    n_dropped = duplicates_before - duplicates_after
    print(f"Dropped {n_dropped} exact duplicate rows")
    return df


def time_based_split(
    df: pd.DataFrame, train_ratio: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by signup_date to avoid temporal leakage.
    Earlier signups → train, later → test.
    """
    df = df.sort_values("signup_date")
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)
    print(
        f"Time-based split: {len(train)} train, {len(test)} test "
        f"(cutoff: {df.iloc[split_idx]['signup_date']})"
    )
    return train, test


def check_no_leakage(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Verify no exact duplicates straddle the train/test boundary."""
    train_ids = set(train["customer_id"])
    test_ids = set(test["customer_id"])
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"Data leakage: {len(overlap)} customer IDs in both train/test")
    print("✓ No duplicate customer_ids straddle train/test boundary")


def preprocess(
    train: pd.DataFrame, test: pd.DataFrame, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare features and targets.
    - Drop customer_id (not a feature)
    - Convert signup_date to days since first signup
    - Scale features (fitted on train only)
    - Return X_train, X_test, y_train, y_test
    """
    # Extract targets
    y_train = train["churned"].values
    y_test = test["churned"].values

    # Drop non-feature columns
    X_train = train.drop(columns=["customer_id", "signup_date", "churned"]).copy()
    X_test = test.drop(columns=["customer_id", "signup_date", "churned"]).copy()

    # Fit scaler on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def check_class_balance(y_train: np.ndarray, y_test: np.ndarray) -> None:
    """Report and verify class balance."""
    train_pos_rate = y_train.mean()
    test_pos_rate = y_test.mean()
    print(
        f"Class balance: train churn_rate={train_pos_rate:.3f}, "
        f"test churn_rate={test_pos_rate:.3f}"
    )
