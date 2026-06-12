"""Data loading, deduplication, and split logic."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def load_and_deduplicate(path: str) -> pd.DataFrame:
    """Load CSV and remove exact duplicate rows before splitting.

    This prevents duplicates from straddling train/test boundaries,
    which would leak information across the split.
    """
    df = pd.read_csv(path)
    initial_rows = len(df)

    df_dedup = df.drop_duplicates()
    removed = initial_rows - len(df_dedup)

    print(f"[DATA] Loaded {initial_rows} rows, removed {removed} duplicates, {len(df_dedup)} remain")
    return df_dedup


def time_based_split(
    df: pd.DataFrame,
    date_col: str = "signup_date",
    train_fraction: float = 0.8,
    target_col: str = "churned",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by time to respect temporal ordering.

    Random splits on temporal data constitute leakage when the task
    is forward-looking. Here we sort by signup_date and split.
    """
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_fraction)

    train = df_sorted.iloc[:split_idx].reset_index(drop=True)
    test = df_sorted.iloc[split_idx:].reset_index(drop=True)

    train_churn_rate = train[target_col].mean()
    test_churn_rate = test[target_col].mean()

    print(f"[SPLIT] Train: {len(train)} rows ({train_churn_rate:.2%} churn) | "
          f"Test: {len(test)} rows ({test_churn_rate:.2%} churn)")

    return train, test


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select non-leaky features.

    Excludes:
    - account_status: derived from churned (perfect leak)
    - customer_id: identifier, not predictive
    - signup_date: temporal feature (already used for split)
    """
    return df[["tenure_months", "monthly_spend", "support_tickets"]]


def preprocess(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale features; fit scaler on train only to prevent leakage.

    StandardScaler fitted on train set computes mean/std, then applies
    the same transformation to test. Never touch test data at fit time.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"[PREPROCESS] Scaled {X_train.shape[1]} features")
    return X_train_scaled, X_test_scaled


def prepare_data(
    csv_path: str,
    date_col: str = "signup_date",
    target_col: str = "churned",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load, deduplicate, split, select features, and preprocess.

    Returns: X_train, X_test, y_train, y_test (all preprocessed)
    """
    df = load_and_deduplicate(csv_path)
    train, test = time_based_split(df, date_col=date_col, target_col=target_col)

    y_train = train[target_col].values
    y_test = test[target_col].values

    X_train = select_features(train)
    X_test = select_features(test)

    X_train_scaled, X_test_scaled = preprocess(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
