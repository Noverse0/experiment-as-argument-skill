"""Dataset loading, deduplication, and splitting."""
import pandas as pd
import numpy as np
from typing import Tuple


def load_and_deduplicate(path: str) -> pd.DataFrame:
    """
    Load CSV and remove exact duplicate rows (keep first).

    Returns DataFrame with duplicates removed, preserving original row count
    in an attribute for reporting.
    """
    df = pd.read_csv(path)
    initial_count = len(df)

    # Find exact duplicates and drop all but the first
    df_dedup = df.drop_duplicates(keep='first')
    dedup_count = len(df_dedup)

    df_dedup.attrs['initial_rows'] = initial_count
    df_dedup.attrs['duplicates_removed'] = initial_count - dedup_count

    return df_dedup


def time_based_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    date_column: str = 'signup_date'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by signup date: train on earlier signups, test on later.

    This respects temporal structure (legitimate train/test separation)
    and prevents duplicates from straddling the boundary.
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    # Sort by date, then split by position
    df_sorted = df.sort_values(date_column).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_fraction))

    train = df_sorted.iloc[:split_idx].reset_index(drop=True)
    test = df_sorted.iloc[split_idx:].reset_index(drop=True)

    return train, test


def get_feature_columns() -> list:
    """
    Return feature columns to use in the model.

    Excludes:
    - days_since_last_login: LEAK (recorded at/after churn, not at signup)
    - customer_id, signup_date: not features, used for split/dedup only
    - churned: target variable

    Includes only honest causal features.
    """
    return ['tenure_months', 'monthly_spend', 'support_tickets']


def prepare_features(df: pd.DataFrame, target: str = 'churned') -> Tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix and target vector."""
    features = get_feature_columns()
    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.int32)
    return X, y
