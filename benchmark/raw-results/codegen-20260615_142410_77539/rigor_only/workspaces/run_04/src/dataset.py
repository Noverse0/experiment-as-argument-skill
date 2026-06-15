"""Dataset loading and preprocessing for churn experiment."""
import pandas as pd
import numpy as np
from typing import Tuple


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the churn dataset from CSV."""
    df = pd.read_csv(csv_path)
    return df


def check_duplicates(df: pd.DataFrame) -> dict:
    """Audit duplicates in the dataset."""
    full_dups = df.duplicated().sum()
    subset_dups = df.duplicated(subset=["tenure_months", "monthly_spend", "support_tickets", "churned"]).sum()
    return {
        "total_rows": len(df),
        "full_duplicates": full_dups,
        "feature_duplicates": subset_dups,
    }


def time_based_split(df: pd.DataFrame, train_fraction: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by signup_date (temporal split) to avoid future leakage.
    This respects the temporal nature of the data: train on earlier customers, test on recent.
    """
    df_sorted = df.sort_values("signup_date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_fraction)
    train = df_sorted[:split_idx].reset_index(drop=True)
    test = df_sorted[split_idx:].reset_index(drop=True)

    # Check for duplicates straddling the boundary
    train_set = set(map(tuple, train[["tenure_months", "monthly_spend", "support_tickets"]].values))
    test_set = set(map(tuple, test[["tenure_months", "monthly_spend", "support_tickets"]].values))
    straddle = len(train_set & test_set)

    return train, test, {"train_size": len(train), "test_size": len(test), "feature_overlaps": straddle}


def get_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract features and target, dropping the leak column.

    Honest causal features: tenure_months, monthly_spend, support_tickets.
    Dropped:
    - days_since_last_login: LEAK. Derived from churn status (churned customers
      have longer inactive periods by definition, recorded post-outcome).
    - customer_id: Not a feature.
    - signup_date: Temporal; handled in split, not directly used.
    """
    features = df[["tenure_months", "monthly_spend", "support_tickets"]].copy()
    target = df["churned"].copy()
    return features, target


def get_all_features_with_leak(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract all features including the leak, for testing purposes."""
    features = df[["tenure_months", "monthly_spend", "support_tickets", "days_since_last_login"]].copy()
    target = df["churned"].copy()
    return features, target
