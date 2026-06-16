"""Data loading, leak detection, and preprocessing."""
import pandas as pd
import numpy as np
from typing import Tuple

def load_churn_data(path: str) -> pd.DataFrame:
    """Load churn dataset."""
    return pd.read_csv(path)

def check_duplicates(df: pd.DataFrame) -> int:
    """Count exact duplicate rows. Return count of duplicates found."""
    # Exclude customer_id since it varies for duplicates
    cols_to_check = [c for c in df.columns if c != 'customer_id']
    duplicates = df.duplicated(subset=cols_to_check, keep=False)
    return duplicates.sum()

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicates, keeping first occurrence."""
    cols_to_check = [c for c in df.columns if c != 'customer_id']
    df_dedup = df.drop_duplicates(subset=cols_to_check, keep='first').reset_index(drop=True)
    return df_dedup

def detect_leak_days_since_login(df: pd.DataFrame) -> dict:
    """
    Timing test for days_since_last_login leakage.
    If churned=1 customers have much higher days_since_last_login,
    this is a strong signal of post-outcome measurement.

    Returns dict with statistics for the report.
    """
    churned = df[df['churned'] == 1]['days_since_last_login']
    active = df[df['churned'] == 0]['days_since_last_login']

    return {
        'churned_mean': float(churned.mean()),
        'churned_std': float(churned.std()),
        'active_mean': float(active.mean()),
        'active_std': float(active.std()),
        'diff_mean': float(churned.mean() - active.mean()),
    }

def prepare_features(df: pd.DataFrame, include_leaky: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature matrix and target.

    Safe features (pre-outcome snapshots):
    - tenure_months: fixed from signup date
    - monthly_spend: historical value
    - support_tickets: count up to data collection

    Leaky feature (excluded by default):
    - days_since_last_login: measured after churn decision

    Excluded:
    - customer_id: identifier
    - signup_date: we have tenure_months
    - churned: target
    """
    if include_leaky:
        feature_cols = ['tenure_months', 'monthly_spend', 'support_tickets', 'days_since_last_login']
    else:
        feature_cols = ['tenure_months', 'monthly_spend', 'support_tickets']

    X = df[feature_cols].values.astype(np.float32)
    y = df['churned'].values.astype(int)

    return X, y

def get_baseline_predictions(y: np.ndarray) -> np.ndarray:
    """Return majority class predictions (churn rate)."""
    churn_rate = y.mean()
    return np.full_like(y, 1 if churn_rate > 0.5 else 0, dtype=int)

def report_class_distribution(y: np.ndarray) -> dict:
    """Report class balance."""
    return {
        'n_samples': len(y),
        'n_churned': int((y == 1).sum()),
        'n_active': int((y == 0).sum()),
        'churn_rate': float(y.mean()),
    }
