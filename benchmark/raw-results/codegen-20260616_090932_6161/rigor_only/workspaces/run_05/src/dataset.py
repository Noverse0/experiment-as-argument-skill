"""Load, deduplicate, and engineer features for the churn dataset."""
import pandas as pd
import numpy as np
from pathlib import Path


def load_and_deduplicate(csv_path: str) -> pd.DataFrame:
    """Load dataset and remove exact duplicates before any processing.

    Args:
        csv_path: Path to the churn.csv file.

    Returns:
        DataFrame with duplicates removed, customer_id preserved.
    """
    df = pd.read_csv(csv_path)
    n_before = len(df)

    # Deduplicate by all columns except customer_id (which may differ on dupes).
    cols_for_dedup = [c for c in df.columns if c != 'customer_id']
    df_dedup = df.drop_duplicates(subset=cols_for_dedup, keep='first')

    n_after = len(df_dedup)
    print(f"Deduplicated: {n_before} → {n_after} rows (removed {n_before - n_after} exact duplicates)")

    return df_dedup.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from raw columns.

    Intentionally excludes days_since_last_login (target leakage).
    Includes: tenure_months, monthly_spend, support_tickets, days_since_signup.

    Args:
        df: Raw dataframe with churn, signup_date, tenure_months, etc.

    Returns:
        DataFrame with engineered features.
    """
    df = df.copy()

    # Create temporal feature: days since signup.
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    # Use a fixed reference date (last row's date or max date) to be deterministic.
    reference_date = pd.Timestamp('2024-12-31')  # Fixed reference.
    df['days_since_signup'] = (reference_date - df['signup_date']).dt.days

    # Prepare feature matrix and target.
    feature_cols = ['tenure_months', 'monthly_spend', 'support_tickets', 'days_since_signup']

    X = df[feature_cols].copy()
    y = df['churned'].copy()

    # Check for missing values.
    assert X.isnull().sum().sum() == 0, "Features contain NaN"
    assert y.isnull().sum() == 0, "Target contains NaN"

    return X, y, feature_cols


def get_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.3, random_state: int = None):
    """Stratified train/test split, respecting class balance.

    Args:
        X: Feature matrix.
        y: Target series.
        test_size: Fraction for test set.
        random_state: Random seed.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def report_class_balance(y: pd.Series, label: str = ""):
    """Report class distribution."""
    counts = y.value_counts().sort_index()
    pct = (counts / len(y) * 100).round(1)
    rate = counts.get(1, 0) / len(y) * 100
    print(f"{label} class balance: {dict(counts)} (churn rate: {rate:.1f}%)")
    return rate
