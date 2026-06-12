"""Dataset loading and preprocessing for churn experiment."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_and_prepare(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load CSV, deduplicate, and prepare for ML.

    Returns:
        (X, y) where X is features and y is target (churned).
        X excludes: customer_id, account_status (leakage), signup_date.
    """
    df = pd.read_csv(csv_path)

    # Report duplicates before removing
    n_before = len(df)
    df = df.drop_duplicates(subset=df.columns.difference(['customer_id']), keep='first')
    n_dupes = n_before - len(df)
    print(f"Duplicate rows removed: {n_dupes} (from {n_before} to {len(df)})")

    # Target
    y = df['churned'].copy()

    # Features: exclude customer_id, signup_date (temporal), account_status (leakage)
    feature_cols = ['tenure_months', 'monthly_spend', 'support_tickets']
    X = df[feature_cols].copy()

    return X, y


def get_feature_names() -> list[str]:
    """Return the feature column names used in the experiment."""
    return ['tenure_months', 'monthly_spend', 'support_tickets']
