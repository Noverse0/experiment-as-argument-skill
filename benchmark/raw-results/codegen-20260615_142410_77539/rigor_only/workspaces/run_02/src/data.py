"""Data loading, deduplication, and preprocessing."""
import pandas as pd
import numpy as np
from typing import Tuple, Dict


def load_and_deduplicate(csv_path: str) -> Tuple[pd.DataFrame, int]:
    """Load CSV and remove exact duplicates.

    Returns:
        (deduplicated_df, n_duplicates_removed)
    """
    df = pd.read_csv(csv_path)
    n_before = len(df)
    df = df.drop_duplicates(keep='first').reset_index(drop=True)
    n_after = len(df)
    n_removed = n_before - n_after
    return df, n_removed


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Select features for modeling.

    Excludes:
    - customer_id: identifier only
    - signup_date: temporal, random split ignores time (would be leakage)
    - days_since_last_login: target leak (churned customers have longer since-login by design)

    Uses only the legitimate causal features.
    """
    feature_cols = ['tenure_months', 'monthly_spend', 'support_tickets']
    excluded_reason = (
        "Excluded features: "
        "customer_id (identifier), "
        "signup_date (temporal, random split ignores time), "
        "days_since_last_login (target leak: churned customers have longer since-login by definition)"
    )
    return df[feature_cols].copy(), excluded_reason


def compute_class_balance(y: pd.Series) -> Dict[str, float]:
    """Compute churn rate and balance stats."""
    churn_rate = y.mean()
    n_churned = (y == 1).sum()
    n_retained = (y == 0).sum()
    return {
        'churn_rate': float(churn_rate),
        'n_churned': int(n_churned),
        'n_retained': int(n_retained),
        'n_total': int(len(y)),
    }


def train_test_split_no_leakage(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    random_state: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train/test BEFORE any fitting or transform.

    This ensures test set is held-out and touched only once at evaluation time.
    """
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # preserve class balance in both sets
    )
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit scaler on train, apply to both.

    Returns numpy arrays (sklearn models expect this).
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled
