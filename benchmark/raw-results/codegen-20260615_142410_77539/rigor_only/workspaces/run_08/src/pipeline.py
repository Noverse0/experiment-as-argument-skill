"""Data loading, deduplication, and preprocessing for churn experiment."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple


def load_and_deduplicate(csv_path: str) -> Tuple[pd.DataFrame, int]:
    """Load CSV and remove exact duplicate rows.

    Returns:
        Deduplicated dataframe and count of duplicates removed.
    """
    df = pd.read_csv(csv_path)
    n_before = len(df)
    # Keep first occurrence of each duplicate
    df = df.drop_duplicates(keep='first')
    n_after = len(df)
    n_removed = n_before - n_after
    return df, n_removed


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Extract features and target, dropping columns with leakage or no predictive value.

    Drops:
    - customer_id: unique identifier with no predictive signal
    - signup_date: temporal feature (documented as limitation in random split)
    - days_since_last_login: TARGET LEAKAGE (churned customers have higher values
                             by definition, recorded at/after the outcome)

    Features retained:
    - tenure_months: time as customer (honest causal signal)
    - monthly_spend: customer spend (honest causal signal)
    - support_tickets: support interactions (honest causal signal)
    """
    X = df[['tenure_months', 'monthly_spend', 'support_tickets']].copy()
    y = df['churned'].values
    return X, y


def split_and_preprocess(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Split before preprocessing to avoid leakage.

    1. Stratified split on target to maintain class balance
    2. Fit scaler on train only
    3. Transform train and test

    Returns:
        X_train, X_test, y_train, y_test, fitted_scaler
    """
    # For small datasets, stratified split may fail; use simple split instead
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )
    except ValueError:
        # Dataset too small for stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def load_and_prepare(csv_path: str, random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Full pipeline: load -> deduplicate -> split -> preprocess.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, metadata (dict with dedup info)
    """
    df, n_dup_removed = load_and_deduplicate(csv_path)

    # Log class balance
    churn_rate = df['churned'].mean()

    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test, scaler = split_and_preprocess(
        X, y, test_size=0.2, random_state=random_state
    )

    metadata = {
        'n_total_before_dedup': len(df) + n_dup_removed,
        'n_duplicates_removed': n_dup_removed,
        'n_total_after_dedup': len(df),
        'churn_rate': churn_rate,
        'n_train': len(y_train),
        'n_test': len(y_test),
        'train_churn_rate': y_train.mean(),
        'test_churn_rate': y_test.mean(),
    }

    return X_train, X_test, y_train, y_test, metadata
