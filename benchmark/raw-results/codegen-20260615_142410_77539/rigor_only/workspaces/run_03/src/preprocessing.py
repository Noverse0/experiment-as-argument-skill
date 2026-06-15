"""Data loading and preprocessing for churn experiment.

Split-before-transform discipline: all fit operations (scaling, encoding)
happen after the train/test split, fitted on train only.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    """Load churn dataset."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows")
    return df


def check_duplicates(df: pd.DataFrame) -> int:
    """Count exact duplicate rows. Report before split."""
    n_dup = df.duplicated(subset=df.columns.difference(['customer_id'])).sum()
    print(f"Found {n_dup} duplicate rows")
    return n_dup


def time_based_split(df: pd.DataFrame, train_ratio: float = 0.8, seed: int = 42):
    """Split by signup_date to respect temporal order (avoid time leakage).

    Returns train_df, test_df.
    """
    df_sorted = df.sort_values('signup_date').reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_ratio)
    train = df_sorted[:split_idx].copy()
    test = df_sorted[split_idx:].copy()
    print(f"Time-based split: train={len(train)}, test={len(test)}")
    return train, test


def engineer_features(df: pd.DataFrame, fit_scaler: StandardScaler = None) -> pd.DataFrame:
    """Engineer features from raw columns.

    LEAK AUDIT:
    - days_since_last_login: DROPPED. This is a target leak: churned customers
      have stopped logging in by definition, so this value is recorded at/after
      the outcome. Keeping it creates a suspiciously high AUC that hides the
      true model performance.
    - signup_date: converted to days since first signup (temporal distance).
      We do not use day-of-week or cyclical features that could encode future
      patterns. Simple linear temporal distance is safer.

    Args:
        df: input dataframe
        fit_scaler: if None, fit a new scaler on this data. If provided, use
                   the existing scaler (for test set). This enforces fit-on-train-only.

    Returns:
        df_features: dataframe with engineered features
        scaler: the scaler object (for apply to test set)
    """
    df = df.copy()

    # Convert signup_date to days since first date
    min_date = pd.Timestamp("2023-01-01")
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['days_since_signup'] = (df['signup_date'] - min_date).dt.days

    # Select features (dropping customer_id and signup_date, removing the leak)
    feature_cols = ['tenure_months', 'monthly_spend', 'support_tickets', 'days_since_signup']
    X = df[feature_cols].copy()

    # Scale features
    if fit_scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        scaler = fit_scaler
        X_scaled = scaler.transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

    return X_scaled, scaler


def prepare_split(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple:
    """Prepare X_train, y_train, X_test, y_test with proper scaling.

    Fit scaler on train, apply to test (enforces fit-on-train-only).

    Returns:
        X_train, y_train, X_test, y_test, scaler
    """
    X_train, scaler = engineer_features(df_train, fit_scaler=None)
    X_test, _ = engineer_features(df_test, fit_scaler=scaler)

    y_train = df_train['churned'].values
    y_test = df_test['churned'].values

    print(f"Train: {len(X_train)} samples, churn rate={y_train.mean():.3f}")
    print(f"Test: {len(X_test)} samples, churn rate={y_test.mean():.3f}")

    return X_train, y_train, X_test, y_test, scaler


def get_baseline_prediction(y_test: np.ndarray) -> np.ndarray:
    """Majority class baseline."""
    churn_rate = y_test.mean()
    return np.full_like(y_test, fill_value=1.0 if churn_rate > 0.5 else 0.0, dtype=float)
