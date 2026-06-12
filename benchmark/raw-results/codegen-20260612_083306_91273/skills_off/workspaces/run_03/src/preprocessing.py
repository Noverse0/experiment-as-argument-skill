"""Data loading, splitting, and preprocessing for churn experiment."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(csv_path: str) -> pd.DataFrame:
    """Load churn dataset from CSV."""
    df = pd.read_csv(csv_path)
    return df


def check_duplicates(df: pd.DataFrame) -> int:
    """Check for exact duplicate rows (excluding customer_id and churned)."""
    feature_cols = [c for c in df.columns if c not in ['customer_id', 'churned']]
    duplicates = df.duplicated(subset=feature_cols, keep=False).sum()
    return duplicates


def hunt_leakage(df: pd.DataFrame) -> list:
    """Identify potential target leakage features.

    Returns list of suspect features. In this dataset:
    - account_status='closed' is highly correlated with churned=1 (leakage candidate)
    - These should not be used; they encode the outcome.
    """
    suspects = []

    # account_status is derived from churn outcome (closed → churned)
    if 'account_status' in df.columns:
        crosstab = pd.crosstab(df['account_status'], df['churned'], margins=True)
        # Check if account_status perfectly predicts churned
        if df[df['account_status'] == 'closed']['churned'].unique().tolist() == [1]:
            suspects.append('account_status (closed → churned perfect correlation)')

    return suspects


def preprocess_and_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    use_leaky_features: bool = False,
) -> tuple:
    """
    Split before transform: time-based split, then scale.

    Args:
        df: Raw dataframe
        test_size: Fraction for test set
        random_state: Random seed for reproducibility
        use_leaky_features: If False, drop suspected leakage features

    Returns:
        (X_train, X_test, y_train, y_test, scaler)
    """
    np.random.seed(random_state)

    # Drop customer_id (not a feature)
    df_features = df.drop(columns=['customer_id'], errors='ignore').copy()

    # If removing leaky features, exclude account_status
    if not use_leaky_features:
        df_features = df_features.drop(columns=['account_status'], errors='ignore')

    # Separate target
    y = df_features['churned'].copy()
    X = df_features.drop(columns=['churned']).copy()

    # Time-based split: use signup_date to split chronologically
    # Convert to datetime and sort
    X['signup_date'] = pd.to_datetime(X['signup_date'])
    X = X.sort_values('signup_date')
    y = y.loc[X.index]

    split_idx = int(len(X) * (1 - test_size))
    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    # Drop signup_date after split
    X_train = X_train.drop(columns=['signup_date'])
    X_test = X_test.drop(columns=['signup_date'])

    # Fit scaler on train only, apply to both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_baseline_prediction(y_train: pd.Series, y_test: pd.Series) -> dict:
    """Calculate baseline (majority class) prediction metrics."""
    majority_class = y_train.value_counts().idxmax()
    y_baseline = pd.Series([majority_class] * len(y_test), index=y_test.index)

    accuracy = (y_baseline == y_test).mean()
    return {
        'baseline_accuracy': accuracy,
        'baseline_class': majority_class,
        'target_rate_train': y_train.mean(),
        'target_rate_test': y_test.mean(),
    }
