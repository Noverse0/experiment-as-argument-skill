"""Data loading, deduplication, and time-based splitting.

Rigor discipline:
- Split before transform: preprocessing (scaling) happens only after split, fitted on train only
- Hunt leakage: days_since_last_login is a target leak (churned=1 -> high days_since_last_login by definition)
- Dedup before split: exact duplicates must be removed before any train/test split
- Time-based split: use signup_date to split (train on earlier, test on later) to respect temporal structure
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load CSV and remove exact duplicates before any split."""
    df = pd.read_csv(csv_path)
    n_before = len(df)

    # Remove exact duplicates (keeping first occurrence)
    df = df.drop_duplicates(subset=None, keep='first')

    n_after = len(df)
    n_removed = n_before - n_after

    print(f"Loaded {csv_path}: {n_before} rows")
    print(f"Removed {n_removed} exact duplicate rows")

    return df


def extract_features_and_target(
    df: pd.DataFrame,
    drop_leaked_features: bool = True
) -> tuple:
    """
    Extract features and target, removing leaked columns.

    Args:
        df: DataFrame with columns [customer_id, signup_date, tenure_months, monthly_spend,
                                    support_tickets, days_since_last_login, churned]
        drop_leaked_features: if True, drop days_since_last_login (target leak)

    Returns:
        (X_features, y_target, dates): feature names, target series, signup dates for time-based split
    """
    # Feature columns: exclude customer_id (not a feature), signup_date (used for split),
    # and days_since_last_login (target leak) if requested
    feature_cols = ['tenure_months', 'monthly_spend', 'support_tickets']

    if not drop_leaked_features:
        # Include the leaked feature for sanity check (should show suspiciously high performance)
        feature_cols.append('days_since_last_login')

    X = df[feature_cols].copy()
    y = df['churned'].copy()
    dates = pd.to_datetime(df['signup_date']).copy()

    return X, y, dates, feature_cols


def time_based_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    train_fraction: float = 0.6
) -> dict:
    """
    Split data using signup_date (temporal split, not random).

    Train on earlier customers, test on later ones.
    This respects temporal structure and avoids near-duplicate rows straddling the split.

    Args:
        X: features
        y: target
        dates: signup dates
        train_fraction: what fraction of the timeline becomes training

    Returns:
        dict with keys: X_train, X_test, y_train, y_test, train_idx, test_idx
    """
    # Sort by date
    sorted_idx = dates.argsort()

    # Split at the train_fraction point in the timeline
    split_point = int(len(sorted_idx) * train_fraction)
    train_idx_sorted = sorted_idx[:split_point]
    test_idx_sorted = sorted_idx[split_point:]

    return {
        'X_train': X.iloc[train_idx_sorted].reset_index(drop=True),
        'X_test': X.iloc[test_idx_sorted].reset_index(drop=True),
        'y_train': y.iloc[train_idx_sorted].reset_index(drop=True),
        'y_test': y.iloc[test_idx_sorted].reset_index(drop=True),
        'train_idx': train_idx_sorted,
        'test_idx': test_idx_sorted,
        'train_size': len(train_idx_sorted),
        'test_size': len(test_idx_sorted),
    }


def preprocess_after_split(split_data: dict) -> dict:
    """
    Fit StandardScaler on train set only, apply to both train and test.

    This prevents preprocessing leakage: the test set should not influence scaling parameters.
    """
    X_train = split_data['X_train']
    X_test = split_data['X_test']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    split_data['X_train_scaled'] = X_train_scaled
    split_data['X_test_scaled'] = X_test_scaled
    split_data['scaler'] = scaler

    return split_data


def get_baseline_prediction(y_train: pd.Series) -> int:
    """Return the majority class (baseline floor)."""
    return int(y_train.value_counts().idxmax())
