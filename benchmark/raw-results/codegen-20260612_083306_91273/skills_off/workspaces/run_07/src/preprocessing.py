"""Data loading, cleaning, and splitting for churn prediction."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    """Load CSV and validate basic structure."""
    df = pd.read_csv(path)
    assert set(df.columns) == {
        'customer_id', 'signup_date', 'tenure_months',
        'monthly_spend', 'support_tickets', 'account_status', 'churned'
    }, f"Unexpected columns: {df.columns.tolist()}"
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows (respecting dedup boundary before split)."""
    n_before = len(df)
    df = df.drop_duplicates(keep='first').reset_index(drop=True)
    n_after = len(df)
    print(f"Removed {n_before - n_after} duplicate rows")
    return df


def validate_no_leak(df: pd.DataFrame):
    """Verify target leakage: account_status must be derived from churned."""
    # This is expected leakage; we detect it and exclude the feature
    leaked = (df['account_status'] == 'closed') == (df['churned'] == 1)
    leak_count = leaked.sum()
    total = len(df)
    if leak_count / total > 0.99:
        print(f"LEAK DETECTED: account_status perfectly derives churned ({leak_count}/{total}). Excluding feature.")
        return True
    return False


def extract_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract features and target.

    Exclude:
    - account_status: LEAK (derived from target)
    - customer_id: no predictive power

    Include:
    - tenure_months: legitimate feature
    - monthly_spend: legitimate feature
    - support_tickets: legitimate feature
    - signup_date: convert to days-since-start for temporal awareness
    """
    # Temporal feature: convert signup_date to ordinal days
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['days_since_signup'] = (df['signup_date'] - df['signup_date'].min()).dt.days

    features = df[['tenure_months', 'monthly_spend', 'support_tickets', 'days_since_signup']].copy()
    target = df['churned'].copy()

    return features, target


def time_based_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_percentile: float = 75.0
) -> tuple:
    """
    Split by days_since_signup percentile (temporal split respects time structure).

    Returns: X_train, X_test, y_train, y_test
    """
    days_since_signup = features['days_since_signup'].values
    threshold = np.percentile(days_since_signup, test_percentile)

    train_mask = days_since_signup <= threshold
    test_mask = ~train_mask

    # Sanity check: no data leakage across split
    assert not (train_mask & test_mask).any(), "Train/test overlap detected"
    assert (train_mask | test_mask).all(), "Not all rows assigned"

    X_train = features[train_mask].drop(columns=['days_since_signup'])
    X_test = features[test_mask].drop(columns=['days_since_signup'])
    y_train = target[train_mask]
    y_test = target[test_mask]

    print(f"Train: {len(X_train)} rows ({y_train.mean():.1%} churn)")
    print(f"Test:  {len(X_test)} rows ({y_test.mean():.1%} churn)")

    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Scale numeric features on train set, apply to test.
    Fit-transform on train, transform-only on test (no leakage).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame for consistency
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    return X_train_scaled, X_test_scaled


def prepare_data(path: str, test_percentile: float = 75.0):
    """
    Full pipeline: load, deduplicate, check leaks, extract, split, scale.
    """
    df = load_data(path)
    df = remove_duplicates(df)
    validate_no_leak(df)

    features, target = extract_features(df)
    X_train, X_test, y_train, y_test = time_based_split(
        features, target, test_percentile=test_percentile
    )
    X_train, X_test = scale_features(X_train, X_test)

    return X_train, X_test, y_train, y_test
