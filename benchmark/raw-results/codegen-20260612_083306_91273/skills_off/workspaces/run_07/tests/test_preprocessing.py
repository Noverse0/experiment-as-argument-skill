"""Tests for data preprocessing pipeline."""
import pandas as pd
import numpy as np
import pytest
from src.preprocessing import (
    load_data,
    remove_duplicates,
    extract_features,
    time_based_split,
    scale_features,
    validate_no_leak,
)


@pytest.fixture
def sample_data():
    """Create sample churn dataset for testing."""
    df = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'signup_date': [
            '2023-01-01', '2023-03-01', '2023-06-01',
            '2023-09-01', '2023-12-01'
        ],
        'tenure_months': [24, 20, 12, 6, 3],
        'monthly_spend': [100.0, 50.0, 75.0, 25.0, 10.0],
        'support_tickets': [1, 2, 0, 3, 5],
        'account_status': ['active', 'closed', 'active', 'closed', 'closed'],
        'churned': [0, 1, 0, 1, 1],
    })
    return df


def test_load_data(tmp_path):
    """Test data loading."""
    csv_file = tmp_path / "test_data.csv"
    df = pd.DataFrame({
        'customer_id': [1],
        'signup_date': ['2023-01-01'],
        'tenure_months': [10],
        'monthly_spend': [50.0],
        'support_tickets': [1],
        'account_status': ['active'],
        'churned': [0],
    })
    df.to_csv(csv_file, index=False)

    loaded = load_data(str(csv_file))
    assert len(loaded) == 1
    assert loaded['customer_id'].iloc[0] == 1


def test_remove_duplicates(sample_data):
    """Test duplicate removal."""
    # Add a duplicate row
    dup_row = sample_data.iloc[0].copy()
    df_with_dup = pd.concat([sample_data, pd.DataFrame([dup_row])], ignore_index=True)

    assert len(df_with_dup) == 6
    cleaned = remove_duplicates(df_with_dup)
    assert len(cleaned) == 5


def test_extract_features(sample_data):
    """Test feature extraction and leak detection."""
    # Verify account_status leak
    leaked = validate_no_leak(sample_data)
    assert leaked is True, "Should detect leak in account_status"

    # Extract features (excludes account_status and customer_id)
    features, target = extract_features(sample_data)

    assert set(features.columns) == {'tenure_months', 'monthly_spend', 'support_tickets', 'days_since_signup'}
    assert len(target) == 5
    assert target.sum() == 3  # Three churned customers


def test_time_based_split(sample_data):
    """Test time-based train/test split."""
    features, target = extract_features(sample_data)

    # Keep days_since_signup before split for checking temporal ordering
    days_since_signup = features['days_since_signup'].values.copy()

    X_train, X_test, y_train, y_test = time_based_split(features, target, test_percentile=60.0)

    # Check split sizes
    assert len(X_train) + len(X_test) == len(features)
    assert len(y_train) + len(y_test) == len(target)

    # Check no overlap
    assert set(X_train.index) & set(X_test.index) == set()

    # Check temporal ordering (train indices should have earlier dates than test)
    train_days = days_since_signup[X_train.index].max()
    test_days = days_since_signup[X_test.index].min()
    assert train_days <= test_days, "Train/test temporal ordering violated"


def test_scale_features(sample_data):
    """Test feature scaling (fit on train, apply to test)."""
    features, target = extract_features(sample_data)
    X_train, X_test, y_train, y_test = time_based_split(features, target)

    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Check that scaled features are centered around 0 (within tolerance for small samples)
    assert np.abs(X_train_scaled.mean(axis=0).mean()) < 1.0
    # Standard deviation may be > 1.0 for small samples due to Bessel's correction
    assert np.abs(X_train_scaled.std(axis=0).mean() - 1.0) < 1.0

    # Check no NaN
    assert not X_train_scaled.isna().any().any()
    assert not X_test_scaled.isna().any().any()


def test_no_feature_leakage(sample_data):
    """Test that account_status is not included in features."""
    features, target = extract_features(sample_data)
    assert 'account_status' not in features.columns
    assert 'customer_id' not in features.columns


def test_target_integrity(sample_data):
    """Test that target is correctly extracted."""
    features, target = extract_features(sample_data)
    assert target.min() == 0
    assert target.max() == 1
    assert target.dtype in [np.int32, np.int64, int]
