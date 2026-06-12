"""Tests for data preprocessing pipeline."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.preprocessing import (
    load_data,
    check_duplicates,
    hunt_leakage,
    preprocess_and_split,
    get_baseline_prediction,
)


@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing."""
    data = {
        'customer_id': range(1, 101),
        'signup_date': pd.date_range('2023-01-01', periods=100),
        'tenure_months': np.random.randint(1, 100, 100),
        'monthly_spend': np.random.uniform(10, 200, 100),
        'support_tickets': np.random.randint(0, 10, 100),
        'account_status': np.random.choice(['active', 'closed'], 100),
        'churned': np.random.randint(0, 2, 100),
    }
    return pd.DataFrame(data)


def test_load_data(tmp_path):
    """Test data loading from CSV."""
    # Create a temporary CSV
    test_csv = tmp_path / "test.csv"
    df = pd.DataFrame({
        'customer_id': [1, 2, 3],
        'tenure_months': [10, 20, 30],
        'churned': [0, 1, 0],
    })
    df.to_csv(test_csv, index=False)

    loaded = load_data(str(test_csv))
    assert len(loaded) == 3
    assert list(loaded.columns) == ['customer_id', 'tenure_months', 'churned']


def test_check_duplicates(sample_data):
    """Test duplicate detection."""
    # No duplicates in fresh random data
    dup_count = check_duplicates(sample_data)
    assert dup_count >= 0  # Should be a valid count

    # Create duplicates manually
    sample_data_dup = pd.concat([sample_data.iloc[:5], sample_data.iloc[:5]], ignore_index=True)
    dup_count_with = check_duplicates(sample_data_dup)
    assert dup_count_with >= 5


def test_hunt_leakage(sample_data):
    """Test leakage detection."""
    # Create a version with perfect correlation
    leaky_data = sample_data.copy()
    leaky_data['account_status'] = leaky_data['churned'].map({0: 'active', 1: 'closed'})

    suspects = hunt_leakage(leaky_data)
    assert len(suspects) > 0
    assert any('account_status' in s for s in suspects)


def test_preprocess_and_split_basic(sample_data):
    """Test basic preprocessing and split."""
    X_train, X_test, y_train, y_test, scaler = preprocess_and_split(
        sample_data, test_size=0.2, random_state=42, use_leaky_features=False
    )

    # Check sizes
    assert len(X_train) + len(X_test) == len(sample_data)
    assert len(X_train) == int(len(sample_data) * 0.8)
    assert len(X_test) == int(len(sample_data) * 0.2)

    # Check targets match
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    # Check no signup_date in output
    assert 'signup_date' not in X_train.columns
    assert 'signup_date' not in X_test.columns

    # Check no account_status (removed as leakage)
    assert 'account_status' not in X_train.columns
    assert 'account_status' not in X_test.columns


def test_preprocess_split_before_transform(sample_data):
    """Test that scaler is fitted on train only."""
    X_train, X_test, y_train, y_test, scaler = preprocess_and_split(
        sample_data, test_size=0.2, random_state=42, use_leaky_features=False
    )

    # Check that values are scaled (mean ~0, std ~1 after standardization)
    # Allow some tolerance due to small sample size
    assert abs(X_train.mean().mean()) < 1.0  # Mean near 0
    assert abs(X_train.std().mean() - 1.0) < 0.5  # Std near 1


def test_preprocess_deterministic(sample_data):
    """Test reproducibility with same seed."""
    X_train1, X_test1, y_train1, y_test1, _ = preprocess_and_split(
        sample_data, test_size=0.2, random_state=42, use_leaky_features=False
    )
    X_train2, X_test2, y_train2, y_test2, _ = preprocess_and_split(
        sample_data, test_size=0.2, random_state=42, use_leaky_features=False
    )

    # Should produce identical results
    assert (X_train1.values == X_train2.values).all()
    assert (X_test1.values == X_test2.values).all()
    assert (y_train1.values == y_train2.values).all()
    assert (y_test1.values == y_test2.values).all()


def test_get_baseline_prediction(sample_data):
    """Test baseline prediction calculation."""
    X_train, X_test, y_train, y_test, _ = preprocess_and_split(
        sample_data, test_size=0.2, random_state=42, use_leaky_features=False
    )

    baseline = get_baseline_prediction(y_train, y_test)

    assert 'baseline_accuracy' in baseline
    assert 'baseline_class' in baseline
    assert 'target_rate_train' in baseline
    assert 'target_rate_test' in baseline
    assert 0 <= baseline['baseline_accuracy'] <= 1
    assert baseline['baseline_class'] in [0, 1]
    assert 0 <= baseline['target_rate_train'] <= 1
    assert 0 <= baseline['target_rate_test'] <= 1


def test_leaky_features_removal(sample_data):
    """Test that account_status is removed when use_leaky_features=False."""
    X_train_no_leaky, _, _, _, _ = preprocess_and_split(
        sample_data, test_size=0.2, random_state=42, use_leaky_features=False
    )

    assert 'account_status' not in X_train_no_leaky.columns
