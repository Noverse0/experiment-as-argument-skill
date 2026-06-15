"""Tests for data loading and preprocessing."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data import (
    load_and_clean,
    extract_features_and_target,
    time_based_split,
    preprocess_after_split,
    get_baseline_prediction,
)


@pytest.fixture
def sample_data():
    """Create a small test dataset."""
    df = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'signup_date': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01'],
        'tenure_months': [10, 20, 30, 40, 50],
        'monthly_spend': [100.0, 200.0, 300.0, 400.0, 500.0],
        'support_tickets': [1, 2, 3, 2, 1],
        'days_since_last_login': [5, 10, 15, 20, 25],
        'churned': [0, 1, 0, 1, 1],
    })
    return df


def test_extract_features_no_leak(sample_data):
    """Test feature extraction without leaked column."""
    X, y, dates, cols = extract_features_and_target(sample_data, drop_leaked_features=True)

    assert len(X) == 5
    assert list(X.columns) == ['tenure_months', 'monthly_spend', 'support_tickets']
    assert len(y) == 5
    assert y.tolist() == [0, 1, 0, 1, 1]


def test_extract_features_with_leak(sample_data):
    """Test feature extraction including leaked column."""
    X, y, dates, cols = extract_features_and_target(sample_data, drop_leaked_features=False)

    assert 'days_since_last_login' in X.columns
    assert 'tenure_months' in X.columns


def test_time_based_split(sample_data):
    """Test that time-based split respects signup_date ordering."""
    X, y, dates, _ = extract_features_and_target(sample_data, drop_leaked_features=True)
    split = time_based_split(X, y, dates, train_fraction=0.6)

    assert split['train_size'] == 3
    assert split['test_size'] == 2
    assert len(split['X_train']) == 3
    assert len(split['X_test']) == 2

    # Train should have earlier dates
    train_dates = dates.iloc[split['train_idx']]
    test_dates = dates.iloc[split['test_idx']]
    assert train_dates.max() <= test_dates.min()


def test_preprocess_after_split(sample_data):
    """Test that scaler is fit on train only."""
    X, y, dates, _ = extract_features_and_target(sample_data, drop_leaked_features=True)
    split = time_based_split(X, y, dates, train_fraction=0.6)
    split = preprocess_after_split(split)

    # Check that scaled arrays exist
    assert split['X_train_scaled'] is not None
    assert split['X_test_scaled'] is not None

    # Check that means are close to zero (scaler fit on train)
    assert abs(split['X_train_scaled'].mean(axis=0).max()) < 0.1


def test_baseline_prediction(sample_data):
    """Test majority class selection."""
    X, y, dates, _ = extract_features_and_target(sample_data, drop_leaked_features=True)
    baseline = get_baseline_prediction(y)

    # In sample_data, churned has [0,1,0,1,1] so majority is 1
    assert baseline == 1


def test_load_and_clean_removes_duplicates(tmp_path):
    """Test that load_and_clean removes exact duplicates."""
    # Create a CSV with duplicates
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({
        'customer_id': [1, 2, 3, 1],  # row 0 and 3 are duplicates
        'signup_date': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-01-01'],
        'tenure_months': [10, 20, 30, 10],
        'monthly_spend': [100.0, 200.0, 300.0, 100.0],
        'support_tickets': [1, 2, 3, 1],
        'days_since_last_login': [5, 10, 15, 5],
        'churned': [0, 1, 0, 0],
    })
    df.to_csv(csv_file, index=False)

    # Load and check dedup
    loaded = load_and_clean(str(csv_file))
    assert len(loaded) == 3  # duplicates removed
