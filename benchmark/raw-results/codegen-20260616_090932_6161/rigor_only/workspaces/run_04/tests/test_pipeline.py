"""Tests for data pipeline."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.pipeline import (
    load_data,
    deduplicate,
    preprocess,
    time_based_split,
    scale_features,
    get_churn_rate,
)


@pytest.fixture
def sample_churn_data():
    """Create a small sample dataset for testing."""
    return pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'signup_date': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01'],
        'tenure_months': [12, 24, 36, 48, 60],
        'monthly_spend': [100.0, 200.0, 150.0, 250.0, 180.0],
        'support_tickets': [1, 2, 0, 1, 3],
        'days_since_last_login': [5, 10, 3, 20, 7],
        'churned': [0, 1, 0, 1, 0],
    })


def test_load_data():
    """Test loading the actual CSV."""
    df = load_data('churn.csv')
    assert len(df) > 0
    assert 'churned' in df.columns
    assert 'tenure_months' in df.columns


def test_deduplicate():
    """Test duplicate row removal."""
    df = pd.DataFrame({
        'a': [1, 2, 2, 3],
        'b': [4, 5, 5, 6],
    })
    result = deduplicate(df)
    assert len(result) == 3  # One duplicate removed


def test_preprocess_removes_leak(sample_churn_data):
    """Test that days_since_last_login is removed."""
    result = preprocess(sample_churn_data)
    assert 'days_since_last_login' not in result.columns
    assert 'customer_id' not in result.columns
    assert 'signup_date' not in result.columns
    assert 'churned' in result.columns
    assert 'days_since_signup' in result.columns


def test_time_based_split(sample_churn_data):
    """Test time-based split maintains order and correct sizes."""
    df = preprocess(sample_churn_data)
    X_train, X_test, y_train, y_test = time_based_split(df, train_frac=0.6, seed=42)

    assert len(X_train) + len(X_test) == len(df)
    assert len(X_train) == int(len(df) * 0.6)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_scale_features(sample_churn_data):
    """Test that scaling fits on train only."""
    df = preprocess(sample_churn_data)
    X_train, X_test, _, _ = time_based_split(df, train_frac=0.6, seed=42)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Check shapes.
    assert X_train_scaled.shape[0] == len(X_train)
    assert X_test_scaled.shape[0] == len(X_test)

    # Check scaling was applied (approximate mean 0, std 1).
    assert np.abs(X_train_scaled.mean()) < 0.1
    assert np.abs(X_train_scaled.std() - 1.0) < 0.2


def test_churn_rate(sample_churn_data):
    """Test churn rate calculation."""
    rate = get_churn_rate(sample_churn_data['churned'])
    assert rate == 0.4  # 2 out of 5


def test_preprocessing_preserves_target(sample_churn_data):
    """Test that target is preserved during preprocessing."""
    result = preprocess(sample_churn_data)
    assert result['churned'].tolist() == sample_churn_data['churned'].tolist()


def test_full_pipeline():
    """Integration test: load, deduplicate, preprocess, split, scale."""
    df = load_data('churn.csv')
    df = deduplicate(df)
    df = preprocess(df)
    X_train, X_test, y_train, y_test = time_based_split(df, train_frac=0.7, seed=42)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert X_train_scaled.shape[1] == X_test_scaled.shape[1]
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
