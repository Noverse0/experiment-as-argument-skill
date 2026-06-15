"""Tests for data loading and preprocessing."""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os

from src.data import (
    load_and_deduplicate,
    select_features,
    compute_class_balance,
    train_test_split_no_leakage,
    scale_features,
)


@pytest.fixture
def sample_csv():
    """Create a sample CSV file for testing."""
    df = pd.DataFrame({
        'customer_id': [1, 2, 3, 3],  # 3 appears twice (duplicate)
        'signup_date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-03'],
        'tenure_months': [12, 24, 36, 36],
        'monthly_spend': [100.0, 200.0, 150.0, 150.0],
        'support_tickets': [1, 2, 3, 3],
        'days_since_last_login': [5, 10, 15, 15],
        'churned': [0, 1, 1, 1],
    })

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False)
        tmp_path = f.name

    yield tmp_path

    os.unlink(tmp_path)


def test_load_and_deduplicate(sample_csv):
    """Test deduplication removes exact duplicates."""
    df, n_removed = load_and_deduplicate(sample_csv)
    assert n_removed == 1
    assert len(df) == 3
    assert df[df['customer_id'] == 3].shape[0] == 1


def test_select_features():
    """Test feature selection excludes correct columns."""
    df = pd.DataFrame({
        'customer_id': [1, 2],
        'signup_date': ['2023-01-01', '2023-01-02'],
        'tenure_months': [12, 24],
        'monthly_spend': [100.0, 200.0],
        'support_tickets': [1, 2],
        'days_since_last_login': [5, 10],
        'churned': [0, 1],
    })
    X, reason = select_features(df)

    assert list(X.columns) == ['tenure_months', 'monthly_spend', 'support_tickets']
    assert 'customer_id' not in X.columns
    assert 'signup_date' not in X.columns
    assert 'days_since_last_login' not in X.columns
    assert 'target leak' in reason


def test_compute_class_balance():
    """Test class balance computation."""
    y = pd.Series([0, 0, 0, 1])
    balance = compute_class_balance(y)

    assert balance['churn_rate'] == pytest.approx(0.25)
    assert balance['n_churned'] == 1
    assert balance['n_retained'] == 3
    assert balance['n_total'] == 4


def test_train_test_split_no_leakage():
    """Test split preserves class balance."""
    X = pd.DataFrame({
        'feat1': range(100),
        'feat2': range(100, 200),
    })
    y = pd.Series([0] * 70 + [1] * 30)

    X_train, X_test, y_train, y_test = train_test_split_no_leakage(
        X, y, test_size=0.3, random_state=42
    )

    assert len(X_train) == 70
    assert len(X_test) == 30
    # Stratified split should roughly preserve proportions
    assert 0.2 < y_train.mean() < 0.4  # Should be close to 30%


def test_scale_features():
    """Test scaling fits on train and applies to test."""
    X_train = pd.DataFrame({
        'feat1': [0, 1, 2, 3, 4],
        'feat2': [10, 20, 30, 40, 50],
    })
    X_test = pd.DataFrame({
        'feat1': [2.5],
        'feat2': [30.0],
    })

    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Check outputs are numpy arrays
    assert isinstance(X_train_scaled, np.ndarray)
    assert isinstance(X_test_scaled, np.ndarray)

    # Check shapes
    assert X_train_scaled.shape == (5, 2)
    assert X_test_scaled.shape == (1, 2)

    # Check values are scaled (mean ~0, std ~1 for train)
    assert np.abs(X_train_scaled.mean()) < 0.1
    assert np.abs(X_train_scaled.std() - 1.0) < 0.1
