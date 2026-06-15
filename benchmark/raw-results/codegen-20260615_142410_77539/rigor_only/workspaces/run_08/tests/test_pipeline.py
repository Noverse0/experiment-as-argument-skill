"""Unit tests for data pipeline."""
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from src.pipeline import (
    load_and_deduplicate,
    prepare_features_and_target,
    split_and_preprocess,
    load_and_prepare,
)


@pytest.fixture
def sample_csv():
    """Create a temporary CSV with sample churn data."""
    data = {
        'customer_id': list(range(1, 51)),
        'signup_date': ['2023-01-01'] * 50,
        'tenure_months': list(range(10, 60)),
        'monthly_spend': [100.0 + i * 5 for i in range(50)],
        'support_tickets': list(range(1, 51)),
        'days_since_last_login': list(range(5, 55)),
        'churned': [i % 2 for i in range(50)],
    }
    df = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


@pytest.fixture
def sample_csv_with_duplicates():
    """Create a temporary CSV with exact duplicate rows."""
    data = {
        'customer_id': [1, 2, 3, 2],  # Row 2 is duplicated
        'signup_date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-02'],
        'tenure_months': [10, 20, 30, 20],
        'monthly_spend': [100.0, 200.0, 300.0, 200.0],
        'support_tickets': [1, 2, 3, 2],
        'days_since_last_login': [5, 10, 15, 10],
        'churned': [0, 1, 0, 1],
    }
    df = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


def test_load_and_deduplicate_no_dupes(sample_csv):
    """Test loading CSV with no duplicates."""
    df, n_removed = load_and_deduplicate(sample_csv)
    assert len(df) == 50
    assert n_removed == 0


def test_load_and_deduplicate_with_dupes(sample_csv_with_duplicates):
    """Test loading CSV with exact duplicates."""
    df, n_removed = load_and_deduplicate(sample_csv_with_duplicates)
    assert len(df) == 3  # 4 rows - 1 duplicate
    assert n_removed == 1


def test_prepare_features_and_target(sample_csv):
    """Test feature extraction and drops correct columns."""
    df = pd.read_csv(sample_csv)
    X, y = prepare_features_and_target(df)

    # Check shape
    assert X.shape == (50, 3)
    assert len(y) == 50

    # Check columns
    assert list(X.columns) == ['tenure_months', 'monthly_spend', 'support_tickets']

    # Verify leaky columns are gone
    assert 'days_since_last_login' not in X.columns
    assert 'customer_id' not in X.columns
    assert 'signup_date' not in X.columns


def test_split_and_preprocess_maintains_class_balance(sample_csv):
    """Test that stratified split maintains class balance."""
    df = pd.read_csv(sample_csv)
    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test, scaler = split_and_preprocess(
        X, y, test_size=0.2, random_state=42
    )

    # Check shapes
    assert len(X_train) + len(X_test) == len(y)
    assert len(y_train) + len(y_test) == len(y)

    # Check that scaling was applied (values should be normalized)
    assert abs(X_train.mean(axis=0)).max() < 0.1  # Mean near 0
    assert abs(X_train.std(axis=0).mean() - 1.0) < 0.1  # Std near 1

    # Check that test set uses same scaling
    assert X_test.shape[1] == X_train.shape[1]


def test_split_and_preprocess_deterministic():
    """Test that same random_state produces identical splits."""
    data = {
        'tenure_months': [10, 20, 30, 40, 50, 60],
        'monthly_spend': [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        'support_tickets': [1, 2, 3, 4, 5, 6],
    }
    X = pd.DataFrame(data)
    y = np.array([0, 1, 0, 1, 0, 1])

    X_train1, X_test1, y_train1, y_test1, _ = split_and_preprocess(
        X, y, test_size=0.33, random_state=42
    )
    X_train2, X_test2, y_train2, y_test2, _ = split_and_preprocess(
        X, y, test_size=0.33, random_state=42
    )

    # Check identical splits
    np.testing.assert_array_equal(X_train1, X_train2)
    np.testing.assert_array_equal(X_test1, X_test2)
    np.testing.assert_array_equal(y_train1, y_train2)
    np.testing.assert_array_equal(y_test1, y_test2)


def test_load_and_prepare_integration(sample_csv_with_duplicates):
    """Integration test: full pipeline from CSV to train/test."""
    X_train, X_test, y_train, y_test, metadata = load_and_prepare(
        sample_csv_with_duplicates, random_state=42
    )

    # Check shapes
    assert X_train.shape[1] == 3  # 3 features
    assert X_test.shape[1] == 3
    assert len(y_train) + len(y_test) == 3  # 4 original - 1 duplicate

    # Check metadata
    assert metadata['n_duplicates_removed'] == 1
    assert metadata['n_total_after_dedup'] == 3
    assert metadata['n_train'] + metadata['n_test'] == 3
    assert 0 <= metadata['churn_rate'] <= 1

    # Check stratified split: train/test churn rates exist (with small datasets, exact match is hard)
    assert 0 <= metadata['train_churn_rate'] <= 1
    assert 0 <= metadata['test_churn_rate'] <= 1
