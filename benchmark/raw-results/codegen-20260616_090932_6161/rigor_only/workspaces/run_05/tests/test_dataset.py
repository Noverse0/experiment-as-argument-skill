"""Tests for dataset loading and preprocessing."""
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys

# Add src to path for imports.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import (
    load_and_deduplicate, engineer_features, get_train_test_split, report_class_balance
)


@pytest.fixture
def sample_churn_csv(tmp_path):
    """Create a small sample churn CSV for testing."""
    df = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'signup_date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'tenure_months': [12, 24, 6, 36, 18],
        'monthly_spend': [50.0, 100.0, 25.0, 150.0, 75.0],
        'support_tickets': [2, 1, 5, 0, 3],
        'days_since_last_login': [5, 10, 100, 2, 50],
        'churned': [0, 0, 1, 0, 1],
    })
    csv_path = tmp_path / 'test_churn.csv'
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_load_and_deduplicate(sample_churn_csv):
    """Test that load_and_deduplicate works and reports deduplication."""
    df = pd.read_csv(sample_churn_csv)
    assert len(df) == 5

    # Add exact duplicate.
    dup = df.iloc[0:1].copy()
    df_with_dup = pd.concat([df, dup], ignore_index=True)
    csv_with_dup = sample_churn_csv.replace('.csv', '_with_dup.csv')
    df_with_dup.to_csv(csv_with_dup, index=False)

    # Load and deduplicate.
    df_dedup = load_and_deduplicate(csv_with_dup)
    assert len(df_dedup) == 5  # Duplicate removed.


def test_engineer_features(sample_churn_csv):
    """Test that engineer_features excludes days_since_last_login and creates days_since_signup."""
    df = pd.read_csv(sample_churn_csv)
    X, y, feature_cols = engineer_features(df)

    # Check that days_since_last_login is not in features.
    assert 'days_since_last_login' not in feature_cols
    assert 'days_since_signup' in feature_cols

    # Check shape.
    assert len(X) == len(df)
    assert len(y) == len(df)
    assert len(feature_cols) == 4

    # Check no NaNs.
    assert X.isnull().sum().sum() == 0
    assert y.isnull().sum() == 0

    # Check target is binary.
    assert set(y.unique()) <= {0, 1}


def test_get_train_test_split(sample_churn_csv):
    """Test that train/test split is stratified and respects random state."""
    df = pd.read_csv(sample_churn_csv)
    X, y, _ = engineer_features(df)

    X_train1, X_test1, y_train1, y_test1 = get_train_test_split(X, y, test_size=0.3, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = get_train_test_split(X, y, test_size=0.3, random_state=42)

    # Same random state should give same split.
    assert len(X_train1) == len(X_train2)
    assert len(X_test1) == len(X_test2)
    pd.testing.assert_frame_equal(X_train1, X_train2)

    # Different random state should (likely) give different split.
    X_train3, X_test3, y_train3, y_test3 = get_train_test_split(X, y, test_size=0.3, random_state=99)
    assert len(X_train3) == len(X_train1)
    # Note: with small data, splits might coincidentally match, so we just check length.

    # Check stratification (classes should be roughly balanced in train and test).
    train_churn_rate = y_train1.sum() / len(y_train1)
    test_churn_rate = y_test1.sum() / len(y_test1)
    overall_churn_rate = y.sum() / len(y)
    # Stratification should keep proportions similar (within tolerance for small data).
    assert abs(train_churn_rate - overall_churn_rate) < 0.5
    assert abs(test_churn_rate - overall_churn_rate) < 0.5


def test_report_class_balance(sample_churn_csv):
    """Test that class balance reporting works."""
    df = pd.read_csv(sample_churn_csv)
    X, y, _ = engineer_features(df)

    rate = report_class_balance(y, "Test")
    assert 0 <= rate <= 100
    assert isinstance(rate, float)
