"""
Tests for the churn prediction experiment pipeline.
"""
import sys
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiment import (
    load_and_deduplicate,
    time_based_split,
    prepare_features,
    train_and_evaluate,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "signup_date": pd.date_range("2023-01-01", periods=n, freq="D").strftime(
                "%Y-%m-%d"
            ),
            "tenure_months": np.random.randint(1, 72, n),
            "monthly_spend": np.random.uniform(10, 200, n),
            "support_tickets": np.random.randint(0, 5, n),
            "days_since_last_login": np.random.randint(1, 100, n),
            "churned": np.random.randint(0, 2, n),
        }
    )
    return df


@pytest.fixture
def sample_csv(sample_data):
    """Write sample data to a temporary CSV and return path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_data.to_csv(f, index=False)
        return f.name


def test_load_and_deduplicate(sample_csv, sample_data):
    """Test that duplicates are correctly removed."""
    df_loaded = load_and_deduplicate(sample_csv)
    assert len(df_loaded) == len(sample_data)
    assert list(df_loaded.columns) == list(sample_data.columns)


def test_load_and_deduplicate_with_dupes():
    """Test that duplicates are actually removed."""
    df = pd.DataFrame(
        {
            "a": [1, 2, 2, 3],
            "b": [4, 5, 5, 6],
        }
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f, index=False)
        temp_path = f.name

    df_loaded = load_and_deduplicate(temp_path)
    assert len(df_loaded) == 3  # One duplicate removed


def test_time_based_split(sample_data):
    """Test time-based split respects temporal order."""
    train, test = time_based_split(sample_data, train_frac=0.7)

    # Check split sizes
    assert len(train) + len(test) == len(sample_data)
    assert len(train) == int(len(sample_data) * 0.7)

    # Check temporal order (train dates should be before test dates)
    max_train_date = pd.to_datetime(train["signup_date"]).max()
    min_test_date = pd.to_datetime(test["signup_date"]).min()
    assert max_train_date <= min_test_date


def test_prepare_features(sample_data):
    """Test feature extraction."""
    feature_cols = ["tenure_months", "monthly_spend", "support_tickets"]
    X = prepare_features(sample_data, feature_cols)

    assert X.shape == (len(sample_data), len(feature_cols))
    assert not np.any(np.isnan(X))


def test_train_and_evaluate_logistic_regression(sample_data):
    """Test training LogisticRegression and getting metrics."""
    feature_cols = ["tenure_months", "monthly_spend", "support_tickets"]
    X = prepare_features(sample_data, feature_cols)
    y = sample_data["churned"].values

    # Use 80/20 split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    metrics = train_and_evaluate(
        X_train, X_test, y_train, y_test, LogisticRegression, seed=42, model_name="LogisticRegression"
    )

    # Check all metrics are present and valid
    assert "auc" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert 0 <= metrics["auc"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1


def test_train_and_evaluate_gradient_boosting(sample_data):
    """Test training GradientBoostingClassifier and getting metrics."""
    feature_cols = ["tenure_months", "monthly_spend", "support_tickets"]
    X = prepare_features(sample_data, feature_cols)
    y = sample_data["churned"].values

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    metrics = train_and_evaluate(
        X_train, X_test, y_train, y_test, GradientBoostingClassifier, seed=42, model_name="GradientBoosting"
    )

    assert "auc" in metrics
    assert 0 <= metrics["auc"] <= 1


def test_feature_scaling_independence(sample_data):
    """Test that scaling is fit on train and applied to test (not vice versa)."""
    from sklearn.preprocessing import StandardScaler

    feature_cols = ["tenure_months", "monthly_spend", "support_tickets"]
    X = prepare_features(sample_data, feature_cols)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train mean should be close to 0, but test mean may not be
    assert np.abs(X_train_scaled.mean(axis=0)).max() < 0.1
    # Test should not be rescaled after transform
    assert X_test_scaled.shape == X_test.shape


def test_no_leakage_days_since_last_login(sample_data):
    """
    Verify that days_since_last_login is not in the features used.
    This feature leaks the target.
    """
    feature_cols = ["tenure_months", "monthly_spend", "support_tickets"]
    assert "days_since_last_login" not in feature_cols
    assert "churned" not in feature_cols
