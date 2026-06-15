"""Tests for data preprocessing."""
import numpy as np
import pandas as pd
import pytest
from src.preprocessing import load_and_validate, preprocess_features, create_stratified_split


@pytest.fixture
def sample_data():
    """Create sample dataset for testing."""
    return pd.DataFrame({
        "customer_id": np.arange(1, 11),
        "signup_date": ["2023-01-01"] * 10,
        "tenure_months": np.arange(1, 11),
        "monthly_spend": np.ones(10) * 100.0,
        "support_tickets": np.zeros(10, dtype=int),
        "days_since_last_login": np.ones(10) * 10,
        "churned": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
    })


def test_preprocess_features_clean(sample_data):
    """Test feature preprocessing with clean feature set."""
    X, y, scaler = preprocess_features(sample_data, feature_set="clean")

    assert X.shape == (10, 4), f"Expected (10, 4), got {X.shape}"
    assert y.shape == (10,)
    assert len(np.unique(y)) == 2, "Target should have two classes"


def test_preprocess_features_leaked(sample_data):
    """Test feature preprocessing with leaked feature set."""
    X, y, scaler = preprocess_features(sample_data, feature_set="leaked")

    assert X.shape == (10, 5), f"Expected (10, 5), got {X.shape}"
    assert y.shape == (10,)


def test_preprocessing_scaling(sample_data):
    """Test that StandardScaler is applied correctly."""
    X, y, scaler = preprocess_features(sample_data, feature_set="clean")

    # Check that features are roughly zero-mean
    assert np.abs(X.mean(axis=0)).max() < 1e-10, "Features should be centered"


def test_stratified_split_preserves_ratio(sample_data):
    """Test that stratified split preserves target ratio."""
    X, y, scaler = preprocess_features(sample_data, feature_set="clean")

    X_train, X_test, y_train, y_test = create_stratified_split(X, y, test_size=0.3, random_state=42)

    # Check sizes
    assert len(X_train) + len(X_test) == len(X)
    assert len(X_train) > 0 and len(X_test) > 0

    # Check class balance is roughly preserved
    original_ratio = y.mean()
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()

    # Allow some variance due to small sample, but should be close
    assert 0.3 < train_ratio < 0.7, f"Train target ratio should be ~0.5, got {train_ratio}"
    assert 0.3 < test_ratio < 0.7, f"Test target ratio should be ~0.5, got {test_ratio}"


def test_multiple_seeds_produce_different_splits(sample_data):
    """Test that different seeds produce different train/test splits."""
    X, y, scaler = preprocess_features(sample_data, feature_set="clean")

    X_train1, X_test1, _, _ = create_stratified_split(X, y, test_size=0.3, random_state=1)
    X_train2, X_test2, _, _ = create_stratified_split(X, y, test_size=0.3, random_state=2)

    # Splits should be different
    assert not np.array_equal(X_train1, X_train2), "Different seeds should produce different splits"


def test_deterministic_split(sample_data):
    """Test that same seed produces same split."""
    X, y, scaler = preprocess_features(sample_data, feature_set="clean")

    X_train1, X_test1, _, _ = create_stratified_split(X, y, test_size=0.3, random_state=42)
    X_train2, X_test2, _, _ = create_stratified_split(X, y, test_size=0.3, random_state=42)

    # Same seed should give same split
    assert np.array_equal(X_train1, X_train2), "Same seed should produce same split"
    assert np.array_equal(X_test1, X_test2)
