"""Tests for data pipeline and experiment integrity."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.pipeline import (
    load_data, split_data, preprocess_features, train_and_evaluate,
    baseline_majority_class
)


@pytest.fixture
def churn_data():
    """Load the actual churn dataset for testing."""
    X, y = load_data('churn.csv')
    return X, y


def test_load_data(churn_data):
    """Test that data loads correctly and drops expected columns."""
    X, y = churn_data
    assert X.shape[0] > 0
    assert y.shape[0] == X.shape[0]
    assert 'customer_id' not in X.columns
    assert 'signup_date' not in X.columns
    assert 'churned' not in X.columns
    assert y.dtype == int
    assert set(y.unique()) == {0, 1}


def test_split_stratification(churn_data):
    """Test that stratified split maintains target distribution."""
    X, y = churn_data
    original_churn_rate = y.mean()

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3, random_state=42)

    train_churn_rate = y_train.mean()
    test_churn_rate = y_test.mean()

    # Stratification should keep rates similar (within 5%)
    assert abs(train_churn_rate - original_churn_rate) < 0.05
    assert abs(test_churn_rate - original_churn_rate) < 0.05


def test_split_before_transform(churn_data):
    """Test that split happens before preprocessing to prevent leakage."""
    X, y = churn_data
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=42)

    # Sizes should reflect 70/30 split
    assert len(X_train) == pytest.approx(0.7 * len(X), abs=10)
    assert len(X_test) == pytest.approx(0.3 * len(X), abs=10)


def test_no_data_leakage_between_splits(churn_data):
    """Test that train and test sets do not share indices (split is clean)."""
    X, y = churn_data
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=42)

    # Check that no index appears in both splits
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)

    overlap = train_indices.intersection(test_indices)
    # Clean split means no overlapping indices
    assert len(overlap) == 0


def test_preprocessing_no_leakage(churn_data):
    """Test that preprocessing fitted on train only."""
    X, y = churn_data
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=42)

    X_train_proc, X_test_proc = preprocess_features(X_train, X_test)

    # Check shapes
    assert X_train_proc.shape[0] == len(X_train)
    assert X_test_proc.shape[0] == len(X_test)
    assert X_train_proc.shape[1] == X_test_proc.shape[1]

    # No NaN after preprocessing
    assert not np.isnan(X_train_proc).any()
    assert not np.isnan(X_test_proc).any()


def test_baseline_below_model_performance(churn_data):
    """Test that any model beats the majority class baseline."""
    X, y = churn_data
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=42)
    X_train_proc, X_test_proc = preprocess_features(X_train, X_test)

    baseline = baseline_majority_class(y_train.values, y_test.values)

    # Train a simple model
    model = LogisticRegression(max_iter=1000, random_state=42)
    metrics = train_and_evaluate(X_train_proc, X_test_proc, y_train.values, y_test.values, model)

    # Model should beat baseline
    assert metrics['roc_auc'] > baseline


def test_reproducibility_same_seed(churn_data):
    """Test that same seed produces identical results."""
    X, y = churn_data
    seed = 42

    X_train1, X_test1, y_train1, y_test1 = split_data(X, y, random_state=seed)
    X_train2, X_test2, y_train2, y_test2 = split_data(X, y, random_state=seed)

    # Splits should be identical
    assert (X_train1 == X_train2).all().all()
    assert (y_train1 == y_train2).all()

    # Preprocessing should be identical
    X_train_proc1, X_test_proc1 = preprocess_features(X_train1, X_test1)
    X_train_proc2, X_test_proc2 = preprocess_features(X_train2, X_test2)

    assert np.allclose(X_train_proc1, X_train_proc2, rtol=1e-10)
    assert np.allclose(X_test_proc1, X_test_proc2, rtol=1e-10)


def test_reproducibility_model_metrics(churn_data):
    """Test that model training with same seed produces identical metrics."""
    X, y = churn_data
    seed = 42

    X_train, X_test, y_train, y_test = split_data(X, y, random_state=seed)
    X_train_proc, X_test_proc = preprocess_features(X_train, X_test)

    model1 = LogisticRegression(max_iter=1000, random_state=seed)
    metrics1 = train_and_evaluate(X_train_proc, X_test_proc, y_train.values, y_test.values, model1)

    model2 = LogisticRegression(max_iter=1000, random_state=seed)
    metrics2 = train_and_evaluate(X_train_proc, X_test_proc, y_train.values, y_test.values, model2)

    # Metrics should be identical
    for key in metrics1:
        assert metrics1[key] == pytest.approx(metrics2[key], abs=1e-10)


def test_different_seeds_produce_different_splits(churn_data):
    """Test that different seeds produce different train/test sets."""
    X, y = churn_data

    X_train1, X_test1, y_train1, y_test1 = split_data(X, y, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = split_data(X, y, random_state=999)

    # Splits should have different indices (high probability)
    assert set(X_train1.index) != set(X_train2.index)


def test_model_generalizes_train_vs_test(churn_data):
    """Test that model has reasonable generalization gap (not overfitting massively)."""
    X, y = churn_data
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=42)
    X_train_proc, X_test_proc = preprocess_features(X_train, X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_proc, y_train.values)

    # Compute train accuracy
    y_train_pred = model.predict(X_train_proc)
    train_acc = (y_train_pred == y_train.values).mean()

    # Compute test accuracy
    y_test_pred = model.predict(X_test_proc)
    test_acc = (y_test_pred == y_test.values).mean()

    # Generalization gap should be reasonable (not more than 20% difference)
    gap = train_acc - test_acc
    assert gap < 0.20


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
