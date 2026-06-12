"""Tests for experiment logic."""
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from src.experiment import (
    baseline_majority_class,
    overfit_tiny_subset_check,
    label_shuffle_sanity_check,
    evaluate_model,
)


@pytest.fixture
def sample_data():
    """Create synthetic train/val/test data."""
    np.random.seed(42)
    n_train, n_val, n_test = 100, 30, 30

    X_train = pd.DataFrame({
        "feat1": np.random.randn(n_train),
        "feat2": np.random.randn(n_train),
    })
    y_train = pd.Series(np.random.randint(0, 2, n_train))

    X_val = pd.DataFrame({
        "feat1": np.random.randn(n_val),
        "feat2": np.random.randn(n_val),
    })
    y_val = pd.Series(np.random.randint(0, 2, n_val))

    X_test = pd.DataFrame({
        "feat1": np.random.randn(n_test),
        "feat2": np.random.randn(n_test),
    })
    y_test = pd.Series(np.random.randint(0, 2, n_test))

    return X_train, y_train, X_val, y_val, X_test, y_test


def test_baseline_majority_class(sample_data):
    """Verify baseline computes majority class accuracy correctly."""
    X_train, y_train, X_val, y_val, X_test, y_test = sample_data

    baseline = baseline_majority_class(y_train, y_test)

    # Baseline should be between 0.5 and 1.0 for binary classification
    assert 0.5 <= baseline <= 1.0

    # Manually verify: predict all as majority class
    majority = np.bincount(y_train).argmax()
    expected = (np.full_like(y_test, majority) == y_test).mean()
    assert baseline == pytest.approx(expected)


def test_overfit_tiny_subset_check_high_accuracy():
    """Verify that model can overfit on a tiny subset with learnable data."""
    np.random.seed(42)
    n = 100

    # Create data where feat1 is strongly predictive of y
    X_train = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n),
    })
    y_train = pd.Series((X_train["feat1"] > 0).astype(int))

    accuracy = overfit_tiny_subset_check(X_train, y_train, LogisticRegression, subset_size=20)

    # Should achieve high accuracy on the tiny subset (close to 1.0)
    assert accuracy > 0.65


def test_label_shuffle_sanity_check_low_accuracy(sample_data):
    """Verify that shuffled labels result in poor performance."""
    X_train, y_train, X_val, y_val, _, _ = sample_data

    np.random.seed(42)
    accuracy = label_shuffle_sanity_check(X_train, y_train, X_val, y_val, LogisticRegression)

    # With shuffled labels, accuracy should be near random (0.5 for binary)
    # Allow some tolerance for randomness
    assert accuracy < 0.65


def test_evaluate_model_returns_all_metrics(sample_data):
    """Verify evaluate_model returns metrics for all splits."""
    X_train, y_train, X_val, y_val, X_test, y_test = sample_data

    model = LogisticRegression(random_state=42)
    results = evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test)

    # Should have results for all three splits
    assert "train" in results
    assert "val" in results
    assert "test" in results

    # Each split should have all required metrics
    for split_name in ["train", "val", "test"]:
        for metric in ["accuracy", "f1", "precision", "recall", "auc_roc"]:
            assert metric in results[split_name]
            assert 0.0 <= results[split_name][metric] <= 1.0


def test_evaluate_model_logs_better_on_data_it_trained():
    """Verify that model performs better on training data (overfitting check)."""
    np.random.seed(42)
    n = 200

    # Create data where model can easily learn the pattern
    X_train = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n),
    })
    y_train = (X_train["feat1"] > 0).astype(int)

    X_test = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n),
    })
    y_test = (X_test["feat1"] > 0).astype(int)

    model = LogisticRegression(random_state=42, max_iter=1000)
    results = evaluate_model(model, X_train, X_train, X_test, y_train, y_train, y_test)

    # Training accuracy should be >= test accuracy (or very close)
    assert results["train"]["accuracy"] >= results["test"]["accuracy"] - 0.05


def test_evaluate_model_with_gradient_boosting(sample_data):
    """Verify evaluate_model works with GradientBoostingClassifier."""
    X_train, y_train, X_val, y_val, X_test, y_test = sample_data

    model = GradientBoostingClassifier(n_estimators=10, random_state=42)
    results = evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test)

    assert "train" in results
    assert "val" in results
    assert "test" in results

    # Verify all metrics are valid
    for split_name in ["train", "val", "test"]:
        for metric in ["accuracy", "f1", "precision", "recall", "auc_roc"]:
            assert 0.0 <= results[split_name][metric] <= 1.0
