"""Tests for experiment logic: splits, preprocessing, models."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.experiment import time_based_split, run_single_trial
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


def test_time_based_split():
    """Test time-based split maintains temporal ordering."""
    n = 100
    X = pd.DataFrame({"feat": np.random.randn(n)})
    y = pd.Series(np.random.randint(0, 2, n))
    signup_dates = pd.date_range("2023-01-01", periods=n, freq="D")

    X_train, X_test, y_train, y_test = time_based_split(X, y, signup_dates, split_date_pct=0.8)

    # Check split sizes
    assert len(X_train) == 80
    assert len(X_test) == 20

    # Check no overlap
    assert len(set(X_train.index) & set(X_test.index)) == 0

    # Check train/test y sizes match X
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_time_split_respects_temporal_order():
    """Test that train indices come before test indices in time."""
    n = 100
    X = pd.DataFrame({"feat": np.random.randn(n)})
    y = pd.Series(np.random.randint(0, 2, n))
    signup_dates = pd.date_range("2023-01-01", periods=n, freq="D")

    X_train, X_test, y_train, y_test = time_based_split(X, y, signup_dates, split_date_pct=0.8)

    # Get the min date in train and max date in test
    # (train rows should come before test rows in the sorted order)
    sorted_idx = np.argsort(signup_dates.values)
    train_indices_in_sorted = np.where(np.isin(sorted_idx, X_train.index))[0]
    test_indices_in_sorted = np.where(np.isin(sorted_idx, X_test.index))[0]

    # All train indices should appear before all test indices in sorted order
    if len(train_indices_in_sorted) > 0 and len(test_indices_in_sorted) > 0:
        assert train_indices_in_sorted[-1] < test_indices_in_sorted[0]


def test_single_trial_logistic_regression():
    """Test that logistic regression training completes and predicts."""
    n = 100
    X = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n),
        "feat3": np.random.randn(n),
    })
    y = pd.Series(np.random.randint(0, 2, n))
    signup_dates = pd.date_range("2023-01-01", periods=n, freq="D")

    metrics = run_single_trial(X, y, signup_dates, LogisticRegression, seed=42)

    # Check all metrics are present and in valid range
    assert "auc" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "accuracy" in metrics

    assert 0 <= metrics["auc"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1
    assert 0 <= metrics["accuracy"] <= 1


def test_single_trial_gradient_boosting():
    """Test that gradient boosting training completes and predicts."""
    n = 100
    X = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n),
        "feat3": np.random.randn(n),
    })
    y = pd.Series(np.random.randint(0, 2, n))
    signup_dates = pd.date_range("2023-01-01", periods=n, freq="D")

    metrics = run_single_trial(X, y, signup_dates, GradientBoostingClassifier, seed=42)

    # Check all metrics are present and in valid range
    assert "auc" in metrics
    assert 0 <= metrics["auc"] <= 1


def test_reproducibility_with_seed():
    """Test that same seed produces same results."""
    n = 100
    X = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n),
        "feat3": np.random.randn(n),
    })
    y = pd.Series(np.random.randint(0, 2, n))
    signup_dates = pd.date_range("2023-01-01", periods=n, freq="D")

    metrics1 = run_single_trial(X, y, signup_dates, LogisticRegression, seed=42)
    metrics2 = run_single_trial(X, y, signup_dates, LogisticRegression, seed=42)

    # Same seed should produce identical results
    for metric in ["auc", "precision", "recall", "f1", "accuracy"]:
        assert metrics1[metric] == metrics2[metric]
