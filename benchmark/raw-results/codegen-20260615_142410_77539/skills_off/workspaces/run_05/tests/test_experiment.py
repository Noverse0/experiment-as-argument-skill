"""Tests for experiment logic."""
import pytest
import numpy as np
import pandas as pd
from src.experiment import (
    train_and_evaluate, baseline_majority_class, sanity_check_overfit_one_batch,
    sanity_check_label_shuffle, run_single_experiment, aggregate_results, summarize_results
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


@pytest.fixture
def dummy_data():
    """Create dummy train/test data."""
    np.random.seed(42)
    X_train = np.random.randn(50, 3)
    X_test = np.random.randn(20, 3)
    y_train = np.random.randint(0, 2, 50)
    y_test = np.random.randint(0, 2, 20)
    return X_train, X_test, y_train, y_test


def test_train_and_evaluate(dummy_data):
    """Test model training and evaluation."""
    X_train, X_test, y_train, y_test = dummy_data
    lr = LogisticRegression(max_iter=1000, random_state=42)
    metrics = train_and_evaluate(X_train, X_test, y_train, y_test, lr, "logistic_regression")

    assert metrics["model"] == "logistic_regression"
    assert "accuracy" in metrics
    assert "auc_roc" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["auc_roc"] <= 1


def test_baseline_majority_class(dummy_data):
    """Test baseline model."""
    X_train, X_test, y_train, y_test = dummy_data
    metrics = baseline_majority_class(y_train, y_test)

    assert metrics["model"] == "baseline_majority"
    assert "accuracy" in metrics
    assert 0 <= metrics["accuracy"] <= 1


def test_sanity_check_overfit_one_batch(dummy_data):
    """Test overfit sanity check."""
    X_train, X_test, y_train, y_test = dummy_data
    result = sanity_check_overfit_one_batch(X_train, y_train)

    assert "sanity_check" in result
    assert result["sanity_check"] == "overfit_one_batch"
    assert "lr_accuracy_on_tiny" in result
    assert "gb_accuracy_on_tiny" in result
    assert "passed" in result


def test_sanity_check_label_shuffle(dummy_data):
    """Test label shuffle sanity check."""
    X_train, X_test, y_train, y_test = dummy_data
    result = sanity_check_label_shuffle(X_train, X_test, y_train, y_test)

    assert result["sanity_check"] == "label_shuffle"
    assert "lr_auc_shuffled" in result
    assert "gb_auc_shuffled" in result
    assert isinstance(result["passed"], bool)


def test_run_single_experiment(dummy_data):
    """Test single experiment run."""
    X_train, X_test, y_train, y_test = dummy_data
    results = run_single_experiment(X_train, X_test, y_train, y_test, seed=42)

    assert len(results) == 3  # baseline + lr + gb
    assert results[0]["model"] == "baseline_majority"
    assert results[1]["model"] == "logistic_regression"
    assert results[2]["model"] == "gradient_boosting"


def test_aggregate_results():
    """Test result aggregation."""
    results = [
        {"model": "lr", "auc_roc": 0.8},
        {"model": "gb", "auc_roc": 0.85},
        {"model": "lr", "auc_roc": 0.82},
        {"model": "gb", "auc_roc": 0.83},
    ]
    df = aggregate_results(results)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4


def test_summarize_results():
    """Test result summarization."""
    results = [
        {"model": "lr", "auc_roc": 0.80, "accuracy": 0.75, "precision": 0.7, "recall": 0.8, "f1": 0.75},
        {"model": "gb", "auc_roc": 0.85, "accuracy": 0.80, "precision": 0.8, "recall": 0.85, "f1": 0.825},
        {"model": "lr", "auc_roc": 0.82, "accuracy": 0.76, "precision": 0.72, "recall": 0.82, "f1": 0.77},
        {"model": "gb", "auc_roc": 0.83, "accuracy": 0.79, "precision": 0.79, "recall": 0.83, "f1": 0.81},
    ]
    df = aggregate_results(results)
    summary = summarize_results(df)

    assert "lr" in summary
    assert "gb" in summary
    assert summary["lr"]["n_runs"] == 2
    assert summary["gb"]["n_runs"] == 2
    assert summary["lr"]["auc_roc"][0] > 0  # mean
    assert summary["lr"]["auc_roc"][1] >= 0  # std
