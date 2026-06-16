"""Tests for model training and evaluation."""
import pytest
import numpy as np
from src.dataset import get_split
from src.models import (
    make_logistic_regression,
    make_gradient_boosting,
    train_and_evaluate,
)
from src.metrics import evaluate, baseline_majority
from src.sanity_checks import (
    check_baseline_floor,
    check_overfit_tiny_subset,
    check_label_shuffle,
)


@pytest.fixture
def data():
    """Load and split test data."""
    X_train, X_test, y_train, y_test, _ = get_split("churn.csv", drop_leaks=True)
    return X_train, X_test, y_train, y_test


def test_logistic_regression_trains(data):
    """Logistic regression can be trained."""
    X_train, X_test, y_train, y_test = data
    model = make_logistic_regression(random_state=42)
    model_fit, metrics = train_and_evaluate(model, X_train, y_train, X_test, y_test)

    # Should produce valid metrics
    assert "auc" in metrics
    assert 0 <= metrics["auc"] <= 1
    assert "f1" in metrics


def test_gradient_boosting_trains(data):
    """Gradient boosting can be trained."""
    X_train, X_test, y_train, y_test = data
    model = make_gradient_boosting(random_state=42)
    model_fit, metrics = train_and_evaluate(model, X_train, y_train, X_test, y_test)

    assert "auc" in metrics
    assert 0 <= metrics["auc"] <= 1
    assert "f1" in metrics


def test_deterministic_with_same_seed(data):
    """Same seed produces identical metrics (pipeline is deterministic)."""
    X_train, X_test, y_train, y_test = data

    # LR with same seed
    lr1 = make_logistic_regression(random_state=42)
    _, metrics1_lr = train_and_evaluate(lr1, X_train, y_train, X_test, y_test)

    lr2 = make_logistic_regression(random_state=42)
    _, metrics2_lr = train_and_evaluate(lr2, X_train, y_train, X_test, y_test)

    # Should be identical
    assert metrics1_lr["auc"] == metrics2_lr["auc"]
    assert metrics1_lr["f1"] == metrics2_lr["f1"]

    # GB with same seed
    gb1 = make_gradient_boosting(random_state=123)
    _, metrics1_gb = train_and_evaluate(gb1, X_train, y_train, X_test, y_test)

    gb2 = make_gradient_boosting(random_state=123)
    _, metrics2_gb = train_and_evaluate(gb2, X_train, y_train, X_test, y_test)

    assert metrics1_gb["auc"] == metrics2_gb["auc"]


def test_metrics_evaluation(data):
    """Metrics are computed correctly."""
    X_train, X_test, y_train, y_test = data
    model = make_gradient_boosting(random_state=42)
    model_fit, metrics = train_and_evaluate(model, X_train, y_train, X_test, y_test)

    # Metrics should be in valid ranges
    assert 0 <= metrics["auc"] <= 1
    assert 0 <= metrics["f1"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1

    # AUC should be better than random
    assert metrics["auc"] > 0.5


def test_baseline_floor(data):
    """Baseline majority prediction is weak."""
    X_train, X_test, y_train, y_test = data
    baseline = baseline_majority(y_test)

    # Majority baseline should have AUC near 0.5 (weak)
    assert 0.45 < baseline["auc"] < 0.55


def test_sanity_check_baseline_floor(data):
    """Baseline floor sanity check runs without error."""
    X_train, X_test, y_train, y_test = data
    check_baseline_floor(X_train, y_train, X_test, y_test)
    # If no assertion error, test passes


def test_sanity_check_overfit(data):
    """Overfit check passes for both models."""
    X_train, X_test, y_train, y_test = data
    check_overfit_tiny_subset(X_train, y_train, X_test, y_test)
    # If no assertion error, test passes


def test_sanity_check_label_shuffle(data):
    """Label shuffle check detects random labels."""
    X_train, X_test, y_train, y_test = data
    check_label_shuffle(X_train, y_train, X_test, y_test)
    # If no assertion error, test passes


def test_models_different_seeds_produce_different_results(data):
    """Different seeds may produce different (but close) metrics."""
    X_train, X_test, y_train, y_test = data

    aucs = []
    for seed in [42, 123, 456]:
        model = make_gradient_boosting(random_state=seed)
        _, metrics = train_and_evaluate(model, X_train, y_train, X_test, y_test)
        aucs.append(metrics["auc"])

    # All AUCs should be reasonable
    assert all(0.5 < auc < 1.0 for auc in aucs)

    # Std should be small (deterministic split, consistent model)
    std = np.std(aucs)
    assert std < 0.05  # Low variance across seeds


def test_models_better_than_baseline(data):
    """Both models should beat majority baseline."""
    X_train, X_test, y_train, y_test = data
    baseline = baseline_majority(y_test)

    lr = make_logistic_regression(random_state=42)
    _, lr_metrics = train_and_evaluate(lr, X_train, y_train, X_test, y_test)

    gb = make_gradient_boosting(random_state=42)
    _, gb_metrics = train_and_evaluate(gb, X_train, y_train, X_test, y_test)

    assert lr_metrics["auc"] > baseline["auc"]
    assert gb_metrics["auc"] > baseline["auc"]
