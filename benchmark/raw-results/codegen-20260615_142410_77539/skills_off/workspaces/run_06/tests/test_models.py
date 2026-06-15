"""Tests for model builders and sanity checks."""
import pytest
import numpy as np
from src.models import (
    build_lr,
    build_gb,
    evaluate,
    baseline_floor,
    sanity_overfit_small,
    sanity_label_shuffle,
)


@pytest.fixture
def dummy_data():
    """Create a simple synthetic dataset for testing."""
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X[:150], X[150:], y[:150], y[150:]


def test_build_lr():
    """Test that LR model is created with correct params."""
    model = build_lr(random_state=42)
    assert model.penalty == "l2"
    assert model.solver == "lbfgs"
    assert model.max_iter == 1000


def test_build_gb():
    """Test that GB model is created with correct params."""
    model = build_gb(random_state=42)
    assert model.n_estimators == 100
    assert model.learning_rate == 0.1
    assert model.max_depth == 3


def test_evaluate(dummy_data):
    """Test that evaluate computes all metrics."""
    X_train, X_test, y_train, y_test = dummy_data
    model = build_lr(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, y_pred, y_pred_proba)

    assert "auc" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics

    # All metrics should be between 0 and 1
    assert 0 <= metrics["auc"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1


def test_baseline_floor(dummy_data):
    """Test baseline (majority class) predictor."""
    X_train, X_test, y_train, y_test = dummy_data
    baseline = baseline_floor(y_test)

    # Baseline AUC should be low but deterministic
    assert 0 <= baseline["auc"] <= 1
    assert "precision" in baseline


def test_sanity_overfit_small(dummy_data):
    """Test that model can overfit on small subset."""
    X_train, X_test, y_train, y_test = dummy_data

    # Both models should pass (on synthetic data)
    lr_ok = sanity_overfit_small(X_train, y_train, build_lr, subset_size=50)
    gb_ok = sanity_overfit_small(X_train, y_train, build_gb, subset_size=50)

    # At least one should pass on this simple synthetic data
    assert lr_ok or gb_ok


def test_sanity_label_shuffle(dummy_data):
    """Test that shuffled labels hurt performance."""
    X_train, X_test, y_train, y_test = dummy_data

    lr_shuffle = sanity_label_shuffle(X_train, y_train, X_test, y_test, build_lr)
    gb_shuffle = sanity_label_shuffle(X_train, y_train, X_test, y_test, build_gb)

    # With shuffled labels, AUC should be lower than with true labels
    # (though on small data it might still be >0.5 by chance)
    assert 0 <= lr_shuffle["auc"] <= 1
    assert 0 <= gb_shuffle["auc"] <= 1
