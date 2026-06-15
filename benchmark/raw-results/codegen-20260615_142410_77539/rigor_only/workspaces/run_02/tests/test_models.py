"""Tests for model training and evaluation."""
import pytest
import numpy as np

from src.models import (
    train_logistic_regression,
    train_gradient_boosting,
    evaluate_model,
    baseline_majority_class,
)


@pytest.fixture
def synthetic_data():
    """Create simple synthetic data for testing."""
    np.random.seed(42)
    n_train, n_test = 100, 30
    n_features = 3

    X_train = np.random.randn(n_train, n_features)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

    X_test = np.random.randn(n_test, n_features)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

    return X_train, X_test, y_train, y_test


def test_train_logistic_regression(synthetic_data):
    """Test logistic regression training."""
    X_train, X_test, y_train, y_test = synthetic_data

    model = train_logistic_regression(X_train, y_train, random_state=42)

    # Check model is trained
    assert model is not None
    assert model.coef_ is not None
    assert model.coef_.shape == (1, 3)

    # Check can predict
    y_pred = model.predict(X_test)
    assert y_pred.shape == (len(y_test),)
    assert set(y_pred).issubset({0, 1})


def test_train_gradient_boosting(synthetic_data):
    """Test gradient boosting training."""
    X_train, X_test, y_train, y_test = synthetic_data

    model = train_gradient_boosting(X_train, y_train, random_state=42)

    # Check model is trained
    assert model is not None
    assert model.n_estimators == 100

    # Check can predict
    y_pred = model.predict(X_test)
    assert y_pred.shape == (len(y_test),)
    assert set(y_pred).issubset({0, 1})


def test_evaluate_model(synthetic_data):
    """Test evaluation metrics."""
    X_train, X_test, y_train, y_test = synthetic_data

    model = train_logistic_regression(X_train, y_train, random_state=42)
    metrics = evaluate_model(model, X_test, y_test, "TestModel")

    # Check all metrics are present
    assert 'model' in metrics
    assert 'roc_auc' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'neg_log_loss' in metrics

    # Check metric values are in valid ranges
    assert 0 <= metrics['roc_auc'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1'] <= 1
    assert metrics['neg_log_loss'] < 0  # negative loss


def test_baseline_majority_class(synthetic_data):
    """Test baseline majority class predictor."""
    X_train, X_test, y_train, y_test = synthetic_data

    baseline = baseline_majority_class(y_train, y_test)

    # Check structure
    assert baseline['model'] == 'Baseline (Majority Class)'
    assert 'roc_auc' in baseline
    assert 'precision' in baseline

    # Baseline ROC AUC should be lower than a good model
    model = train_logistic_regression(X_train, y_train, random_state=42)
    model_metrics = evaluate_model(model, X_test, y_test)

    # Model should be better than baseline on AUC
    assert model_metrics['roc_auc'] >= baseline['roc_auc']


def test_model_reproducibility(synthetic_data):
    """Test that same seed gives same results."""
    X_train, X_test, y_train, y_test = synthetic_data

    model1 = train_logistic_regression(X_train, y_train, random_state=42)
    model2 = train_logistic_regression(X_train, y_train, random_state=42)

    y_pred1 = model1.predict_proba(X_test)
    y_pred2 = model2.predict_proba(X_test)

    np.testing.assert_array_almost_equal(y_pred1, y_pred2)
