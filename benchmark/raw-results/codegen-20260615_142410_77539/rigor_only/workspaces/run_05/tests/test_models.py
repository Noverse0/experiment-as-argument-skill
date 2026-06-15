"""Tests for model training and evaluation."""
import pytest
import numpy as np

from src.models import ChurnPredictor, baseline_majority_class_score


@pytest.fixture
def dummy_data():
    """Create dummy training data."""
    np.random.seed(42)
    X_train = np.random.randn(100, 3)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.randn(30, 3)
    y_test = np.random.randint(0, 2, 30)
    return X_train, y_train, X_test, y_test


def test_logistic_regression_training(dummy_data):
    """Test that logistic regression trains without error."""
    X_train, y_train, X_test, y_test = dummy_data

    predictor = ChurnPredictor('logistic_regression')
    predictor.train(X_train, y_train)

    metrics = predictor.evaluate(X_test, y_test)

    assert 'auc' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'accuracy' in metrics
    assert 0 <= metrics['auc'] <= 1


def test_gradient_boosting_training(dummy_data):
    """Test that gradient boosting trains without error."""
    X_train, y_train, X_test, y_test = dummy_data

    predictor = ChurnPredictor('gradient_boosting')
    predictor.train(X_train, y_train)

    metrics = predictor.evaluate(X_test, y_test)

    assert 'auc' in metrics
    assert 0 <= metrics['auc'] <= 1


def test_predict_output_shape(dummy_data):
    """Test prediction shapes."""
    X_train, y_train, X_test, y_test = dummy_data

    predictor = ChurnPredictor('logistic_regression')
    predictor.train(X_train, y_train)

    y_pred = predictor.predict(X_test)
    y_proba = predictor.predict_proba(X_test)

    assert y_pred.shape == (30,)
    assert y_proba.shape == (30, 2)
    assert np.all((y_proba >= 0) & (y_proba <= 1))
    assert np.allclose(y_proba.sum(axis=1), 1.0)


def test_baseline_majority_class(dummy_data):
    """Test baseline scorer."""
    X_train, y_train, X_test, y_test = dummy_data

    # Majority class is 0 or 1 depending on distribution
    majority = np.bincount(y_test).argmax()

    metrics = baseline_majority_class_score(y_test, majority)

    assert 'auc' in metrics
    assert 0 <= metrics['accuracy'] <= 1


def test_invalid_model_name():
    """Test that invalid model name raises ValueError."""
    with pytest.raises(ValueError):
        ChurnPredictor('invalid_model')
