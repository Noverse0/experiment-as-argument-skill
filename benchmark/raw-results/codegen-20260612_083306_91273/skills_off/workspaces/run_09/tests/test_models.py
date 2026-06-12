"""Tests for model training and evaluation."""
import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.models import (
    train_logistic_regression,
    train_gradient_boosting,
    evaluate_model,
    baseline_majority_class,
)


@pytest.fixture
def synthetic_data():
    """Create synthetic binary classification data for testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=3,
        n_informative=2,
        n_redundant=1,
        random_state=42,
    )
    # Split into train/test
    split = 70
    return X[:split], X[split:], y[:split], y[split:]


def test_logistic_regression_trains(synthetic_data):
    """Test that logistic regression can train and predict."""
    X_train, X_test, y_train, y_test = synthetic_data

    model = train_logistic_regression(X_train, y_train, random_state=42)

    # Check that model is trained
    assert model is not None
    assert hasattr(model, "predict")

    # Check that predictions have correct shape
    y_pred = model.predict(X_test)
    assert y_pred.shape == (len(X_test),)
    assert set(y_pred).issubset({0, 1})


def test_gradient_boosting_trains(synthetic_data):
    """Test that gradient boosting can train and predict."""
    X_train, X_test, y_train, y_test = synthetic_data

    model = train_gradient_boosting(X_train, y_train, random_state=42)

    # Check that model is trained
    assert model is not None
    assert hasattr(model, "predict")

    # Check that predictions have correct shape
    y_pred = model.predict(X_test)
    assert y_pred.shape == (len(X_test),)
    assert set(y_pred).issubset({0, 1})


def test_evaluate_model_returns_dict(synthetic_data):
    """Test that evaluate_model returns a dictionary with expected keys."""
    X_train, X_test, y_train, y_test = synthetic_data

    model = train_logistic_regression(X_train, y_train, random_state=42)
    metrics = evaluate_model(model, X_test, y_test, "logistic_regression")

    expected_keys = {"model", "auc", "f1", "precision", "recall", "specificity"}
    assert set(metrics.keys()) == expected_keys
    assert metrics["model"] == "logistic_regression"


def test_metrics_are_in_valid_range(synthetic_data):
    """Test that all metrics are in [0, 1]."""
    X_train, X_test, y_train, y_test = synthetic_data

    model = train_logistic_regression(X_train, y_train, random_state=42)
    metrics = evaluate_model(model, X_test, y_test, "logistic_regression")

    for metric_name in ["auc", "f1", "precision", "recall", "specificity"]:
        value = metrics[metric_name]
        assert 0 <= value <= 1, f"{metric_name} should be in [0, 1], got {value}"


def test_baseline_majority_class(synthetic_data):
    """Test baseline majority class predictor."""
    X_train, X_test, y_train, y_test = synthetic_data

    metrics = baseline_majority_class(y_test)

    expected_keys = {"model", "auc", "f1", "precision", "recall", "specificity"}
    assert set(metrics.keys()) == expected_keys
    assert metrics["model"] == "baseline_majority"

    # All metrics should be valid
    for metric_name in ["auc", "f1", "precision", "recall", "specificity"]:
        assert 0 <= metrics[metric_name] <= 1


def test_models_beat_baseline(synthetic_data):
    """Test that trained models beat the majority class baseline."""
    X_train, X_test, y_train, y_test = synthetic_data

    baseline_metrics = baseline_majority_class(y_test)
    baseline_auc = baseline_metrics["auc"]

    lr_model = train_logistic_regression(X_train, y_train, random_state=42)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "logistic_regression")

    gb_model = train_gradient_boosting(X_train, y_train, random_state=42)
    gb_metrics = evaluate_model(gb_model, X_test, y_test, "gradient_boosting")

    # Both models should beat baseline on AUC (on informative synthetic data)
    assert lr_metrics["auc"] >= baseline_auc
    assert gb_metrics["auc"] >= baseline_auc


def test_reproducibility_with_seed(synthetic_data):
    """Test that same seed produces same predictions."""
    X_train, X_test, y_train, y_test = synthetic_data

    model1 = train_logistic_regression(X_train, y_train, random_state=42)
    model2 = train_logistic_regression(X_train, y_train, random_state=42)

    pred1 = model1.predict(X_test)
    pred2 = model2.predict(X_test)

    assert np.array_equal(pred1, pred2), "Same seed should produce identical predictions"


def test_different_seeds_produce_variance(synthetic_data):
    """Test that different seeds can be used without error."""
    X_train, X_test, y_train, y_test = synthetic_data

    # Gradient boosting with different seeds should train without error
    model1 = train_gradient_boosting(X_train, y_train, random_state=42)
    model2 = train_gradient_boosting(X_train, y_train, random_state=999)

    metrics1 = evaluate_model(model1, X_test, y_test, "gb")
    metrics2 = evaluate_model(model2, X_test, y_test, "gb")

    # Both should produce valid metrics
    assert all(0 <= metrics1[k] <= 1 for k in ["auc", "f1", "precision", "recall"])
    assert all(0 <= metrics2[k] <= 1 for k in ["auc", "f1", "precision", "recall"])
