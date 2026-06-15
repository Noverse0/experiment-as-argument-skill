"""Tests for model training and evaluation."""
import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.models import (
    train_logistic_regression,
    train_gradient_boosting,
    evaluate_model,
    baseline_majority_class
)


@pytest.fixture
def synthetic_data():
    """Create a synthetic classification dataset."""
    X_train, y_train = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=42
    )
    X_test, y_test = make_classification(
        n_samples=50,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=43
    )
    return X_train, y_train, X_test, y_test


def test_train_logistic_regression(synthetic_data):
    """Test logistic regression training."""
    X_train, y_train, _, _ = synthetic_data
    model = train_logistic_regression(X_train, y_train, seed=42)

    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

    # Check that the model can make predictions
    pred = model.predict(X_train[:10])
    assert len(pred) == 10
    assert all(p in [0, 1] for p in pred)


def test_train_gradient_boosting(synthetic_data):
    """Test gradient boosting training."""
    X_train, y_train, _, _ = synthetic_data
    model = train_gradient_boosting(X_train, y_train, seed=42)

    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

    # Check that the model can make predictions
    pred = model.predict(X_train[:10])
    assert len(pred) == 10
    assert all(p in [0, 1] for p in pred)


def test_evaluate_model(synthetic_data):
    """Test model evaluation metrics."""
    X_train, y_train, X_test, y_test = synthetic_data
    model = train_logistic_regression(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, "logistic_regression")

    # Check all required metrics are present
    expected_metrics = ["accuracy", "auc_roc", "f1", "precision", "recall"]
    assert all(m in metrics for m in expected_metrics)

    # Check metric values are in valid range
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["auc_roc"] <= 1
    assert 0 <= metrics["f1"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1

    assert metrics["model"] == "logistic_regression"


def test_baseline_majority_class():
    """Test baseline majority-class predictor."""
    y_test = np.array([0, 0, 1, 0, 0])  # 80% class 0
    baseline = baseline_majority_class(y_test)

    assert baseline["model"] == "baseline_majority_class"
    assert 0 <= baseline["accuracy"] <= 1
    assert 0 <= baseline["auc_roc"] <= 1


def test_models_beat_baseline(synthetic_data):
    """Test that both models beat the majority-class baseline."""
    X_train, y_train, X_test, y_test = synthetic_data

    lr_model = train_logistic_regression(X_train, y_train)
    gb_model = train_gradient_boosting(X_train, y_train)

    lr_metrics = evaluate_model(lr_model, X_test, y_test, "lr")
    gb_metrics = evaluate_model(gb_model, X_test, y_test, "gb")
    baseline = baseline_majority_class(y_test)

    # Both models should beat baseline on AUC
    assert lr_metrics["auc_roc"] >= baseline["auc_roc"]
    assert gb_metrics["auc_roc"] >= baseline["auc_roc"]
