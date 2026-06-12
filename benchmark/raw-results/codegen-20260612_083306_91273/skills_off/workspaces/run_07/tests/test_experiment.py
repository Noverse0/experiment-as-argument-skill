"""Tests for experiment logic."""
import pandas as pd
import numpy as np
import pytest
from sklearn.datasets import make_classification
from src.experiment import run_model_trials, sanity_check_label_shuffle, sanity_check_overfit_tiny_batch
from src.models import get_baseline_model, get_logistic_regression, get_gradient_boosting


@pytest.fixture
def synthetic_data():
    """Create synthetic classification data for testing."""
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_clusters_per_class=1,
        random_state=42,
        flip_y=0.1,
    )
    X = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(4)])
    y = pd.Series(y)

    # Split
    train_idx = y.index[:150]
    test_idx = y.index[150:]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    return X_train, X_test, y_train, y_test


def test_baseline_model_trains(synthetic_data):
    """Test that baseline model trains and predicts."""
    X_train, X_test, y_train, y_test = synthetic_data
    baseline = get_baseline_model()
    baseline.fit(X_train, y_train)

    y_pred = baseline.predict(X_test)
    assert len(y_pred) == len(y_test)
    assert set(y_pred) <= {0, 1}


def test_logistic_regression_trains(synthetic_data):
    """Test logistic regression model."""
    X_train, X_test, y_train, y_test = synthetic_data
    model = get_logistic_regression(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    assert len(y_pred) == len(y_test)
    assert y_proba.shape == (len(y_test), 2)
    assert np.all((y_proba >= 0) & (y_proba <= 1))


def test_gradient_boosting_trains(synthetic_data):
    """Test gradient boosting model."""
    X_train, X_test, y_train, y_test = synthetic_data
    model = get_gradient_boosting(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    assert len(y_pred) == len(y_test)
    assert y_proba.shape == (len(y_test), 2)


def test_run_model_trials(synthetic_data):
    """Test multi-trial model evaluation."""
    X_train, X_test, y_train, y_test = synthetic_data
    model = get_logistic_regression()
    results = run_model_trials(model, X_train, X_test, y_train, y_test, n_trials=3)

    assert 'roc_auc' in results
    assert 'precision' in results
    assert 'recall' in results
    assert 'f1' in results

    for metric in results:
        assert 'mean' in results[metric]
        assert 'std' in results[metric]
        assert 'n' in results[metric]
        assert results[metric]['n'] == 3


def test_sanity_check_label_shuffle(synthetic_data):
    """Test label shuffle sanity check."""
    X_train, X_test, y_train, y_test = synthetic_data
    result = sanity_check_label_shuffle(X_train, X_test, y_train, y_test)
    assert isinstance(result, bool)


def test_sanity_check_overfit_tiny_batch(synthetic_data):
    """Test overfit tiny batch sanity check."""
    X_train, X_test, y_train, y_test = synthetic_data
    result = sanity_check_overfit_tiny_batch(X_train, X_test, y_train, y_test)
    assert isinstance(result, bool)


def test_model_determinism(synthetic_data):
    """Test that same random_state produces same predictions."""
    X_train, X_test, y_train, y_test = synthetic_data

    model1 = get_logistic_regression(random_state=42)
    model1.fit(X_train, y_train)
    pred1 = model1.predict(X_test)

    model2 = get_logistic_regression(random_state=42)
    model2.fit(X_train, y_train)
    pred2 = model2.predict(X_test)

    assert np.array_equal(pred1, pred2), "Same seed should produce same predictions"


def test_gradient_boosting_vs_logistic_regression(synthetic_data):
    """Test that both models produce different predictions (not identical)."""
    X_train, X_test, y_train, y_test = synthetic_data

    lr_model = get_logistic_regression(random_state=42)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    gb_model = get_gradient_boosting(random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)

    # Models should differ on at least some predictions
    assert not np.array_equal(lr_pred, gb_pred)
