"""Tests for the model factory."""
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.models import (
    MODEL_REGISTRY,
    make_baseline,
    make_gradient_boosting,
    make_logistic_regression,
)


@pytest.fixture
def tiny_dataset():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 4))
    y = rng.integers(0, 2, 60)
    return X, y


def test_make_logistic_regression_type():
    model = make_logistic_regression(seed=0)
    assert isinstance(model, LogisticRegression)


def test_make_gradient_boosting_type():
    model = make_gradient_boosting(seed=0)
    assert isinstance(model, GradientBoostingClassifier)


def test_make_baseline_type():
    model = make_baseline()
    assert isinstance(model, DummyClassifier)


def test_model_registry_keys():
    assert "logistic_regression" in MODEL_REGISTRY
    assert "gradient_boosting" in MODEL_REGISTRY


def test_lr_different_seeds_same_data_same_result(tiny_dataset):
    X, y = tiny_dataset
    m0 = make_logistic_regression(0)
    m1 = make_logistic_regression(1)
    m0.fit(X, y)
    m1.fit(X, y)
    # LR is deterministic given the same data regardless of random_state
    np.testing.assert_array_almost_equal(
        m0.predict_proba(X), m1.predict_proba(X), decimal=5
    )


def test_lr_predict_proba_shape(tiny_dataset):
    X, y = tiny_dataset
    model = make_logistic_regression(0)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_gb_predict_proba_shape(tiny_dataset):
    X, y = tiny_dataset
    model = make_gradient_boosting(0)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_all_registry_models_fit_and_predict(tiny_dataset):
    X, y = tiny_dataset
    for name, make_fn in MODEL_REGISTRY.items():
        model = make_fn(seed=0)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y), f"{name}: wrong prediction length"
