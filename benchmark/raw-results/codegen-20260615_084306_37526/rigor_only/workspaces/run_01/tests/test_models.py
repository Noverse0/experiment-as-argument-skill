"""Tests for model factory functions."""
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.models import make_gradient_boosting, make_logistic_regression


@pytest.fixture
def small_data():
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame({
        "tenure_months": rng.integers(1, 72, n),
        "monthly_spend": rng.uniform(10, 200, n),
        "support_tickets": rng.integers(0, 5, n),
        "signup_month": rng.integers(1, 13, n),
    })
    y = pd.Series((rng.random(n) > 0.65).astype(int))
    return X, y


def test_lr_fit_and_predict(small_data):
    X, y = small_data
    model = make_logistic_regression()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_gbm_fit_and_predict(small_data):
    X, y = small_data
    model = make_gradient_boosting()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_lr_predict_proba_sums_to_one(small_data):
    X, y = small_data
    model = make_logistic_regression()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_gbm_predict_proba_sums_to_one(small_data):
    X, y = small_data
    model = make_gradient_boosting()
    model.fit(X, y)
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_lr_unfitted_raises(small_data):
    X, y = small_data
    model = make_logistic_regression()
    with pytest.raises(NotFittedError):
        model.predict(X)


def test_make_returns_independent_instances(small_data):
    X, y = small_data
    m1 = make_logistic_regression()
    m2 = make_logistic_regression()
    m1.fit(X, y)
    # m2 must still be unfitted
    with pytest.raises(NotFittedError):
        m2.predict(X)
