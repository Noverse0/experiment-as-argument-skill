"""Tests for model pipeline construction and basic fit/predict behaviour."""
import numpy as np
import pandas as pd
import pytest

from src.pipeline import make_gb_pipeline, make_lr_pipeline


@pytest.fixture
def dummy_data():
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({
        "tenure_months": rng.integers(1, 72, n),
        "monthly_spend": rng.gamma(2, 30, n),
        "support_tickets": rng.poisson(1.2, n),
        "signup_year": np.full(n, 2023),
        "signup_month": rng.integers(1, 13, n),
        "signup_dayofyear": rng.integers(1, 366, n),
    })
    y = pd.Series(rng.integers(0, 2, n))
    return X, y


def test_lr_pipeline_fit_predict(dummy_data):
    X, y = dummy_data
    pipe = make_lr_pipeline()
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_gb_pipeline_fit_predict(dummy_data):
    X, y = dummy_data
    pipe = make_gb_pipeline()
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_lr_predict_proba_shape_and_range(dummy_data):
    X, y = dummy_data
    pipe = make_lr_pipeline()
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_gb_predict_proba_shape_and_range(dummy_data):
    X, y = dummy_data
    pipe = make_gb_pipeline()
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_pipelines_are_independent_instances():
    p1 = make_lr_pipeline()
    p2 = make_lr_pipeline()
    assert p1 is not p2


def test_gb_pipeline_uses_random_state(dummy_data):
    X, y = dummy_data
    p1 = make_gb_pipeline(random_state=0)
    p2 = make_gb_pipeline(random_state=0)
    p1.fit(X, y)
    p2.fit(X, y)
    np.testing.assert_array_equal(
        p1.predict_proba(X), p2.predict_proba(X)
    )
