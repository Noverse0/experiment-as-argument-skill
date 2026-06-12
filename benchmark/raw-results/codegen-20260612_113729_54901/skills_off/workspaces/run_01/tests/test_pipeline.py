"""Tests for pipeline construction and determinism."""
import pytest
import numpy as np
import pandas as pd

from src.pipeline import build_lr_pipeline, build_gb_pipeline


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({
        "tenure_months": rng.integers(1, 72, n),
        "monthly_spend": rng.gamma(2.0, 30.0, n),
        "support_tickets": rng.poisson(1.2, n),
        "signup_days": np.arange(n),
    })
    y = pd.Series(rng.integers(0, 2, n))
    return X, y


def test_lr_pipeline_predict_proba_shape(toy_data):
    X, y = toy_data
    p = build_lr_pipeline()
    p.fit(X, y)
    out = p.predict_proba(X)
    assert out.shape == (len(X), 2)


def test_lr_pipeline_probabilities_sum_to_one(toy_data):
    X, y = toy_data
    p = build_lr_pipeline()
    p.fit(X, y)
    sums = p.predict_proba(X).sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)


def test_gb_pipeline_predict_proba_shape(toy_data):
    X, y = toy_data
    p = build_gb_pipeline()
    p.fit(X, y)
    out = p.predict_proba(X)
    assert out.shape == (len(X), 2)


def test_lr_pipeline_is_deterministic(toy_data):
    X, y = toy_data
    p1 = build_lr_pipeline(seed=42)
    p2 = build_lr_pipeline(seed=42)
    p1.fit(X, y)
    p2.fit(X, y)
    np.testing.assert_array_equal(p1.predict(X), p2.predict(X))


def test_gb_pipeline_is_deterministic(toy_data):
    X, y = toy_data
    p1 = build_gb_pipeline(seed=99)
    p2 = build_gb_pipeline(seed=99)
    p1.fit(X, y)
    p2.fit(X, y)
    np.testing.assert_array_equal(p1.predict(X), p2.predict(X))


def test_different_seeds_can_differ(toy_data):
    """Different seeds should (with high probability) produce different predictions."""
    X, y = toy_data
    p1 = build_gb_pipeline(seed=1)
    p2 = build_gb_pipeline(seed=999)
    p1.fit(X, y)
    p2.fit(X, y)
    # Not guaranteed but nearly certain on random data
    assert not np.array_equal(p1.predict_proba(X), p2.predict_proba(X))
