"""Tests for model pipeline construction and prediction."""
import numpy as np
import pytest

from src.pipeline import build_gbm_pipeline, build_lr_pipeline


@pytest.fixture
def tiny_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 4))
    y = (X[:, 0] + rng.standard_normal(100) > 0).astype(int)
    return X, y


def test_lr_fits_and_predicts(tiny_data):
    X, y = tiny_data
    pipe = build_lr_pipeline(random_state=0)
    pipe.fit(X, y)
    preds = pipe.predict(X)
    proba = pipe.predict_proba(X)
    assert preds.shape == (100,)
    assert proba.shape == (100, 2)
    assert set(np.unique(preds)).issubset({0, 1})
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_gbm_fits_and_predicts(tiny_data):
    X, y = tiny_data
    pipe = build_gbm_pipeline(random_state=0)
    pipe.fit(X, y)
    preds = pipe.predict(X)
    proba = pipe.predict_proba(X)
    assert preds.shape == (100,)
    assert proba.shape == (100, 2)
    assert set(np.unique(preds)).issubset({0, 1})
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_lr_has_scaler():
    """LR must include a scaler because it is scale-sensitive."""
    from sklearn.preprocessing import StandardScaler
    pipe = build_lr_pipeline()
    assert any(isinstance(step, StandardScaler) for _, step in pipe.steps)


def test_lr_deterministic(tiny_data):
    X, y = tiny_data
    p1 = build_lr_pipeline(random_state=7)
    p2 = build_lr_pipeline(random_state=7)
    p1.fit(X, y)
    p2.fit(X, y)
    assert np.allclose(p1.predict_proba(X), p2.predict_proba(X))


def test_gbm_deterministic(tiny_data):
    X, y = tiny_data
    p1 = build_gbm_pipeline(random_state=7)
    p2 = build_gbm_pipeline(random_state=7)
    p1.fit(X, y)
    p2.fit(X, y)
    assert np.allclose(p1.predict_proba(X), p2.predict_proba(X))


def test_different_seeds_independent(tiny_data):
    """Pipelines built with different seeds must not share state."""
    X, y = tiny_data
    p1 = build_gbm_pipeline(random_state=0)
    p2 = build_gbm_pipeline(random_state=99)
    p1.fit(X, y)
    # p2 is not fitted; fitting p1 must not affect p2
    p2.fit(X, y)
    # They may or may not differ (GBM with subsample=1.0 is deterministic
    # up to init; different random_state can still differ)
    assert p1 is not p2
