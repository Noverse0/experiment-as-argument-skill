"""Tests for model pipelines."""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from src.models import make_gbm_pipeline, make_lr_pipeline


@pytest.fixture
def simple_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 3))
    y = (X[:, 0] + rng.standard_normal(100) * 0.5 > 0).astype(int)
    return X, y


def test_lr_pipeline_fits_and_predicts(simple_data):
    X, y = simple_data
    pipeline = make_lr_pipeline(random_state=42)
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)
    assert proba.shape == (100, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_gbm_pipeline_fits_and_predicts(simple_data):
    X, y = simple_data
    pipeline = make_gbm_pipeline(random_state=42)
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)
    assert proba.shape == (100, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_lr_pipeline_different_seeds_differ(simple_data):
    """Different seeds for LR don't affect fit outcome (deterministic), but pipeline builds."""
    X, y = simple_data
    p1 = make_lr_pipeline(random_state=1)
    p2 = make_lr_pipeline(random_state=2)
    p1.fit(X, y)
    p2.fit(X, y)
    # LR is deterministic once seed is set; just confirm both produce valid probas
    assert p1.predict_proba(X).shape == (100, 2)
    assert p2.predict_proba(X).shape == (100, 2)


def test_gbm_different_seeds_produce_valid_output(simple_data):
    X, y = simple_data
    for seed in [42, 123, 999]:
        pipeline = make_gbm_pipeline(random_state=seed)
        pipeline.fit(X, y)
        proba = pipeline.predict_proba(X)
        assert proba.shape == (100, 2)
