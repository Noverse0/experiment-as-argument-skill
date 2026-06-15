"""Tests for pipeline factories."""
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.pipeline import make_gb_pipeline, make_lr_pipeline, MODELS


@pytest.fixture
def small_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 4))
    y = rng.integers(0, 2, 100)
    return X, y


def test_lr_pipeline_is_pipeline():
    pipe = make_lr_pipeline()
    assert isinstance(pipe, Pipeline)


def test_gb_pipeline_is_pipeline():
    pipe = make_gb_pipeline()
    assert isinstance(pipe, Pipeline)


def test_lr_pipeline_fit_predict(small_data):
    X, y = small_data
    pipe = make_lr_pipeline(random_state=0)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (100, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_gb_pipeline_fit_predict(small_data):
    X, y = small_data
    pipe = make_gb_pipeline(random_state=0)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (100, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_different_seeds_give_different_gb_proba(small_data):
    X, y = small_data
    pipe1 = make_gb_pipeline(random_state=0)
    pipe2 = make_gb_pipeline(random_state=99)
    pipe1.fit(X, y)
    pipe2.fit(X, y)
    # Probabilities from different seeds should differ at least slightly.
    assert not np.allclose(pipe1.predict_proba(X), pipe2.predict_proba(X))


def test_models_dict_contains_expected_keys():
    assert "LogisticRegression" in MODELS
    assert "GradientBoosting" in MODELS


def test_models_dict_values_are_callable():
    for name, fn in MODELS.items():
        pipe = fn(random_state=42)
        assert isinstance(pipe, Pipeline), f"{name} factory did not return a Pipeline"
