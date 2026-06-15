"""Tests for model pipeline factories."""
import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.pipeline import make_gb_pipeline, make_lr_pipeline


@pytest.fixture
def small_dataset():
    X, y = make_classification(
        n_samples=200,
        n_features=3,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        random_state=0,
    )
    return X, y


def test_lr_pipeline_fits_and_predicts(small_dataset):
    X, y = small_dataset
    pipe = make_lr_pipeline()
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (200, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_gb_pipeline_fits_and_predicts(small_dataset):
    X, y = small_dataset
    pipe = make_gb_pipeline()
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (200, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_lr_pipeline_has_scaler():
    pipe = make_lr_pipeline()
    assert "scaler" in pipe.named_steps


def test_gb_pipeline_has_no_scaler():
    pipe = make_gb_pipeline()
    assert "scaler" not in pipe.named_steps


def test_lr_pipeline_deterministic(small_dataset):
    X, y = small_dataset
    p1 = make_lr_pipeline(random_state=7)
    p2 = make_lr_pipeline(random_state=7)
    p1.fit(X, y); p2.fit(X, y)
    np.testing.assert_array_equal(p1.predict(X), p2.predict(X))


def test_gb_pipeline_deterministic(small_dataset):
    X, y = small_dataset
    p1 = make_gb_pipeline(random_state=7)
    p2 = make_gb_pipeline(random_state=7)
    p1.fit(X, y); p2.fit(X, y)
    np.testing.assert_array_equal(p1.predict(X), p2.predict(X))
