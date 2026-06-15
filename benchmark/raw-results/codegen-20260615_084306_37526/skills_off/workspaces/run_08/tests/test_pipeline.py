"""Tests for model pipelines."""
import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.pipeline import make_gb_pipeline, make_lr_pipeline


@pytest.fixture
def dataset():
    X, y = make_classification(
        n_samples=200, n_features=3, n_informative=2, n_redundant=1, random_state=0
    )
    return X, y


def test_lr_predict_shape(dataset):
    X, y = dataset
    p = make_lr_pipeline(seed=42)
    p.fit(X[:160], y[:160])
    preds = p.predict(X[160:])
    probs = p.predict_proba(X[160:])
    assert preds.shape == (40,)
    assert probs.shape == (40, 2)


def test_gb_predict_shape(dataset):
    X, y = dataset
    p = make_gb_pipeline(seed=42)
    p.fit(X[:160], y[:160])
    preds = p.predict(X[160:])
    probs = p.predict_proba(X[160:])
    assert preds.shape == (40,)
    assert probs.shape == (40, 2)


def test_lr_deterministic(dataset):
    X, y = dataset
    p1, p2 = make_lr_pipeline(seed=7), make_lr_pipeline(seed=7)
    p1.fit(X, y)
    p2.fit(X, y)
    np.testing.assert_array_equal(p1.predict(X), p2.predict(X))


def test_gb_deterministic(dataset):
    X, y = dataset
    p1, p2 = make_gb_pipeline(seed=7), make_gb_pipeline(seed=7)
    p1.fit(X, y)
    p2.fit(X, y)
    np.testing.assert_array_equal(p1.predict(X), p2.predict(X))


def test_lr_scaler_fitted_on_train_only(dataset):
    """Scaler mean should reflect only training rows, not full dataset."""
    X, y = dataset
    p = make_lr_pipeline(seed=42)
    p.fit(X[:100], y[:100])
    scaler_mean = p.named_steps["scaler"].mean_
    assert scaler_mean.shape == (3,)
    # Mean computed on first 100 rows should differ from full-dataset mean.
    full_mean = X.mean(axis=0)
    assert not np.allclose(scaler_mean, full_mean), (
        "Scaler mean matches full dataset — scaler may have seen test data"
    )


def test_lr_output_probabilities_sum_to_one(dataset):
    X, y = dataset
    p = make_lr_pipeline(seed=42)
    p.fit(X[:160], y[:160])
    probs = p.predict_proba(X[160:])
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
