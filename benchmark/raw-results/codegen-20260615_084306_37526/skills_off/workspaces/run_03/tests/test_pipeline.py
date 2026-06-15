import numpy as np
import pytest

from src.pipeline import make_lr, make_gb


@pytest.fixture
def small_dataset():
    np.random.seed(42)
    X = np.random.randn(200, 3)
    y = (X[:, 0] + np.random.randn(200) * 0.5 > 0).astype(int)
    return X, y


def test_lr_fits_and_predicts_proba(small_dataset):
    X, y = small_dataset
    pipe = make_lr(seed=42)
    pipe.fit(X[:150], y[:150])
    proba = pipe.predict_proba(X[150:])
    assert proba.shape == (50, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_gb_fits_and_predicts_proba(small_dataset):
    X, y = small_dataset
    pipe = make_gb(seed=42)
    pipe.fit(X[:150], y[:150])
    proba = pipe.predict_proba(X[150:])
    assert proba.shape == (50, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_lr_probabilities_in_unit_interval(small_dataset):
    X, y = small_dataset
    pipe = make_lr(seed=0)
    pipe.fit(X[:150], y[:150])
    proba = pipe.predict_proba(X[150:])[:, 1]
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0


def test_gb_probabilities_in_unit_interval(small_dataset):
    X, y = small_dataset
    pipe = make_gb(seed=0)
    pipe.fit(X[:150], y[:150])
    proba = pipe.predict_proba(X[150:])[:, 1]
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0


def test_pipelines_have_scaler_step(small_dataset):
    X, y = small_dataset
    for make_fn in [make_lr, make_gb]:
        pipe = make_fn(seed=0)
        pipe.fit(X, y)
        assert "scaler" in pipe.named_steps
        assert "clf" in pipe.named_steps


def test_scaler_fitted_on_train_not_test(small_dataset):
    X, y = small_dataset
    pipe = make_lr(seed=42)
    pipe.fit(X[:150], y[:150])
    # Scaler mean should reflect training data statistics, not full data
    train_mean = X[:150].mean(axis=0)
    scaler_mean = pipe.named_steps["scaler"].mean_
    np.testing.assert_allclose(scaler_mean, train_mean, atol=1e-10)
