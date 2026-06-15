"""Tests for pipeline construction and no-leakage property."""
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.pipeline import make_gb_pipeline, make_lr_pipeline


def _small_data(n: int = 120, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    y = (X[:, 0] + rng.standard_normal(n) * 0.5 > 0).astype(int)
    return X, y


def test_lr_pipeline_returns_pipeline():
    assert isinstance(make_lr_pipeline(), Pipeline)


def test_gb_pipeline_returns_pipeline():
    assert isinstance(make_gb_pipeline(), Pipeline)


def test_lr_pipeline_fits_and_predicts():
    X, y = _small_data()
    pipe = make_lr_pipeline()
    pipe.fit(X, y)
    preds = pipe.predict(X)
    proba = pipe.predict_proba(X)
    assert preds.shape == (len(y),)
    assert proba.shape == (len(y), 2)
    assert set(preds).issubset({0, 1})


def test_gb_pipeline_fits_and_predicts():
    X, y = _small_data()
    pipe = make_gb_pipeline()
    pipe.fit(X, y)
    preds = pipe.predict(X)
    proba = pipe.predict_proba(X)
    assert preds.shape == (len(y),)
    assert proba.shape == (len(y), 2)


def test_pipelines_are_independent():
    """Fitting LR must not change GB predictions."""
    X, y = _small_data()
    lr = make_lr_pipeline()
    gb = make_gb_pipeline()
    gb.fit(X, y)
    gb_pred_before = gb.predict_proba(X).copy()
    lr.fit(X, y)
    gb_pred_after = gb.predict_proba(X)
    np.testing.assert_array_equal(gb_pred_before, gb_pred_after)


def test_scaler_fitted_on_train_split_only():
    """StandardScaler inside LR pipeline must not see test rows."""
    X, y = _small_data(200)
    pipe = make_lr_pipeline()
    X_train, X_test = X[:100], X[100:]
    y_train = y[:100]
    pipe.fit(X_train, y_train)
    scaler = pipe.named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0), rtol=1e-10)
    # Full-data mean must differ (would be equal if scaler saw everything)
    full_mean = X.mean(axis=0)
    assert not np.allclose(scaler.mean_, full_mean), (
        "Scaler mean matches full dataset — possible leakage"
    )


def test_random_state_parameter_accepted():
    for fn in (make_lr_pipeline, make_gb_pipeline):
        pipe = fn(random_state=0)
        assert isinstance(pipe, Pipeline)
