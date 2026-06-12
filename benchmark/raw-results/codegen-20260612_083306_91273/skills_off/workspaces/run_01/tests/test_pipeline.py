"""Tests for pipeline building and reproducibility."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.pipeline import build_pipeline, make_lr, make_gb
from sklearn.linear_model import LogisticRegression


def _make_xy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(n) > 0).astype(int)
    return X, y


def test_pipeline_has_scaler_and_clf():
    pipe = make_lr()
    assert "scaler" in pipe.named_steps
    assert "clf" in pipe.named_steps


def test_lr_pipeline_fits_and_predicts():
    X, y = _make_xy()
    pipe = make_lr()
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_gb_pipeline_fits_and_predicts():
    X, y = _make_xy()
    pipe = make_gb()
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_same_seed_same_metrics():
    """Identical seed must yield identical metrics (determinism check)."""
    X, y = _make_xy()
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    pipe1 = make_lr(seed=42)
    pipe1.fit(X_train, y_train)
    auc1 = roc_auc_score(y_test, pipe1.predict_proba(X_test)[:, 1])

    pipe2 = make_lr(seed=42)
    pipe2.fit(X_train, y_train)
    auc2 = roc_auc_score(y_test, pipe2.predict_proba(X_test)[:, 1])

    assert auc1 == auc2, f"Same seed should give same AUC: {auc1} vs {auc2}"
