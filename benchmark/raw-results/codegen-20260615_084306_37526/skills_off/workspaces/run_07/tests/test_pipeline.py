"""Tests for model pipelines and evaluation helpers."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.pipeline import make_gbt, make_lr


def _make_xy(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "tenure_months": rng.integers(1, 72, n),
        "monthly_spend": rng.gamma(2, 30, n),
        "support_tickets": rng.poisson(1.2, n),
    })
    y = (rng.random(n) < 0.3).astype(int)
    return X, y


def test_lr_fit_predict_shapes():
    X, y = _make_xy()
    pipe = make_lr()
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_gbt_fit_predict_shapes():
    X, y = _make_xy()
    pipe = make_gbt()
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_lr_scaler_fit_on_train_only():
    """Scaler fitted on train; applying to out-of-range test must not raise."""
    X_train, y_train = _make_xy(n=200, seed=0)
    X_test, _ = _make_xy(n=50, seed=99)
    # Push test values well outside training range to confirm no clipping/error
    X_test = X_test * 100
    pipe = make_lr()
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)
    assert proba.shape == (50, 2)


def test_label_shuffle_degrades_lr_auc():
    """Train on shuffled labels; out-of-sample AUC should be near 0.5."""
    rng = np.random.default_rng(1)
    X_tr, y_tr = _make_xy(n=200, seed=1)
    X_te, y_te = _make_xy(n=100, seed=10)
    y_shuffled = rng.permutation(y_tr)
    pipe = make_lr()
    pipe.fit(X_tr, y_shuffled)
    auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
    assert abs(auc - 0.5) < 0.2, f"AUC with shuffled labels: {auc:.3f}"


def test_label_shuffle_degrades_gbt_auc():
    """Train on shuffled labels; out-of-sample AUC should be near 0.5."""
    rng = np.random.default_rng(2)
    X_tr, y_tr = _make_xy(n=200, seed=2)
    X_te, y_te = _make_xy(n=100, seed=20)
    y_shuffled = rng.permutation(y_tr)
    pipe = make_gbt()
    pipe.fit(X_tr, y_shuffled)
    auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
    assert abs(auc - 0.5) < 0.2, f"AUC with shuffled labels: {auc:.3f}"


def test_make_lr_returns_fresh_pipeline():
    """Factory must return independent instances (not shared state)."""
    p1 = make_lr(seed=0)
    p2 = make_lr(seed=0)
    assert p1 is not p2


def test_make_gbt_returns_fresh_pipeline():
    p1 = make_gbt(seed=0)
    p2 = make_gbt(seed=0)
    assert p1 is not p2
