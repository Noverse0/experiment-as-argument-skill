"""Tests for evaluation utilities."""
import numpy as np
import pytest

from src.evaluate import cv_metrics, final_eval
from src.pipeline import make_lr, make_gb


def _make_xy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(n) > 0).astype(int)
    return X, y


def test_cv_metrics_returns_expected_keys():
    X, y = _make_xy()
    pipe = make_lr()
    stats = cv_metrics(pipe, X, y, seeds=[42], cv=3)
    for metric in ["roc_auc", "f1", "precision", "recall"]:
        assert metric in stats
        assert "mean" in stats[metric]
        assert "std" in stats[metric]
        assert "n" in stats[metric]


def test_cv_metrics_values_in_range():
    X, y = _make_xy()
    pipe = make_lr()
    stats = cv_metrics(pipe, X, y, seeds=[42], cv=3)
    for metric in ["roc_auc", "f1", "precision", "recall"]:
        assert 0.0 <= stats[metric]["mean"] <= 1.0
        assert stats[metric]["std"] >= 0.0


def test_cv_metrics_n_equals_seeds_times_folds():
    X, y = _make_xy()
    pipe = make_lr()
    stats = cv_metrics(pipe, X, y, seeds=[1, 2, 3], cv=5)
    assert stats["roc_auc"]["n"] == 15


def test_final_eval_keys():
    X, y = _make_xy()
    n = int(0.8 * len(X))
    pipe = make_lr()
    metrics = final_eval(pipe, X[:n], X[n:], y[:n], y[n:])
    for k in ["roc_auc", "f1", "precision", "recall"]:
        assert k in metrics


def test_final_eval_values_in_range():
    X, y = _make_xy()
    n = int(0.8 * len(X))
    pipe = make_lr()
    metrics = final_eval(pipe, X[:n], X[n:], y[:n], y[n:])
    for v in metrics.values():
        assert 0.0 <= v <= 1.0
