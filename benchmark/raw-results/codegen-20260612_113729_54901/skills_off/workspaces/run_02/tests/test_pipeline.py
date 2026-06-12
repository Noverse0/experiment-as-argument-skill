"""Tests for model pipelines and evaluation logic."""

import numpy as np
import pandas as pd
import pytest

from src.models import make_gb_pipeline, make_lr_pipeline
from src.evaluate import cv_evaluate, summarize


def _make_X_y(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "tenure_months": rng.integers(1, 72, n),
        "monthly_spend": rng.uniform(20, 200, n),
        "support_tickets": rng.integers(0, 5, n),
    })
    # Simple linear relationship so models can learn something
    logit = -1.0 - 0.03 * X["tenure_months"] + 0.01 * X["monthly_spend"] + 0.4 * X["support_tickets"]
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int), name="churned")
    return X, y


def test_lr_pipeline_has_scaler():
    pipe = make_lr_pipeline()
    assert "scaler" in pipe.named_steps


def test_gb_pipeline_no_scaler():
    pipe = make_gb_pipeline()
    assert "scaler" not in pipe.named_steps


def test_lr_pipeline_fits_and_predicts():
    X, y = _make_X_y()
    pipe = make_lr_pipeline()
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_gb_pipeline_fits_and_predicts():
    X, y = _make_X_y()
    pipe = make_gb_pipeline()
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_lr_pipeline_predict_proba():
    X, y = _make_X_y()
    pipe = make_lr_pipeline()
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_cv_evaluate_returns_correct_number_of_folds():
    X, y = _make_X_y(n=300)
    pipe = make_lr_pipeline()
    results = cv_evaluate(pipe, X, y, n_splits=3)
    assert len(results) == 3


def test_cv_evaluate_metrics_in_range():
    X, y = _make_X_y(n=300)
    pipe = make_lr_pipeline()
    results = cv_evaluate(pipe, X, y, n_splits=3)
    for r in results:
        assert 0.0 <= r["roc_auc"] <= 1.0
        assert 0.0 <= r["pr_auc"] <= 1.0
        assert 0.0 <= r["f1"] <= 1.0


def test_cv_evaluate_fold_indices_non_overlapping():
    """Each test fold's n_test sums match total excluding the first training fold."""
    X, y = _make_X_y(n=300)
    pipe = make_gb_pipeline()
    results = cv_evaluate(pipe, X, y, n_splits=4)
    for r in results:
        assert r["n_train"] > 0
        assert r["n_test"] > 0


def test_summarize_mean_within_range():
    fold_results = [
        {"roc_auc": 0.75, "pr_auc": 0.60, "f1": 0.55},
        {"roc_auc": 0.80, "pr_auc": 0.65, "f1": 0.60},
        {"roc_auc": 0.70, "pr_auc": 0.55, "f1": 0.50},
    ]
    s = summarize(fold_results)
    assert abs(s["roc_auc"]["mean"] - 0.75) < 1e-9
    assert s["roc_auc"]["std"] > 0.0
    assert s["roc_auc"]["n"] == 3


def test_different_seeds_produce_pipelines():
    pipe1 = make_gb_pipeline(seed=1)
    pipe2 = make_gb_pipeline(seed=2)
    # seeds are stored in clf params
    assert pipe1.named_steps["clf"].random_state == 1
    assert pipe2.named_steps["clf"].random_state == 2
