"""Tests for cross-validation evaluation utilities."""
import numpy as np
import pandas as pd
import pytest

from src.evaluate import cv_evaluate
from src.pipeline import make_lr_pipeline


@pytest.fixture
def balanced_data():
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({
        "tenure_months": rng.integers(1, 72, n),
        "monthly_spend": rng.gamma(2, 30, n),
        "support_tickets": rng.poisson(1.2, n),
        "signup_year": np.full(n, 2023),
        "signup_month": rng.integers(1, 13, n),
        "signup_dayofyear": rng.integers(1, 366, n),
    })
    y = pd.Series([i % 2 for i in range(n)])
    return X, y


def test_cv_evaluate_returns_expected_metrics(balanced_data):
    X, y = balanced_data
    result = cv_evaluate(make_lr_pipeline(), X, y, n_splits=3)
    for metric in ("roc_auc", "avg_precision", "f1"):
        assert metric in result


def test_cv_evaluate_metric_values_in_range(balanced_data):
    X, y = balanced_data
    result = cv_evaluate(make_lr_pipeline(), X, y, n_splits=3)
    for metric in ("roc_auc", "avg_precision", "f1"):
        assert 0.0 <= result[metric]["mean"] <= 1.0
        assert result[metric]["std"] >= 0.0


def test_cv_evaluate_n_folds_recorded(balanced_data):
    X, y = balanced_data
    result = cv_evaluate(make_lr_pipeline(), X, y, n_splits=4)
    assert result["roc_auc"]["n_folds"] == 4
    assert len(result["roc_auc"]["per_fold"]) == 4


def test_cv_evaluate_per_fold_matches_mean(balanced_data):
    X, y = balanced_data
    result = cv_evaluate(make_lr_pipeline(), X, y, n_splits=3)
    per_fold = result["roc_auc"]["per_fold"]
    np.testing.assert_allclose(
        np.mean(per_fold), result["roc_auc"]["mean"], atol=1e-6
    )


def test_cv_evaluate_reproducible(balanced_data):
    X, y = balanced_data
    r1 = cv_evaluate(make_lr_pipeline(), X, y, n_splits=3, random_state=7)
    r2 = cv_evaluate(make_lr_pipeline(), X, y, n_splits=3, random_state=7)
    assert r1["roc_auc"]["mean"] == r2["roc_auc"]["mean"]
