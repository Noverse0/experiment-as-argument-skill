"""Tests for sanity-check functions."""
import numpy as np
import pandas as pd
import pytest

from src.models import make_gradient_boosting, make_logistic_regression
from src.sanity import (
    check_baseline,
    check_class_balance,
    check_label_shuffle,
    check_overfit_tiny,
)


@pytest.fixture
def medium_data():
    rng = np.random.default_rng(99)
    n = 300
    X = pd.DataFrame({
        "tenure_months": rng.integers(1, 72, n),
        "monthly_spend": rng.uniform(10, 200, n),
        "support_tickets": rng.integers(0, 5, n),
        "signup_month": rng.integers(1, 13, n),
    })
    logit = -0.5 + 0.01 * X["monthly_spend"] + 0.3 * X["support_tickets"]
    p = 1 / (1 + np.exp(-logit.values))
    y = pd.Series((rng.random(n) < p).astype(int))
    return X, y


def test_class_balance_keys(medium_data):
    X, y = medium_data
    b = check_class_balance(y)
    assert {"n", "n_positive", "positive_rate"} == set(b.keys())
    assert b["n"] == len(y)
    assert b["n_positive"] == int(y.sum())
    assert 0.0 <= b["positive_rate"] <= 1.0


def test_baseline_auc_is_float_in_range(medium_data):
    X, y = medium_data
    auc = check_baseline(X, y, n_splits=3)
    assert isinstance(auc, float)
    assert 0.0 <= auc <= 1.0


def test_overfit_tiny_lr_high_accuracy(medium_data):
    X, y = medium_data
    acc = check_overfit_tiny(X, y, make_logistic_regression(), n_samples=30)
    assert acc >= 0.8


def test_overfit_tiny_gbm_high_accuracy(medium_data):
    X, y = medium_data
    acc = check_overfit_tiny(X, y, make_gradient_boosting(), n_samples=30)
    assert acc >= 0.8


def test_label_shuffle_near_chance(medium_data):
    X, y = medium_data
    auc = check_label_shuffle(X, y, make_logistic_regression(), n_splits=3)
    # Shuffled labels → no real signal → AUC should be near 0.5
    assert abs(auc - 0.5) < 0.15


def test_label_shuffle_returns_float(medium_data):
    X, y = medium_data
    auc = check_label_shuffle(X, y, make_gradient_boosting(), n_splits=3)
    assert isinstance(auc, float)
    assert 0.0 <= auc <= 1.0
