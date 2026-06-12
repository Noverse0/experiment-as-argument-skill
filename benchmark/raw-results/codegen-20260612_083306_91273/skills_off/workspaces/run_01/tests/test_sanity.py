"""Tests for sanity check functions."""
import numpy as np
import pytest

from src.sanity import baseline_floor, leakage_ceiling_check, label_shuffle_test
from src.pipeline import make_lr


def _make_binary_labels(n=200, churn_rate=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice([0, 1], size=n, p=[1 - churn_rate, churn_rate])


def _make_xy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] + rng.standard_normal(n) > 0).astype(int)
    return X, y


def test_baseline_floor_returns_float():
    y = _make_binary_labels(n=200, churn_rate=0.3)
    result = baseline_floor(y)
    assert isinstance(result, float)


def test_baseline_floor_near_half():
    # Majority class classifier has AUC ≈ 0.5 when minority class is < 50%
    y = _make_binary_labels(n=500, churn_rate=0.3)
    auc = baseline_floor(y)
    assert 0.45 <= auc <= 0.75


def test_leakage_ceiling_check_no_warning_below_threshold():
    assert leakage_ceiling_check(0.85, threshold=0.98) is False


def test_leakage_ceiling_check_warns_above_threshold():
    assert leakage_ceiling_check(0.99, threshold=0.98) is True


def test_label_shuffle_degrades_performance():
    """With shuffled labels, the model should not achieve meaningful discrimination."""
    X, y = _make_xy(n=300)
    pipe = make_lr(seed=0)
    shuffle_auc = label_shuffle_test(pipe, X, y, seed=0)
    # AUC must be near chance on the training data itself when labels are random
    # For a well-behaved model, it should be ≤ 0.70 (not perfectly discriminative)
    assert shuffle_auc <= 0.70, f"Shuffle AUC too high: {shuffle_auc}"
