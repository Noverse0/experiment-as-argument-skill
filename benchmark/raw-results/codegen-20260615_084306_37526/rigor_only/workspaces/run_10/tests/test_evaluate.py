"""Unit tests for cross-validation and sanity-check utilities."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.evaluate import label_shuffle_auc, run_cv


@pytest.fixture(scope="module")
def separable_data():
    """Well-separated classification dataset (LR should easily beat baseline)."""
    X, y = make_classification(
        n_samples=600,
        n_features=5,
        n_informative=3,
        random_state=42,
    )
    return X, y


# ── run_cv ────────────────────────────────────────────────────────────────────

def test_run_cv_returns_expected_keys(separable_data):
    X, y = separable_data
    result = run_cv(X, y, LogisticRegression(max_iter=1000), n_splits=3)
    for key in ("roc_auc_mean", "roc_auc_std", "f1_mean", "f1_std", "n_splits"):
        assert key in result, f"Missing key: {key}"


def test_run_cv_n_splits_recorded(separable_data):
    X, y = separable_data
    result = run_cv(X, y, LogisticRegression(max_iter=1000), n_splits=3)
    assert result["n_splits"] == 3


def test_run_cv_auc_bounds(separable_data):
    X, y = separable_data
    result = run_cv(X, y, LogisticRegression(max_iter=1000, random_state=0), n_splits=3)
    assert 0.0 <= result["roc_auc_mean"] <= 1.0
    assert result["roc_auc_std"] >= 0.0


def test_run_cv_lr_beats_baseline(separable_data):
    X, y = separable_data
    baseline = run_cv(X, y, DummyClassifier(strategy="most_frequent"), n_splits=3)
    lr = run_cv(X, y, LogisticRegression(max_iter=1000, random_state=0), n_splits=3)
    assert lr["roc_auc_mean"] > baseline["roc_auc_mean"]


def test_run_cv_std_non_negative(separable_data):
    X, y = separable_data
    result = run_cv(X, y, LogisticRegression(max_iter=1000), n_splits=3)
    assert result["roc_auc_std"] >= 0.0
    assert result["f1_std"] >= 0.0


# ── label_shuffle_auc ─────────────────────────────────────────────────────────

def test_label_shuffle_degrades_to_near_baseline(separable_data):
    """Shuffled labels must destroy signal; AUC should be near 0.5."""
    X, y = separable_data
    auc = label_shuffle_auc(
        X, y, LogisticRegression(max_iter=1000, random_state=0), n_splits=3
    )
    assert auc < 0.60, (
        f"Label-shuffle AUC {auc:.3f} is too high — possible leakage in features"
    )


def test_label_shuffle_lower_than_real(separable_data):
    """Shuffled-label AUC must be below the real AUC on a meaningful dataset."""
    X, y = separable_data
    real_auc = run_cv(
        X, y, LogisticRegression(max_iter=1000, random_state=0), n_splits=3
    )["roc_auc_mean"]
    shuffled_auc = label_shuffle_auc(
        X, y, LogisticRegression(max_iter=1000, random_state=0), n_splits=3
    )
    assert shuffled_auc < real_auc
