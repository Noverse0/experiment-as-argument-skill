"""Tests for the CV evaluation logic."""
import numpy as np
import pytest

from src.evaluation import evaluate_model, sanity_checks
from src.pipeline import make_lr_pipeline, make_gb_pipeline


@pytest.fixture
def sorted_data():
    """Monotonically-indexed data required by TimeSeriesSplit.

    Uses n=1000 with a strong signal so held-out metrics are stable enough to
    assert on, and shuffled labels reliably yield AUC near 0.5.
    """
    rng = np.random.default_rng(42)
    n = 1000
    X = rng.standard_normal((n, 3))
    # Strong signal: makes real-label held-out AUC reliably high.
    logit = X[:, 0] * 2.0 + X[:, 1] * 1.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def test_evaluate_model_returns_expected_keys(sorted_data):
    X, y = sorted_data
    result = evaluate_model(X, y, make_lr_pipeline, n_splits=3, seeds=[0])
    for key in ("auc_mean", "auc_std", "f1_mean", "f1_std", "n_observations"):
        assert key in result, f"Missing key: {key}"


def test_evaluate_model_n_observations(sorted_data):
    X, y = sorted_data
    result = evaluate_model(X, y, make_lr_pipeline, n_splits=3, seeds=[0, 1])
    assert result["n_observations"] == 3 * 2  # n_splits × n_seeds


def test_evaluate_model_auc_in_range(sorted_data):
    X, y = sorted_data
    result = evaluate_model(X, y, make_lr_pipeline, n_splits=3, seeds=[0])
    assert 0.0 <= result["auc_mean"] <= 1.0
    assert result["auc_std"] >= 0.0


def test_evaluate_model_f1_in_range(sorted_data):
    X, y = sorted_data
    result = evaluate_model(X, y, make_lr_pipeline, n_splits=3, seeds=[0])
    assert 0.0 <= result["f1_mean"] <= 1.0


def test_sanity_checks_keys(sorted_data):
    X, y = sorted_data
    result = sanity_checks(X, y, make_lr_pipeline)
    for key in ("train_auc_full", "shuffled_label_auc", "label_shuffle_ok", "churn_rate"):
        assert key in result


def test_sanity_label_shuffle_degrades_auc(sorted_data):
    X, y = sorted_data
    result = sanity_checks(X, y, make_lr_pipeline)
    # Real-label held-out AUC should be well above baseline.
    assert result["train_auc_full"] > 0.75
    # Shuffled-label held-out AUC must fall near baseline (< 0.65 threshold).
    assert result["shuffled_label_auc"] < 0.65
    assert result["label_shuffle_ok"] is True


def test_gb_evaluate_model_runs(sorted_data):
    X, y = sorted_data
    result = evaluate_model(X, y, make_gb_pipeline, n_splits=2, seeds=[0])
    assert result["n_observations"] == 2
    assert 0.0 <= result["auc_mean"] <= 1.0
