"""Tests for evaluation harness."""
import pytest
import numpy as np
import pandas as pd

from src.pipeline import build_lr_pipeline, build_gb_pipeline
from src.evaluate import evaluate_pipeline, label_shuffle_check


@pytest.fixture
def temporal_data():
    """Sorted data so TimeSeriesSplit is meaningful."""
    rng = np.random.default_rng(42)
    n = 600
    X = pd.DataFrame({
        "tenure_months": rng.integers(1, 72, n),
        "monthly_spend": rng.gamma(2.0, 30.0, n),
        "support_tickets": rng.poisson(1.2, n),
        "signup_days": np.arange(n),  # already sorted
    })
    # Planted signal so AUC > 0.5 is achievable
    logit = -1.2 - 0.03 * X["tenure_months"] + 0.01 * X["monthly_spend"] + 0.45 * X["support_tickets"]
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    return X, y


def test_evaluate_returns_expected_keys(temporal_data):
    X, y = temporal_data
    result = evaluate_pipeline(build_lr_pipeline(), X, y, n_splits=3)
    for key in ("roc_auc", "f1", "pr_auc", "n_folds"):
        assert key in result


def test_evaluate_roc_auc_in_unit_interval(temporal_data):
    X, y = temporal_data
    result = evaluate_pipeline(build_lr_pipeline(), X, y, n_splits=3)
    assert 0.0 <= result["roc_auc"]["mean"] <= 1.0


def test_evaluate_fold_count_matches_request(temporal_data):
    X, y = temporal_data
    result = evaluate_pipeline(build_lr_pipeline(), X, y, n_splits=4)
    assert result["n_folds"] == 4
    assert len(result["roc_auc"]["values"]) == 4


def test_evaluate_std_is_nonnegative(temporal_data):
    X, y = temporal_data
    result = evaluate_pipeline(build_gb_pipeline(), X, y, n_splits=3)
    assert result["roc_auc"]["std"] >= 0.0


def test_signal_beats_noise(temporal_data):
    """With planted signal, LR should score well above 0.5."""
    X, y = temporal_data
    result = evaluate_pipeline(build_lr_pipeline(), X, y, n_splits=3)
    assert result["roc_auc"]["mean"] > 0.6


def test_label_shuffle_degrades_performance(temporal_data):
    """Shuffled labels must fall toward random — confirms no label-free leakage."""
    X, y = temporal_data
    shuffled_auc = label_shuffle_check(build_lr_pipeline(), X, y, seed=0)
    assert shuffled_auc < 0.65, f"Shuffled AUC {shuffled_auc:.3f} too high — possible leakage"
