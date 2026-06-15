"""Tests for the cross-validation harness."""
import numpy as np
import pytest

from src.evaluate import run_cv
from src.pipeline import make_gb_pipeline, make_lr_pipeline


def _separable_data(n: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    y = (X[:, 0] + rng.standard_normal(n) * 0.3 > 0).astype(int)
    return X, y


def test_run_cv_returns_expected_keys():
    X, y = _separable_data()
    summary = run_cv(make_lr_pipeline, X, y, n_splits=3, n_repeats=2)
    for metric in ("roc_auc", "f1", "accuracy"):
        assert metric in summary
        for key in ("mean", "std", "n", "all"):
            assert key in summary[metric], f"Missing '{key}' in {metric}"


def test_cv_fold_count_matches_splits_times_repeats():
    X, y = _separable_data()
    summary = run_cv(make_lr_pipeline, X, y, n_splits=3, n_repeats=2)
    assert summary["roc_auc"]["n"] == 6  # 3 × 2


def test_cv_auc_above_trivial_baseline():
    """LR on separable data must clearly beat 0.5."""
    X, y = _separable_data()
    summary = run_cv(make_lr_pipeline, X, y, n_splits=3, n_repeats=2)
    assert summary["roc_auc"]["mean"] > 0.6


def test_cv_gb_auc_above_trivial_baseline():
    X, y = _separable_data()
    summary = run_cv(make_gb_pipeline, X, y, n_splits=3, n_repeats=2)
    assert summary["roc_auc"]["mean"] > 0.6


def test_cv_all_scores_length_matches_n():
    X, y = _separable_data()
    for fn in (make_lr_pipeline, make_gb_pipeline):
        summary = run_cv(fn, X, y, n_splits=2, n_repeats=2)
        for metric_data in summary.values():
            assert len(metric_data["all"]) == metric_data["n"]


def test_cv_metrics_in_valid_range():
    X, y = _separable_data()
    summary = run_cv(make_lr_pipeline, X, y, n_splits=3, n_repeats=2)
    for metric, data in summary.items():
        assert 0.0 <= data["mean"] <= 1.0, f"{metric} mean out of [0,1]"
        assert data["std"] >= 0.0, f"{metric} std negative"


def test_cv_reproducible_with_same_seed():
    X, y = _separable_data()
    s1 = run_cv(make_lr_pipeline, X, y, n_splits=3, n_repeats=2, random_state=0)
    s2 = run_cv(make_lr_pipeline, X, y, n_splits=3, n_repeats=2, random_state=0)
    assert s1["roc_auc"]["all"] == s2["roc_auc"]["all"]
