"""Tests for evaluation utilities."""
import numpy as np
import pytest
from sklearn.model_selection import TimeSeriesSplit

from src.evaluate import compute_metrics, run_cv, summarize_runs
from src.pipeline import build_lr_pipeline


@pytest.fixture
def perfect_case():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.05, 0.1, 0.9, 0.95, 0.1, 0.8])
    return y_true, y_pred, y_prob


def test_compute_metrics_perfect(perfect_case):
    y_true, y_pred, y_prob = perfect_case
    m = compute_metrics(y_true, y_pred, y_prob)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)


def test_compute_metrics_has_all_keys(perfect_case):
    y_true, y_pred, y_prob = perfect_case
    m = compute_metrics(y_true, y_pred, y_prob)
    assert set(m.keys()) == {"roc_auc", "f1", "precision", "recall", "accuracy"}


def test_compute_metrics_range(perfect_case):
    y_true, y_pred, y_prob = perfect_case
    m = compute_metrics(y_true, y_pred, y_prob)
    for v in m.values():
        assert 0.0 <= v <= 1.0


def test_summarize_runs_mean():
    runs = [
        {"roc_auc": 0.80, "f1": 0.70},
        {"roc_auc": 0.90, "f1": 0.80},
        {"roc_auc": 0.85, "f1": 0.75},
    ]
    s = summarize_runs(runs)
    assert s["roc_auc_mean"] == pytest.approx(0.85)
    assert s["f1_mean"] == pytest.approx(0.75)
    assert s["roc_auc_n"] == 3
    assert "roc_auc_sd" in s


def test_summarize_runs_sd_zero_for_constant():
    runs = [{"roc_auc": 0.80}] * 5
    s = summarize_runs(runs)
    assert s["roc_auc_sd"] == pytest.approx(0.0, abs=1e-10)


def test_run_cv_returns_one_entry_per_fold():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 4))
    y = (X[:, 0] > 0).astype(int)

    cv = TimeSeriesSplit(n_splits=3)
    runs = run_cv(build_lr_pipeline, X, y, cv, random_state=0)
    assert len(runs) == 3
    for r in runs:
        assert set(r.keys()) == {"roc_auc", "f1", "precision", "recall", "accuracy"}
        assert 0.0 <= r["roc_auc"] <= 1.0


def test_run_cv_reproducible():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 4))
    y = (X[:, 0] > 0).astype(int)

    cv = TimeSeriesSplit(n_splits=3)
    runs1 = run_cv(build_lr_pipeline, X, y, cv, random_state=0)
    runs2 = run_cv(build_lr_pipeline, X, y, cv, random_state=0)
    for r1, r2 in zip(runs1, runs2):
        assert r1["roc_auc"] == pytest.approx(r2["roc_auc"])
