"""Tests for evaluation metrics and sanity checks."""

import numpy as np
import pytest

from src.evaluate import aggregate_runs, compute_metrics


def test_compute_metrics_perfect():
    y_true = np.array([0, 1, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.8, 0.2])
    m = compute_metrics(y_true, y_proba)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)


def test_compute_metrics_keys():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.4, 0.6, 0.3, 0.7])
    m = compute_metrics(y_true, y_proba)
    assert set(m.keys()) == {"roc_auc", "f1", "precision", "recall"}


def test_compute_metrics_proba_range():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.4, 0.6, 0.3, 0.7])
    m = compute_metrics(y_true, y_proba)
    for v in m.values():
        assert 0.0 <= v <= 1.0


def test_aggregate_runs_mean_std():
    runs = [
        {"roc_auc": 0.80, "f1": 0.70},
        {"roc_auc": 0.82, "f1": 0.72},
        {"roc_auc": 0.81, "f1": 0.71},
    ]
    agg = aggregate_runs(runs)
    assert agg["roc_auc"]["mean"] == pytest.approx(0.81, abs=1e-9)
    assert agg["roc_auc"]["n"] == 3
    assert "std" in agg["roc_auc"]


def test_aggregate_single_run_zero_std():
    runs = [{"roc_auc": 0.75}]
    agg = aggregate_runs(runs)
    assert agg["roc_auc"]["std"] == pytest.approx(0.0)
