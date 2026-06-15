import numpy as np
import pytest

from src.evaluate import compute_metrics


def test_perfect_classifier():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.1, 0.9, 1.0])
    m = compute_metrics(y_true, y_prob)
    assert m["roc_auc"] == 1.0


def test_worst_classifier():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([1.0, 0.9, 0.1, 0.0])
    m = compute_metrics(y_true, y_prob)
    assert m["roc_auc"] == 0.0


def test_metrics_keys_present():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.3, 0.7, 0.4, 0.6])
    m = compute_metrics(y_true, y_prob)
    assert set(m.keys()) == {"roc_auc", "avg_precision", "f1"}


def test_all_values_are_floats():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.3, 0.7, 0.4, 0.6])
    m = compute_metrics(y_true, y_prob)
    for v in m.values():
        assert isinstance(v, float)


def test_metrics_in_valid_range():
    np.random.seed(0)
    y_true = np.random.randint(0, 2, 100)
    y_prob = np.random.rand(100)
    m = compute_metrics(y_true, y_prob)
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["avg_precision"] <= 1.0
    assert 0.0 <= m["f1"] <= 1.0


def test_random_classifier_near_chance():
    np.random.seed(99)
    y_true = np.random.randint(0, 2, 500)
    y_prob = np.random.rand(500)
    m = compute_metrics(y_true, y_prob)
    assert 0.4 < m["roc_auc"] < 0.6
