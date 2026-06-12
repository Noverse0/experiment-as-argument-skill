"""Tests for metric calculations."""

import pytest
import numpy as np

from src.metrics import compute_metrics


def test_compute_metrics_perfect():
    """Test metrics on perfect predictions."""
    y_true = np.array([0, 1, 0, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8])
    y_pred_class = np.array([0, 1, 0, 1])

    metrics = compute_metrics(y_true, y_pred_proba, y_pred_class)

    assert metrics['roc_auc'] == 1.0
    assert metrics['accuracy'] == 1.0
    assert metrics['f1'] == 1.0
    assert metrics['precision'] == 1.0
    assert metrics['recall'] == 1.0


def test_compute_metrics_baseline():
    """Test metrics on baseline (majority class) predictions."""
    y_true = np.array([0, 0, 0, 1])  # 75% negative
    y_pred_proba = np.array([0.1, 0.1, 0.1, 0.1])  # Predict all ~0
    y_pred_class = np.array([0, 0, 0, 0])

    metrics = compute_metrics(y_true, y_pred_proba, y_pred_class)

    assert metrics['accuracy'] == 0.75  # 3/4 correct
    assert metrics['precision'] == 0.0  # No positive predictions
    assert metrics['recall'] == 0.0  # No TP


def test_compute_metrics_all_present():
    """Test that all expected metrics are present."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred_proba = np.array([0.2, 0.8, 0.7, 0.3, 0.6])
    y_pred_class = np.array([0, 1, 1, 0, 1])

    metrics = compute_metrics(y_true, y_pred_proba, y_pred_class)

    required_keys = ['roc_auc', 'f1', 'precision', 'recall', 'accuracy']
    for key in required_keys:
        assert key in metrics
        assert 0 <= metrics[key] <= 1


def test_compute_metrics_values_valid():
    """Test that metrics are in valid ranges."""
    y_true = np.array([0, 1, 1, 0, 1, 0] * 10)
    y_pred_proba = np.random.uniform(0, 1, len(y_true))
    y_pred_class = (y_pred_proba > 0.5).astype(int)

    metrics = compute_metrics(y_true, y_pred_proba, y_pred_class)

    for metric_name, metric_value in metrics.items():
        assert 0 <= metric_value <= 1, f"{metric_name} out of range: {metric_value}"
