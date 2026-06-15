"""Tests for model pipelines and evaluation utilities."""
import numpy as np
import pytest

from src.evaluate import label_shuffle_auc, majority_baseline, overfit_check, score
from src.pipeline import make_gb_pipeline, make_lr_pipeline


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(42)
    n = 300
    X = rng.standard_normal((n, 3))
    # Add a real signal so models have something to learn.
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(n) * 0.5 > 0).astype(int)
    return X, y


def test_lr_pipeline_fits_and_predicts(binary_data):
    X, y = binary_data
    clf = make_lr_pipeline(random_state=0)
    clf.fit(X[:200], y[:200])
    preds = clf.predict(X[200:])
    proba = clf.predict_proba(X[200:])
    assert preds.shape == (100,)
    assert proba.shape == (100, 2)
    assert set(preds).issubset({0, 1})


def test_gb_pipeline_fits_and_predicts(binary_data):
    X, y = binary_data
    clf = make_gb_pipeline(random_state=0)
    clf.fit(X[:200], y[:200])
    preds = clf.predict(X[200:])
    proba = clf.predict_proba(X[200:])
    assert preds.shape == (100,)
    assert proba.shape == (100, 2)
    assert set(preds).issubset({0, 1})


def test_score_returns_expected_keys(binary_data):
    X, y = binary_data
    clf = make_lr_pipeline(random_state=0)
    clf.fit(X[:200], y[:200])
    metrics = score(clf, X[200:], y[200:])
    assert set(metrics.keys()) == {"roc_auc", "f1", "precision", "recall"}
    for v in metrics.values():
        assert 0.0 <= v <= 1.0


def test_label_shuffle_auc_near_chance(binary_data):
    X, y = binary_data
    auc = label_shuffle_auc(make_lr_pipeline, X[:200], y[:200], X[200:], y[200:])
    assert auc < 0.65, f"Label-shuffle AUC {auc:.3f} too high — likely a leak in the test"


def test_overfit_check_above_majority(binary_data):
    X, y = binary_data
    majority_acc = max(y.mean(), 1 - y.mean())
    acc = overfit_check(make_gb_pipeline, X, y, seed=0, n=50)
    assert acc > majority_acc, (
        f"GB cannot overfit 50 samples (acc={acc:.3f}, majority={majority_acc:.3f})"
    )


def test_majority_baseline_auc_near_half(binary_data):
    X, y = binary_data
    metrics = majority_baseline(X[:200], y[:200], X[200:], y[200:])
    assert metrics["roc_auc"] < 0.6, f"Baseline AUC {metrics['roc_auc']:.3f} unexpectedly high"


def test_different_seeds_produce_same_lr_auc(binary_data):
    """LR with same data but different seeds should give near-identical AUC."""
    X, y = binary_data
    aucs = []
    for seed in [0, 1, 2]:
        clf = make_lr_pipeline(random_state=seed)
        clf.fit(X[:200], y[:200])
        m = score(clf, X[200:], y[200:])
        aucs.append(m["roc_auc"])
    # LR is nearly deterministic — variance should be tiny.
    assert max(aucs) - min(aucs) < 0.02, f"LR AUC variance too high across seeds: {aucs}"
