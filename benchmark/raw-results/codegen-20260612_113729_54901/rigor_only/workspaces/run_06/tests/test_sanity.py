"""Tests for pre-experiment sanity checks."""
import numpy as np
import pytest

from src.pipeline import build_gbm_pipeline, build_lr_pipeline
from src.sanity import check_baseline, check_label_shuffle, check_overfit_tiny


@pytest.fixture
def simple_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((300, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    X_train, X_test = X[:200], X[200:]
    y_train, y_test = y[:200], y[200:]
    return X_train, X_test, y_train, y_test


def test_baseline_returns_expected_keys(simple_data):
    X_tr, X_te, y_tr, y_te = simple_data
    result = check_baseline(X_tr, y_tr, X_te, y_te)
    assert "baseline_accuracy" in result
    assert "test_target_rate" in result


def test_baseline_accuracy_in_range(simple_data):
    X_tr, X_te, y_tr, y_te = simple_data
    result = check_baseline(X_tr, y_tr, X_te, y_te)
    assert 0.0 <= result["baseline_accuracy"] <= 1.0


def test_baseline_matches_majority_class(simple_data):
    """Baseline accuracy equals the rate of the training-majority class in the test set."""
    X_tr, X_te, y_tr, y_te = simple_data
    result = check_baseline(X_tr, y_tr, X_te, y_te)
    # DummyClassifier predicts the training-set majority class for every test row
    train_majority = int(y_tr.mean() >= 0.5)
    expected = float((y_te == train_majority).mean())
    assert result["baseline_accuracy"] == pytest.approx(expected, abs=1e-9)


def test_overfit_tiny_lr_high_train_acc(simple_data):
    X_tr, _, y_tr, _ = simple_data
    result = check_overfit_tiny(build_lr_pipeline(0), X_tr, y_tr, n=50)
    assert result["overfit_tiny_train_acc"] > 0.8


def test_overfit_tiny_gbm_high_train_acc(simple_data):
    X_tr, _, y_tr, _ = simple_data
    result = check_overfit_tiny(build_gbm_pipeline(0), X_tr, y_tr, n=50)
    assert result["overfit_tiny_train_acc"] > 0.8


def test_overfit_tiny_returns_n(simple_data):
    X_tr, _, y_tr, _ = simple_data
    result = check_overfit_tiny(build_lr_pipeline(0), X_tr, y_tr, n=30)
    assert result["overfit_n"] == 30


def test_label_shuffle_near_half(simple_data):
    X_tr, X_te, y_tr, y_te = simple_data
    result = check_label_shuffle(
        build_lr_pipeline, X_tr, y_tr, X_te, y_te, n_shuffles=2
    )
    assert abs(result["shuffle_mean_auc"] - 0.5) < 0.15


def test_label_shuffle_returns_n(simple_data):
    X_tr, X_te, y_tr, y_te = simple_data
    result = check_label_shuffle(
        build_lr_pipeline, X_tr, y_tr, X_te, y_te, n_shuffles=2
    )
    assert result["shuffle_n"] == 2
