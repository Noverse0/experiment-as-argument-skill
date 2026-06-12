import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.sanity import baseline_floor, overfit_tiny_subset, label_shuffle_test


@pytest.fixture
def simple_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 4))
    y = (X[:, 0] + rng.standard_normal(1000) * 0.3 > 0).astype(int)
    split = 800
    return X[:split], y[:split], X[split:], y[split:]


def test_baseline_floor_returns_dict(simple_data):
    X_tr, y_tr, X_te, y_te = simple_data
    result = baseline_floor(X_tr, y_tr, X_te, y_te)
    assert "majority_class_accuracy" in result
    assert "target_rate_train" in result


def test_baseline_majority_class_accuracy_valid(simple_data):
    X_tr, y_tr, X_te, y_te = simple_data
    result = baseline_floor(X_tr, y_tr, X_te, y_te)
    assert 0.0 <= result["majority_class_accuracy"] <= 1.0


def test_overfit_tiny_subset_passes_for_lr(simple_data):
    X_tr, y_tr, _, _ = simple_data
    lr = LogisticRegression(C=1.0, max_iter=1000)
    result = overfit_tiny_subset(lr, X_tr, y_tr, n=64)
    assert result["passed"]


def test_overfit_tiny_subset_returns_accuracy(simple_data):
    X_tr, y_tr, _, _ = simple_data
    lr = LogisticRegression(C=1.0, max_iter=1000)
    result = overfit_tiny_subset(lr, X_tr, y_tr)
    assert 0.0 <= result["accuracy_on_tiny"] <= 1.0


def test_label_shuffle_reduces_auc(simple_data):
    X_tr, y_tr, X_te, y_te = simple_data
    lr = LogisticRegression(C=1.0, max_iter=1000)
    result = label_shuffle_test(lr, X_tr, y_tr, X_te, y_te)
    assert result["mean_auc_with_shuffled_labels"] < 0.65


def test_label_shuffle_passed_flag(simple_data):
    X_tr, y_tr, X_te, y_te = simple_data
    lr = LogisticRegression(C=1.0, max_iter=1000)
    result = label_shuffle_test(lr, X_tr, y_tr, X_te, y_te)
    assert result["passed"]


def test_label_shuffle_returns_individual_aucs(simple_data):
    X_tr, y_tr, X_te, y_te = simple_data
    lr = LogisticRegression(C=1.0, max_iter=1000)
    result = label_shuffle_test(lr, X_tr, y_tr, X_te, y_te, n_shuffles=3)
    assert len(result["individual_aucs"]) == 3
