"""Tests for the sanity check functions."""
import numpy as np
import pytest

from src.sanity import (
    check_label_shuffle,
    check_no_target_leak,
    check_overfit_tiny_subset,
    run_all,
)
from src.models import make_logistic


@pytest.fixture
def simple_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    split = 320
    return X[:split], X[split:], y[:split], y[split:]


def test_no_target_leak_passes_clean_features():
    check_no_target_leak(["tenure_months", "monthly_spend", "support_tickets"])


def test_no_target_leak_fails_on_account_status():
    with pytest.raises(AssertionError, match="account_status"):
        check_no_target_leak(["account_status", "tenure_months"])


def test_overfit_tiny_subset_passes(simple_data):
    X_tr, _, y_tr, _ = simple_data
    model = make_logistic(seed=0)
    check_overfit_tiny_subset(model, X_tr, y_tr, n=64)


def test_overfit_tiny_subset_fails_on_impossibly_high_threshold(simple_data):
    """check_overfit_tiny_subset raises AssertionError when min_auc > 1.0."""
    X_tr, _, y_tr, _ = simple_data
    model = make_logistic(seed=0)
    # min_auc=1.1 is mathematically impossible for any AUC value → always raises
    with pytest.raises(AssertionError):
        check_overfit_tiny_subset(model, X_tr, y_tr, n=64, min_auc=1.1)


def test_label_shuffle_collapses():
    """With pure-noise features (y independent of X), shuffled AUC stays near 0.5."""
    rng = np.random.default_rng(0)
    X_tr = rng.standard_normal((320, 3))
    X_te = rng.standard_normal((80, 3))
    y_tr = rng.integers(0, 2, 320)  # completely random — no signal
    y_te = rng.integers(0, 2, 80)
    model = make_logistic(seed=0)
    auc = check_label_shuffle(model, X_tr, y_tr, X_te, y_te, seed=1, max_auc=0.70)
    assert auc <= 0.70


def test_label_shuffle_raises_on_impossible_threshold():
    """check_label_shuffle raises AssertionError when max_auc is impossibly strict."""
    rng = np.random.default_rng(0)
    X_tr = rng.standard_normal((320, 3))
    X_te = rng.standard_normal((80, 3))
    y_tr = rng.integers(0, 2, 320)
    y_te = rng.integers(0, 2, 80)
    model = make_logistic(seed=0)
    # max_auc=0.0 is always violated — any real AUC > 0 triggers the assertion
    with pytest.raises(AssertionError, match="leakage"):
        check_label_shuffle(model, X_tr, y_tr, X_te, y_te, seed=1, max_auc=0.0)


def test_run_all_passes_on_clean_data(tmp_path):
    """run_all passes every sanity check on the actual generated churn dataset."""
    import subprocess
    from src.data import prepare

    csv = tmp_path / "churn.csv"
    subprocess.run(["python3", "make_dataset.py", "--out", str(csv)], check=True)
    data = prepare(str(csv))
    model = make_logistic(seed=0)
    result = run_all(
        model,
        data["X_train"], data["y_train"],
        data["X_test"], data["y_test"],
        data["feature_names"],
    )
    assert "floor_auc" in result
    assert "shuffle_auc" in result
    assert result["floor_auc"] > 0.52
    assert result["shuffle_auc"] <= 0.65
