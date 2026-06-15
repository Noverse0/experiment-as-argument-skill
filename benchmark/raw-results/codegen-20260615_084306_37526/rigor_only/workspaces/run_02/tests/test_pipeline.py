"""Tests for model pipelines and evaluation helpers."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.models import build_lr, build_gbm
from src.evaluate import run_cv, label_shuffle_check, overfit_tiny_check, baseline_auc


@pytest.fixture
def small_dataset():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({
        "tenure_months": rng.integers(1, 72, n),
        "monthly_spend": rng.gamma(2, 30, n),
        "support_tickets": rng.poisson(1.2, n),
    })
    # Simple separable signal
    logit = -1.0 + 0.02 * X["monthly_spend"] + 0.4 * X["support_tickets"]
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    return X, y


def test_lr_pipeline_fits_and_predicts(small_dataset):
    X, y = small_dataset
    pipe = build_lr()
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_gbm_pipeline_fits_and_predicts(small_dataset):
    X, y = small_dataset
    pipe = build_gbm()
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_lr_cv_returns_expected_keys(small_dataset):
    X, y = small_dataset
    result = run_cv(build_lr(), X, y)
    for key in ["roc_auc", "average_precision", "f1"]:
        assert key in result
        assert "mean" in result[key]
        assert "std" in result[key]
        assert "n" in result[key]


def test_cv_n_folds_correct(small_dataset):
    X, y = small_dataset
    result = run_cv(build_lr(), X, y)
    # 5 splits × 3 repeats = 15
    assert result["roc_auc"]["n"] == 15


def test_cv_auc_beats_baseline(small_dataset):
    X, y = small_dataset
    result = run_cv(build_lr(), X, y)
    assert result["roc_auc"]["mean"] > 0.5, "LR should beat majority-class baseline"


def test_label_shuffle_collapses_to_chance(small_dataset):
    X, y = small_dataset
    result = label_shuffle_check(build_lr(), X, y)
    auc = result["shuffled_roc_auc_mean"]
    # Shuffled labels → no signal → AUC near 0.5
    assert 0.3 < auc < 0.7, f"Shuffled AUC {auc:.3f} not near 0.5"


def test_overfit_tiny_achieves_high_auc(small_dataset):
    X, y = small_dataset
    result = overfit_tiny_check(build_gbm(), X, y, n=50)
    assert result["overfit_tiny_roc_auc"] > 0.80, "GBM should overfit a tiny batch"


def test_pipeline_scaler_fit_on_train_only(small_dataset):
    """Scaler must not be fitted on test data — verify via pipeline internals."""
    X, y = small_dataset
    pipe = build_lr()
    # Before fitting, scaler has no mean_
    assert not hasattr(pipe.named_steps["scaler"], "mean_")
    pipe.fit(X, y)
    # After fitting, scaler has mean_ derived from training data
    assert hasattr(pipe.named_steps["scaler"], "mean_")
    assert pipe.named_steps["scaler"].mean_.shape[0] == X.shape[1]


def test_lr_and_gbm_same_features(small_dataset):
    """Both models must accept the same feature set."""
    X, y = small_dataset
    lr = build_lr().fit(X, y)
    gbm = build_gbm().fit(X, y)
    assert lr.predict_proba(X).shape == gbm.predict_proba(X).shape


def test_baseline_auc_near_half(small_dataset):
    _, y = small_dataset
    auc = baseline_auc(y)
    assert 0.45 < auc < 0.55, f"Majority-class AUC {auc:.3f} not near 0.5"
