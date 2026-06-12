"""Tests for model pipelines and evaluation utilities."""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import compute_metrics, run_seeds
from src.models import make_pipeline


@pytest.fixture
def small_data():
    X, y = make_classification(
        n_samples=200, n_features=4, n_informative=2, n_redundant=1,
        n_repeated=0, random_state=0
    )
    n_train = 160
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


@pytest.mark.parametrize("model_name", ["logistic", "gbm"])
def test_pipeline_produces_valid_probabilities(small_data, model_name):
    X_train, y_train, X_test, _ = small_data
    pipe = make_pipeline(model_name, seed=0)
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2), "predict_proba must return (n, 2)"
    assert np.allclose(proba.sum(axis=1), 1.0), "probabilities must sum to 1"
    assert proba.min() >= 0.0 and proba.max() <= 1.0, "probabilities must be in [0,1]"


@pytest.mark.parametrize("model_name", ["logistic", "gbm"])
def test_pipeline_auc_above_random_baseline(small_data, model_name):
    X_train, y_train, X_test, y_test = small_data
    pipe = make_pipeline(model_name, seed=0)
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, proba)
    assert metrics["roc_auc"] > 0.5, (
        f"{model_name} AUC={metrics['roc_auc']:.3f} must exceed random baseline (0.5)"
    )


def test_scaler_fitted_on_train_only(small_data):
    """Pipeline scaler mean must equal the training data mean."""
    X_train, y_train, X_test, _ = small_data
    pipe = make_pipeline("logistic", seed=0)
    pipe.fit(X_train, y_train)
    scaler = pipe.named_steps["scaler"]
    np.testing.assert_allclose(
        scaler.mean_,
        X_train.mean(axis=0),
        rtol=1e-5,
        err_msg="Scaler mean must match training data mean (no test leakage)",
    )


def test_gbm_deterministic_with_same_seed(small_data):
    """Identical seeds must produce identical predictions."""
    X_train, y_train, X_test, _ = small_data
    p1 = make_pipeline("gbm", seed=42)
    p2 = make_pipeline("gbm", seed=42)
    p1.fit(X_train, y_train)
    p2.fit(X_train, y_train)
    np.testing.assert_array_equal(
        p1.predict_proba(X_test),
        p2.predict_proba(X_test),
        err_msg="Same seed must produce identical GBM predictions",
    )


def test_gbm_different_seeds_may_differ(small_data):
    """Different seeds should (usually) produce at least slightly different predictions."""
    X_train, y_train, X_test, _ = small_data
    p0 = make_pipeline("gbm", seed=0)
    p1 = make_pipeline("gbm", seed=1)
    p0.fit(X_train, y_train)
    p1.fit(X_train, y_train)
    # subsample=0.8 means different seeds select different subsets — predictions differ
    assert not np.array_equal(
        p0.predict_proba(X_test), p1.predict_proba(X_test)
    ), "Different seeds should produce different GBM predictions (subsample=0.8)"


def test_compute_metrics_keys(small_data):
    X_train, y_train, X_test, y_test = small_data
    pipe = make_pipeline("logistic", seed=0)
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, proba)
    for key in ("roc_auc", "f1", "precision", "recall"):
        assert key in metrics, f"Missing metric: {key}"
        assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]} out of [0,1]"


def test_run_seeds_structure(small_data):
    X_train, y_train, X_test, y_test = small_data
    result = run_seeds(
        lambda seed: make_pipeline("logistic", seed),
        X_train, y_train, X_test, y_test,
        seeds=[0, 1, 2],
    )
    assert result["n_seeds"] == 3
    assert len(result["per_seed"]) == 3
    assert "roc_auc_mean" in result
    assert "roc_auc_std" in result
    assert result["roc_auc_mean"] > 0.5


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        make_pipeline("xgboost", seed=0)
