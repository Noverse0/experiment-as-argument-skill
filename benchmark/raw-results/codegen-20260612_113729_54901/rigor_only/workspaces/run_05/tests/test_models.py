"""Tests for model factories and evaluation."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluate import evaluate, summarize_runs
from src.models import MODEL_FACTORIES, make_gbm, make_logistic


@pytest.fixture
def small_dataset():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    # Linearly separable signal for quick test
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    split = 160
    return X[:split], X[split:], y[:split], y[split:]


def test_model_factories_produce_distinct_seeds():
    m0 = make_logistic(seed=0)
    m1 = make_logistic(seed=1)
    assert m0.random_state != m1.random_state


def test_all_models_fit_and_predict(small_dataset):
    X_tr, X_te, y_tr, y_te = small_dataset
    for name, factory in MODEL_FACTORIES.items():
        model = factory(seed=0)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert preds.shape == (len(X_te),)
        proba = model.predict_proba(X_te)
        assert proba.shape == (len(X_te), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


def test_evaluate_keys(small_dataset):
    X_tr, X_te, y_tr, y_te = small_dataset
    model = make_logistic(seed=0)
    model.fit(X_tr, y_tr)
    metrics = evaluate(model, X_te, y_te)
    assert set(metrics.keys()) == {"roc_auc", "avg_precision", "f1"}
    for v in metrics.values():
        assert 0.0 <= v <= 1.0


def test_evaluate_roc_auc_reasonable(small_dataset):
    X_tr, X_te, y_tr, y_te = small_dataset
    model = make_logistic(seed=0)
    model.fit(X_tr, y_tr)
    metrics = evaluate(model, X_te, y_te)
    assert metrics["roc_auc"] > 0.55, "Logistic should beat chance on a linearly separable signal"


def test_summarize_runs_mean_std():
    runs = [{"roc_auc": 0.70, "f1": 0.60}, {"roc_auc": 0.80, "f1": 0.70}]
    summary = summarize_runs(runs)
    assert abs(summary["roc_auc_mean"] - 0.75) < 1e-9
    assert abs(summary["f1_mean"] - 0.65) < 1e-9
    assert summary["n"] == 2
    assert "roc_auc_std" in summary


def test_summarize_single_run():
    runs = [{"roc_auc": 0.75, "f1": 0.65}]
    summary = summarize_runs(runs)
    assert summary["roc_auc_std"] == 0.0
