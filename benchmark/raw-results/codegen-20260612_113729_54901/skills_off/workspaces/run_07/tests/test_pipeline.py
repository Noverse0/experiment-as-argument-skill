"""Tests for model pipelines and evaluation utilities."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluate import cv_metrics, holdout_metrics
from src.pipeline import make_gb_pipeline, make_lr_pipeline


@pytest.fixture
def small_dataset():
    X_arr, y_arr = make_classification(
        n_samples=300, n_features=4, n_informative=3, n_redundant=1, random_state=42
    )
    X = pd.DataFrame(X_arr, columns=["f1", "f2", "f3", "f4"])
    y = pd.Series(y_arr, name="churned")
    return X, y


def test_lr_pipeline_predict_shape(small_dataset):
    X, y = small_dataset
    pipe = make_lr_pipeline(seed=42)
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (len(y),)
    assert set(preds).issubset({0, 1})


def test_gb_pipeline_predict_shape(small_dataset):
    X, y = small_dataset
    pipe = make_gb_pipeline(seed=42)
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (len(y),)
    assert set(preds).issubset({0, 1})


def test_lr_probabilities_sum_to_one(small_dataset):
    X, y = small_dataset
    pipe = make_lr_pipeline(seed=42)
    pipe.fit(X, y)
    probas = pipe.predict_proba(X)
    assert probas.shape == (len(y), 2)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)


def test_gb_probabilities_sum_to_one(small_dataset):
    X, y = small_dataset
    pipe = make_gb_pipeline(seed=42)
    pipe.fit(X, y)
    probas = pipe.predict_proba(X)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)


def test_cv_metrics_keys(small_dataset):
    X, y = small_dataset
    pipe = make_lr_pipeline(seed=42)
    metrics = cv_metrics(pipe, X, y, n_splits=3)
    for key in ("roc_auc", "f1", "precision", "recall"):
        assert key in metrics
        assert "mean" in metrics[key]
        assert "std" in metrics[key]
        assert "scores" in metrics[key]
        assert len(metrics[key]["scores"]) == 3


def test_cv_metrics_roc_auc_in_range(small_dataset):
    X, y = small_dataset
    pipe = make_lr_pipeline(seed=42)
    metrics = cv_metrics(pipe, X, y, n_splits=3)
    assert 0.5 <= metrics["roc_auc"]["mean"] <= 1.0


def test_holdout_metrics_keys(small_dataset):
    X, y = small_dataset
    pipe = make_lr_pipeline(seed=42)
    half = len(X) // 2
    metrics = holdout_metrics(pipe, X.iloc[:half], y.iloc[:half], X.iloc[half:], y.iloc[half:])
    for key in ("roc_auc", "f1", "precision", "recall"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_scaler_fitted_on_train_only(small_dataset):
    """Scaler fitted on a subset should have different stats than one fitted on all data."""
    X, y = small_dataset
    half = len(X) // 2

    pipe_train = make_lr_pipeline(seed=42)
    pipe_full = make_lr_pipeline(seed=42)

    pipe_train.fit(X.iloc[:half], y.iloc[:half])
    pipe_full.fit(X, y)

    mean_train = pipe_train.named_steps["scaler"].mean_
    mean_full = pipe_full.named_steps["scaler"].mean_
    assert not np.allclose(mean_train, mean_full), (
        "Scaler fitted on half the data should differ from scaler fitted on full data"
    )


def test_deterministic_with_same_seed(small_dataset):
    """Same seed → identical predictions."""
    X, y = small_dataset
    half = len(X) // 2

    pipe1 = make_lr_pipeline(seed=42)
    pipe2 = make_lr_pipeline(seed=42)
    pipe1.fit(X.iloc[:half], y.iloc[:half])
    pipe2.fit(X.iloc[:half], y.iloc[:half])

    np.testing.assert_array_equal(pipe1.predict(X.iloc[half:]), pipe2.predict(X.iloc[half:]))
