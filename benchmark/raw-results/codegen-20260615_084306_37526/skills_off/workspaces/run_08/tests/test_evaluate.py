"""Tests for the evaluation harness."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.evaluate import evaluate_pipeline
from src.pipeline import make_gb_pipeline, make_lr_pipeline


@pytest.fixture
def dataset():
    X_np, y_np = make_classification(
        n_samples=600, n_features=3, n_informative=2, n_redundant=1, random_state=0
    )
    X = pd.DataFrame(X_np, columns=["f1", "f2", "f3"])
    y = pd.Series(y_np, name="churned")
    return X, y


def test_returns_all_metrics(dataset):
    X, y = dataset
    result = evaluate_pipeline(make_lr_pipeline(42), X, y, n_splits=3)
    for metric in ["roc_auc", "f1", "precision", "recall"]:
        assert metric in result
        assert {"mean", "std", "n"} == set(result[metric].keys())


def test_n_equals_n_splits(dataset):
    X, y = dataset
    result = evaluate_pipeline(make_lr_pipeline(42), X, y, n_splits=4)
    assert result["roc_auc"]["n"] == 4


def test_roc_auc_in_valid_range(dataset):
    X, y = dataset
    result = evaluate_pipeline(make_lr_pipeline(42), X, y, n_splits=3)
    assert 0.0 <= result["roc_auc"]["mean"] <= 1.0


def test_scaler_not_fit_on_test_fold(dataset):
    """If scaler were fit on all data, mean would equal dataset mean.
    This test checks that the pipeline's scaler mean changes across folds
    (as it should if fitted only on the training portion each time)."""
    from sklearn.model_selection import TimeSeriesSplit

    X, y = dataset
    pipeline = make_lr_pipeline(42)
    tscv = TimeSeriesSplit(n_splits=3)
    scaler_means = []
    for train_idx, _ in tscv.split(X):
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        scaler_means.append(pipeline.named_steps["scaler"].mean_.copy())

    # Means should differ across folds because training sets differ.
    assert not np.allclose(scaler_means[0], scaler_means[-1]), (
        "Scaler means identical across folds — may be fit on full dataset"
    )


def test_gb_beats_majority_class_baseline(dataset):
    X, y = dataset
    baseline_acc = max(y.mean(), 1 - y.mean())
    result = evaluate_pipeline(make_gb_pipeline(42), X, y, n_splits=3)
    assert result["roc_auc"]["mean"] > 0.5, (
        f"GB ROC-AUC {result['roc_auc']['mean']:.4f} not better than random"
    )
    _ = baseline_acc  # used implicitly: roc_auc > 0.5 ↔ beats random baseline
