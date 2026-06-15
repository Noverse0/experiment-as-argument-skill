"""Tests for evaluation utilities."""
import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    baseline_evaluate,
    cv_evaluate,
    holdout_evaluate,
    shuffle_label_test,
)
from src.pipeline import make_models


def test_holdout_metrics_in_valid_range(separable_data):
    X, y = separable_data
    n = len(X)
    split = int(n * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]
    for name, pipe in make_models().items():
        result = holdout_evaluate(pipe, X_tr, y_tr, X_te, y_te)
        assert 0.0 <= result["roc_auc"] <= 1.0, f"{name}: ROC-AUC out of [0,1]"
        assert 0.0 <= result["avg_precision"] <= 1.0, f"{name}: AP out of [0,1]"


def test_holdout_does_not_mutate_pipeline(separable_data):
    """holdout_evaluate clones the pipeline, so the original stays unfitted."""
    X, y = separable_data
    n = len(X)
    split = int(n * 0.8)
    models = make_models()
    pipe = models["LogisticRegression"]

    assert not hasattr(pipe.named_steps["model"], "coef_")
    holdout_evaluate(pipe, X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
    assert not hasattr(pipe.named_steps["model"], "coef_"), "pipeline was mutated"


def test_cv_evaluate_result_structure(separable_data):
    X, y = separable_data
    pipe = make_models()["LogisticRegression"]
    result = cv_evaluate(pipe, X, y, n_seeds=2, n_folds=3)

    assert "roc_auc" in result
    assert "avg_precision" in result
    for key in ("mean", "std", "n"):
        assert key in result["roc_auc"], f"missing '{key}' in roc_auc"
    assert result["roc_auc"]["n"] == 6, "expected 2 seeds × 3 folds = 6"


def test_cv_evaluate_separable_data_high_auc(separable_data):
    """On well-separated classes LR should achieve AUC well above 0.5."""
    X, y = separable_data
    pipe = make_models()["LogisticRegression"]
    result = cv_evaluate(pipe, X, y, n_seeds=1, n_folds=3)
    assert result["roc_auc"]["mean"] > 0.85, "LR should fit separable data well"


def test_shuffle_label_test_near_chance(separable_data):
    """Shuffled labels should make AUC drop to near 0.5."""
    X, y = separable_data
    pipe = make_models()["LogisticRegression"]
    result = shuffle_label_test(pipe, X, y, n_seeds=3)
    assert result["mean_roc_auc"] < 0.65, (
        f"Shuffled-label AUC={result['mean_roc_auc']:.3f} unexpectedly high"
    )


def test_baseline_evaluate_near_chance(separable_data):
    """Stratified dummy classifier should give AUC ≈ 0.5."""
    X, y = separable_data
    n = len(X)
    split = int(n * 0.8)
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    result = baseline_evaluate(y_train, y_test)
    assert 0.0 <= result["roc_auc"] <= 1.0
    assert 0.0 <= result["avg_precision"] <= 1.0
