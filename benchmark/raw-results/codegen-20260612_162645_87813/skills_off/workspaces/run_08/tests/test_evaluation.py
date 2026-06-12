"""Tests for the evaluation harness and sanity checks."""
from __future__ import annotations

import numpy as np

from src import data as data_mod
from src import evaluation as ev
from src import models as models_mod


def test_evaluate_model_returns_per_fold_metrics(churn_csv):
    prepared = data_mod.prepare(churn_csv)
    res = ev.evaluate_model(
        "gb", models_mod.make_gradient_boosting(), prepared.X, prepared.y, n_splits=3
    )
    assert len(res.folds) == 3
    for f in res.folds:
        assert 0.0 <= f.roc_auc <= 1.0
        assert f.n_train > 0 and f.n_test > 0


def test_determinism_same_seed_same_metrics(churn_csv):
    """Identical pipeline + seed must produce identical metrics on re-run."""
    prepared = data_mod.prepare(churn_csv)
    a = ev.evaluate_model(
        "gb", models_mod.make_gradient_boosting(), prepared.X, prepared.y, n_splits=3
    )
    b = ev.evaluate_model(
        "gb", models_mod.make_gradient_boosting(), prepared.X, prepared.y, n_splits=3
    )
    assert a._vals("roc_auc").tolist() == b._vals("roc_auc").tolist()


def test_baseline_floor_near_half(churn_csv):
    prepared = data_mod.prepare(churn_csv)
    out = ev.baseline_floor(prepared.X, prepared.y, n_splits=3)
    assert out["passed"], out
    assert abs(out["mean_roc_auc"] - 0.5) < 0.05


def test_label_shuffle_collapses_to_chance(churn_csv):
    prepared = data_mod.prepare(churn_csv)
    out = ev.label_shuffle_test(
        models_mod.make_gradient_boosting(), prepared.X, prepared.y, n_splits=3
    )
    assert out["passed"], out


def test_overfit_tiny_subset(churn_csv):
    prepared = data_mod.prepare(churn_csv)
    out = ev.overfit_tiny_subset(
        models_mod.make_gradient_boosting(), prepared.X, prepared.y
    )
    assert out["passed"], out


def test_leakage_ceiling_is_near_perfect(churn_csv):
    """Including account_status should drive AUC to ~1.0 -> confirms the leak."""
    X_leaky, y = data_mod.leaky_features(churn_csv)
    out = ev.leakage_ceiling(
        models_mod.make_gradient_boosting(), X_leaky, y, n_splits=3
    )
    assert out["passed"], out
    assert out["mean_roc_auc"] > 0.98


def test_paired_comparison_shape(churn_csv):
    prepared = data_mod.prepare(churn_csv)
    arms = {
        n: ev.evaluate_model(n, p, prepared.X, prepared.y, n_splits=3)
        for n, p in models_mod.make_models().items()
    }
    comp = ev.paired_auc_per_fold(arms)
    assert len(comp["per_fold_diff"]) == 3
    assert "mean_diff" in comp and "sd_diff" in comp
