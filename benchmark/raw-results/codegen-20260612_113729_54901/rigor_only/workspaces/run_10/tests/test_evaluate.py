"""Tests for the cross-validation and summarization utilities."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluate import evaluate_model, summarize
from pipeline import make_gb_pipeline, make_lr_pipeline


# ---------------------------------------------------------------------------
# Fixture: small dataset with a learnable signal
# ---------------------------------------------------------------------------

@pytest.fixture
def learnable_xy():
    rng = np.random.default_rng(0)
    n = 400
    tenure = rng.integers(1, 72, n)
    spend = rng.gamma(2.0, 30.0, n)
    tickets = rng.poisson(1.2, n)
    logit = -1.2 - 0.03 * tenure + 0.01 * spend + 0.45 * tickets
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({
        "tenure_months": tenure,
        "monthly_spend": spend.round(2),
        "support_tickets": tickets,
        "signup_age_days": np.arange(n),  # ascending = temporal order
    })
    y = pd.Series(churned)
    return X, y


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

def test_evaluate_returns_required_metric_keys(learnable_xy):
    X, y = learnable_xy
    result = evaluate_model(make_lr_pipeline(), X, y, n_splits=3)
    assert set(result.keys()) == {"roc_auc", "f1", "accuracy"}


def test_evaluate_correct_fold_count(learnable_xy):
    X, y = learnable_xy
    result = evaluate_model(make_lr_pipeline(), X, y, n_splits=4)
    assert len(result["roc_auc"]) == 4
    assert len(result["f1"]) == 4


def test_evaluate_auc_above_chance(learnable_xy):
    """Informative features → mean AUC well above 0.5."""
    X, y = learnable_xy
    result = evaluate_model(make_lr_pipeline(), X, y, n_splits=3)
    assert result["roc_auc"].mean() > 0.55


@pytest.mark.parametrize("factory", [make_lr_pipeline, make_gb_pipeline])
def test_evaluate_auc_in_valid_range(learnable_xy, factory):
    X, y = learnable_xy
    result = evaluate_model(factory(), X, y, n_splits=3)
    assert (result["roc_auc"] >= 0).all() and (result["roc_auc"] <= 1).all()


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def test_summarize_keys_present(learnable_xy):
    X, y = learnable_xy
    raw = evaluate_model(make_lr_pipeline(), X, y, n_splits=3)
    s = summarize(raw)
    for metric in ("roc_auc", "f1", "accuracy"):
        assert metric in s
        assert {"mean", "std", "n", "values"} == set(s[metric].keys())


def test_summarize_mean_consistent_with_values(learnable_xy):
    X, y = learnable_xy
    raw = evaluate_model(make_lr_pipeline(), X, y, n_splits=3)
    s = summarize(raw)
    for metric in ("roc_auc", "f1", "accuracy"):
        assert abs(s[metric]["mean"] - np.mean(s[metric]["values"])) < 1e-9


def test_summarize_n_equals_n_splits(learnable_xy):
    X, y = learnable_xy
    raw = evaluate_model(make_lr_pipeline(), X, y, n_splits=3)
    s = summarize(raw)
    assert s["roc_auc"]["n"] == 3
