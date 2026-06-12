"""Tests for the leak-prevention pipeline and evaluation. These guard the experiment's
*argument*, not just that code runs: dedup happens, leaks stay out, splits respect time,
and the run is deterministic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data as D
from src.evaluate import evaluate_arms, paired_comparison
from src.models import make_arms
from src import sanity

CSV = "churn.csv"


@pytest.fixture(scope="module")
def loaded():
    return D.load(CSV)


def test_dataset_present():
    import os
    assert os.path.exists(CSV), "run: python3 make_dataset.py --out churn.csv"


def test_duplicates_dropped_before_split(loaded):
    # Generator appends 200 exact duplicates; they must be removed before any split.
    assert loaded.n_duplicates_dropped == 200
    assert loaded.n_clean == loaded.n_raw - 200
    assert not loaded.df.drop(columns=[]).duplicated().any()


def test_leak_and_id_columns_excluded_from_features():
    assert "account_status" not in D.FEATURES
    assert "customer_id" not in D.FEATURES
    assert "signup_date" not in D.FEATURES
    assert "churned" not in D.FEATURES


def test_features_target_only_exposes_allowed_columns(loaded):
    X, y = D.features_target(loaded.df)
    assert list(X.columns) == D.FEATURES
    assert "account_status" not in X.columns
    assert set(np.unique(y)) <= {0, 1}


def test_rows_are_time_sorted(loaded):
    ts = loaded.df[D.TIME_COLUMN]
    assert ts.is_monotonic_increasing


def test_account_status_is_a_perfect_leak(loaded):
    # Documents WHY we drop it: it determines the target exactly.
    df = loaded.df
    closed_churn = df.loc[df["account_status"] == "closed", "churned"]
    active_churn = df.loc[df["account_status"] == "active", "churned"]
    assert (closed_churn == 1).all()
    assert (active_churn == 0).all()


def test_evaluation_is_deterministic(loaded):
    X, y = D.features_target(loaded.df)
    r1 = evaluate_arms(make_arms(42), X, y)
    r2 = evaluate_arms(make_arms(42), X, y)
    for name in r1:
        assert r1[name].roc_auc == pytest.approx(r2[name].roc_auc)


def test_models_beat_baseline_floor(loaded):
    # Honest features must carry real (sub-perfect) signal: AUC clearly above 0.5, below ~0.9.
    X, y = D.features_target(loaded.df)
    res = evaluate_arms(make_arms(42), X, y)
    for name, arm in res.items():
        mean_auc = float(np.mean(arm.roc_auc))
        assert 0.55 < mean_auc < 0.9, f"{name} AUC {mean_auc} is implausible (leak or no signal?)"


def test_sanity_checks_pass(loaded):
    results = sanity.run_all(loaded.df, 42)
    failed = [c["check"] for c in results if not c["passed"]]
    assert not failed, f"sanity checks failed: {failed}"


def test_paired_comparison_shape(loaded):
    X, y = D.features_target(loaded.df)
    res = evaluate_arms(make_arms(42), X, y)
    cmp = paired_comparison(res["gboost"], res["logreg"], "roc_auc")
    assert cmp["n"] == len(res["gboost"].roc_auc)
    assert len(cmp["per_fold_diff"]) == cmp["n"]
    assert isinstance(cmp["detectable_difference"], bool)
