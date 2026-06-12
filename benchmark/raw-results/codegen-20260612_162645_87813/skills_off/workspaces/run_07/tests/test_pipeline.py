"""Tests for the churn experiment pipeline.

These guard the rigor properties, not just "does it run": leak exclusion,
dedup-before-split, time ordering, and the sanity-check behaviors.
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC))

from churn_experiment.data import FEATURES, leakage_audit, prepare
from churn_experiment.evaluate import paired_difference, time_series_cv
from churn_experiment.models import make_models
from churn_experiment import sanity


def _raw(n=400, dups=20, seed=0):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    spend = rng.gamma(2.0, 30.0, n).round(2)
    tickets = rng.poisson(1.2, n)
    signup = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")
    logit = -1.2 - 0.03 * tenure + 0.01 * spend + 0.45 * tickets
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "signup_date": signup.strftime("%Y-%m-%d"),
            "tenure_months": tenure,
            "monthly_spend": spend,
            "support_tickets": tickets,
            "account_status": np.where(churn == 1, "closed", "active"),
            "churned": churn,
        }
    )
    dup = df.sample(n=dups, random_state=seed)
    df = pd.concat([df, dup], ignore_index=True)
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    return df


def test_leakage_audit_detects_planted_leak():
    df = _raw()
    assert leakage_audit(df)["account_status_leak_fraction"] == pytest.approx(1.0)


def test_prepare_drops_leak_and_id_columns():
    X = prepare(_raw()).X
    assert list(X.columns) == FEATURES
    assert "account_status" not in X.columns
    assert "customer_id" not in X.columns


def test_prepare_dedups_before_split():
    p = prepare(_raw(n=400, dups=20))
    assert p.n_duplicates_dropped == 20
    assert len(p.X) == 400


def test_prepare_is_time_sorted():
    p = prepare(_raw())
    assert p.time.is_monotonic_increasing


def test_models_present_and_distinct():
    m = make_models(0)
    assert set(m) == {"logistic_regression", "gradient_boosting"}


def test_cv_returns_per_fold_metrics():
    p = prepare(_raw())
    fac = lambda: make_models(0)["logistic_regression"]
    fac.name = "logistic_regression"
    res = time_series_cv(fac, p.X, p.y, n_splits=4)
    assert len(res.per_fold) == 4
    ms = res.mean_std()
    assert 0.0 <= ms["roc_auc"]["mean"] <= 1.0


def test_real_model_beats_chance():
    p = prepare(_raw(n=1200, seed=3))
    fac = lambda: make_models(0)["gradient_boosting"]
    fac.name = "gb"
    auc = time_series_cv(fac, p.X, p.y, n_splits=4).mean_std()["roc_auc"]["mean"]
    assert auc > 0.55  # there is real (if weak) signal


def test_label_shuffle_collapses_to_chance():
    """The core anti-leakage check: shuffled labels => AUC ~ 0.5."""
    p = prepare(_raw(n=1200, seed=3))
    fac = lambda: make_models(0)["gradient_boosting"]
    fac.name = "gb"
    res = sanity.check_label_shuffle(fac, p, n_splits=4)
    assert res["passed"]
    assert res["shuffled_roc_auc"] < 0.6


def test_sanity_leak_excluded_passes():
    df = _raw()
    assert sanity.check_leak_excluded(df, prepare(df))["passed"]


def test_baseline_floor_is_chance():
    p = prepare(_raw(n=1200, seed=3))
    assert sanity.check_baseline_floor(p, n_splits=4)["passed"]


def test_overfit_tiny_slice():
    p = prepare(_raw(n=400))
    fac = lambda: make_models(0)["gradient_boosting"]
    fac.name = "gb"
    assert sanity.check_overfit_tiny(fac, p, n=60)["passed"]


def test_paired_difference_zero_when_identical():
    p = prepare(_raw(n=800, seed=1))
    fac = lambda: make_models(0)["logistic_regression"]
    fac.name = "lr"
    a = time_series_cv(fac, p.X, p.y, n_splits=4)
    b = time_series_cv(fac, p.X, p.y, n_splits=4)
    diff = paired_difference(a, b, "roc_auc")
    assert diff["mean_diff"] == pytest.approx(0.0, abs=1e-9)
    assert diff["crosses_zero"]


def test_determinism_same_seed_same_metrics():
    p = prepare(_raw(n=800, seed=1))
    fac = lambda: make_models(7)["gradient_boosting"]
    fac.name = "gb"
    r1 = time_series_cv(fac, p.X, p.y, n_splits=4).mean_std()["roc_auc"]["mean"]
    r2 = time_series_cv(fac, p.X, p.y, n_splits=4).mean_std()["roc_auc"]["mean"]
    assert r1 == r2


def test_betainc_matches_known_t_pvalue():
    """t=2.776, df=4 is the two-sided 0.05 critical value -> p ~ 0.05."""
    from churn_experiment.evaluate import _t_sf_two_sided

    assert _t_sf_two_sided(2.776, 4) == pytest.approx(0.05, abs=2e-3)
