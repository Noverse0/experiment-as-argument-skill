"""Tests that guard the rigor of the churn experiment, not just that it runs.

Each test pins down one property the comparison depends on: no leakage column
survives, dedup happens before the split, preprocessing is fit per-fold, the
pipeline is deterministic, and the sanity-check floors/ceilings hold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import FEATURES, LEAK_COLS, load_clean, split_xy
from src.experiment import (
    aggregate,
    baseline_auc,
    evaluate_model,
    label_shuffle_auc,
    make_model,
    overfit_tiny_auc,
)


@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory):
    """A small synthetic CSV mirroring the real dataset's traps."""
    rng = np.random.default_rng(0)
    n = 400
    churn = (rng.random(n) < 0.3).astype(int)
    df = pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "signup_date": (
                pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 600, n), unit="D")
            ).strftime("%Y-%m-%d"),
            "tenure_months": rng.integers(1, 72, n),
            "monthly_spend": rng.gamma(2.0, 30.0, n).round(2),
            "support_tickets": rng.poisson(1.2, n),
            "account_status": np.where(churn == 1, "closed", "active"),
            "churned": churn,
        }
    )
    dup = df.sample(n=40, random_state=0)  # planted exact duplicates
    full = pd.concat([df, dup], ignore_index=True)
    path = tmp_path_factory.mktemp("data") / "mini.csv"
    full.to_csv(path, index=False)
    return str(path), 40


def test_dedup_happens_before_split(synthetic_csv):
    path, n_dupes = synthetic_csv
    df, stats = load_clean(path)
    assert stats.n_exact_duplicates == n_dupes
    assert stats.n_after_dedup == stats.n_raw - n_dupes
    assert df.duplicated().sum() == 0


def test_data_is_time_sorted(synthetic_csv):
    path, _ = synthetic_csv
    df, _ = load_clean(path)
    assert df["signup_date"].is_monotonic_increasing


def test_leak_column_not_in_features(synthetic_csv):
    path, _ = synthetic_csv
    df, _ = load_clean(path)
    X, y = split_xy(df, include_leak=False)
    for leak in LEAK_COLS:
        assert leak not in X.columns
    assert list(X.columns) == FEATURES
    assert "churned" not in X.columns


def test_leak_probe_is_near_perfect(synthetic_csv):
    """account_status must be a perfect predictor -> proves it is leakage."""
    path, _ = synthetic_csv
    df, _ = load_clean(path)
    X_leak, y = split_xy(df, include_leak=True)
    aucs = [f.roc_auc for f in evaluate_model("logreg", X_leak, y)]
    assert np.mean(aucs) > 0.99


def test_preprocessing_fit_per_fold_no_global_leak(synthetic_csv):
    """The scaler must be fit inside fit(), so refitting on a subset changes it."""
    path, _ = synthetic_csv
    df, _ = load_clean(path)
    X, y = split_xy(df)
    m = make_model("logreg")
    m.fit(X.iloc[:100], y.iloc[:100])
    mean_small = m.named_steps["pre"].mean_.copy()
    m.fit(X, y)
    mean_full = m.named_steps["pre"].mean_
    assert not np.allclose(mean_small, mean_full)


def test_determinism_same_seed(synthetic_csv):
    path, _ = synthetic_csv
    df, _ = load_clean(path)
    X, y = split_xy(df)
    a = [f.roc_auc for f in evaluate_model("gbm", X, y, seed=7)]
    b = [f.roc_auc for f in evaluate_model("gbm", X, y, seed=7)]
    assert a == b


def test_baseline_floor(synthetic_csv):
    path, _ = synthetic_csv
    df, _ = load_clean(path)
    X, y = split_xy(df)
    assert abs(baseline_auc(X, y) - 0.5) < 0.15


def test_label_shuffle_collapses(synthetic_csv):
    path, _ = synthetic_csv
    df, _ = load_clean(path)
    X, y = split_xy(df)
    for name in ("logreg", "gbm"):
        assert abs(label_shuffle_auc(name, X, y) - 0.5) < 0.2


def test_overfit_tiny_slice(synthetic_csv):
    path, _ = synthetic_csv
    df, _ = load_clean(path)
    X, y = split_xy(df)
    assert overfit_tiny_auc("gbm", X, y) > 0.95


def test_aggregate_shapes(synthetic_csv):
    path, _ = synthetic_csv
    df, _ = load_clean(path)
    X, y = split_xy(df)
    agg = aggregate(evaluate_model("logreg", X, y))
    assert agg["n_folds"] == 5
    assert 0.0 <= agg["roc_auc_mean"] <= 1.0
    assert agg["roc_auc_sd"] >= 0.0
