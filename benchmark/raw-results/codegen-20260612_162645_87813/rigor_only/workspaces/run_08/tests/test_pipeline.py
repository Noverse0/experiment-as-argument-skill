"""Tests that the rigor guarantees actually hold in the pipeline."""
import numpy as np
import pandas as pd

from src.data import (
    FEATURE_COLUMNS,
    LEAK_COLUMNS,
    audit,
    clean,
    features_and_target,
)
from src.experiment import run_full_experiment, time_based_split_eval
from src import sanity


# ---- data discipline -------------------------------------------------------

def test_dedup_removes_duplicates_before_split(df_raw):
    a = audit(df_raw)
    assert a.n_duplicates == 200, "generator plants 200 exact duplicates"
    cleaned = clean(df_raw)
    assert cleaned.duplicated().sum() == 0
    assert len(cleaned) == a.n_after_dedup == a.n_raw - 200


def test_leak_and_id_columns_excluded_from_features(df_raw):
    X, y = features_and_target(clean(df_raw), include_leak=False)
    assert list(X.columns) == FEATURE_COLUMNS
    for bad in LEAK_COLUMNS + ["customer_id", "signup_date", "churned"]:
        assert bad not in X.columns
    assert set(y.unique()) <= {0, 1}


def test_clean_is_sorted_by_time(df_raw):
    cleaned = clean(df_raw)
    times = pd.to_datetime(cleaned["signup_date"])
    assert times.is_monotonic_increasing


# ---- sanity checks ---------------------------------------------------------

def test_account_status_is_a_perfect_leak(df_raw):
    res = sanity.leakage_demo(df_raw)
    assert res["passed"], res
    assert res["auc_with_leak"] > 0.99


def test_clean_features_not_near_perfect(df_raw):
    res = sanity.clean_not_near_perfect(df_raw)
    assert res["passed"], res
    assert res["auc_clean"] < 0.95


def test_label_shuffle_collapses_to_chance(df_raw):
    res = sanity.label_shuffle(df_raw)
    assert res["passed"], res
    assert 0.42 <= res["auc_shuffled"] <= 0.58


def test_both_models_beat_baseline(df_raw):
    res = sanity.beats_baseline(df_raw)
    assert res["passed"], res


# ---- experiment behavior ---------------------------------------------------

def test_majority_baseline_is_chance(df_raw):
    out = run_full_experiment(df_raw, seed=1)
    assert abs(out["majority_baseline"]["roc_auc_mean"] - 0.5) < 0.02


def test_experiment_is_deterministic(df_raw):
    a = run_full_experiment(df_raw, seed=42)
    b = run_full_experiment(df_raw, seed=42)
    for arm in a["arms"]:
        assert a["arms"][arm]["roc_auc_mean"] == b["arms"][arm]["roc_auc_mean"]
    assert a["comparison"]["p_value"] == b["comparison"]["p_value"]


def test_comparison_reports_variance_and_ci(df_raw):
    out = run_full_experiment(df_raw, seed=3)
    cmp = out["comparison"]
    assert cmp["n_pairs"] == out["config"]["n_estimates"]
    lo, hi = cmp["ci95_diff"]
    assert lo <= cmp["mean_diff_b_minus_a"] <= hi
    # every arm must carry a per-fold spread (variance, not a single anecdote)
    for arm in out["arms"].values():
        assert arm["n_estimates"] == out["config"]["n_estimates"]
        assert len(arm["roc_auc_per_fold"]) == arm["n_estimates"]


def test_time_based_split_respects_order(df_raw):
    out = time_based_split_eval(df_raw, seed=5)
    assert out["n_train"] > out["n_test"] > 0
    for arm in out["arms"].values():
        assert 0.5 < arm["roc_auc"] < 0.95  # learns signal, not leaking
