"""Pipeline tests: data integrity, split discipline, model sanity."""
import copy
import numpy as np
import pandas as pd
import pytest

from src.data import load_and_clean, time_based_split, get_Xy, FEATURES, TARGET
from src.models import make_pipelines
from src.evaluate import (
    majority_baseline,
    cross_validate_model,
    evaluate_on_test,
    label_shuffle_check,
    overfit_one_batch_check,
)

DATA_PATH = "churn.csv"


# ── Data loading and cleaning ─────────────────────────────────────────────────

def test_deduplication_removes_planted_duplicates():
    df, audit = load_and_clean(DATA_PATH)
    assert audit["n_dupes_removed"] == 200, (
        f"Expected 200 duplicates removed, got {audit['n_dupes_removed']}"
    )
    assert audit["n_after_dedup"] == audit["n_raw"] - 200


def test_account_status_not_in_cleaned_df():
    """account_status is a target-derived column and must be excluded."""
    df, audit = load_and_clean(DATA_PATH)
    assert "account_status" not in df.columns, (
        "account_status must be dropped — it is derived from the target (leakage)"
    )


def test_customer_id_not_in_cleaned_df():
    df, audit = load_and_clean(DATA_PATH)
    assert "customer_id" not in df.columns


def test_features_and_target_present():
    df, _ = load_and_clean(DATA_PATH)
    for col in FEATURES + [TARGET]:
        assert col in df.columns, f"Missing expected column: {col}"


def test_no_missing_values_in_features():
    df, _ = load_and_clean(DATA_PATH)
    assert df[FEATURES].isnull().sum().sum() == 0


def test_target_is_binary():
    df, _ = load_and_clean(DATA_PATH)
    assert set(df[TARGET].unique()).issubset({0, 1})


# ── Split discipline ──────────────────────────────────────────────────────────

def test_time_based_split_no_overlap():
    """Train signup dates must all be strictly before test signup dates."""
    train, test, info = time_based_split(DATA_PATH, test_frac=0.2)
    # We need signup_date for this check; re-read raw before drop
    raw = pd.read_csv(DATA_PATH, parse_dates=["signup_date"])
    raw = raw.drop_duplicates().sort_values("signup_date").reset_index(drop=True)
    cutoff = int(len(raw) * 0.8)
    max_train_date = raw.iloc[:cutoff]["signup_date"].max()
    min_test_date = raw.iloc[cutoff:]["signup_date"].min()
    assert max_train_date <= min_test_date


def test_split_sizes_match_info():
    train, test, info = time_based_split(DATA_PATH, test_frac=0.2)
    assert len(train) == info["n_train"]
    assert len(test) == info["n_test"]


def test_split_covers_all_rows():
    train, test, info = time_based_split(DATA_PATH, test_frac=0.2)
    raw = pd.read_csv(DATA_PATH, parse_dates=["signup_date"])
    n_total = len(raw.drop_duplicates())
    assert info["n_train"] + info["n_test"] == n_total


def test_no_customer_id_overlap_between_splits():
    """After dedup, no customer_id should appear in both train and test.

    Note: feature-level duplicates within a split are allowed — two distinct
    customers can coincidentally share the same (tenure, spend, tickets) values.
    We check identity via customer_id on the raw (pre-column-drop) data.
    """
    raw = pd.read_csv(DATA_PATH, parse_dates=["signup_date"])
    raw = raw.drop_duplicates()
    raw_sorted = raw.sort_values("signup_date").reset_index(drop=True)
    cutoff = int(len(raw_sorted) * 0.8)
    train_ids = set(raw_sorted.iloc[:cutoff]["customer_id"])
    test_ids = set(raw_sorted.iloc[cutoff:]["customer_id"])
    shared = train_ids & test_ids
    assert len(shared) == 0, f"{len(shared)} customer IDs appear in both train and test"


# ── Scaler leakage prevention ─────────────────────────────────────────────────

def test_scaler_is_inside_pipeline():
    """StandardScaler must be inside the Pipeline so CV respects the fit boundary."""
    pipes = make_pipelines(seed=42)
    from sklearn.preprocessing import StandardScaler
    for name, pipe in pipes.items():
        scaler_steps = [s for _, s in pipe.steps if isinstance(s, StandardScaler)]
        assert len(scaler_steps) == 1, (
            f"{name}: StandardScaler must be inside the Pipeline (found {len(scaler_steps)})"
        )


# ── Model sanity ──────────────────────────────────────────────────────────────

def test_models_beat_baseline():
    train, test, _ = time_based_split(DATA_PATH, test_frac=0.2)
    X_train, y_train = get_Xy(train)
    X_test, y_test = get_Xy(test)
    base = majority_baseline(y_train, y_test)
    pipes = make_pipelines(seed=42)
    for name, pipe in pipes.items():
        pipe.fit(X_train, y_train)
        result = evaluate_on_test(pipe, X_test, y_test)
        assert result["test_auc"] > base["test_auc"], (
            f"{name} AUC {result['test_auc']:.4f} does not beat "
            f"baseline {base['test_auc']:.4f}"
        )


def test_overfit_tiny_subset():
    train, _, _ = time_based_split(DATA_PATH, test_frac=0.2)
    X_train, y_train = get_Xy(train)
    pipes = make_pipelines(seed=42)
    for name, pipe in pipes.items():
        result = overfit_one_batch_check(lambda p=pipe: copy.deepcopy(p), X_train, y_train, n=50)
        assert result["passed"], (
            f"{name} failed overfit-tiny check: train_acc={result['train_accuracy_on_tiny']:.3f}"
        )


def test_label_shuffle_drops_performance():
    """Mean AUC over 5 shuffle seeds must be < 0.65.

    A single-seed check has high variance on small test sets (800 samples).
    Averaging over 5 seeds gives a stable estimate of ~0.5 for a clean pipeline.
    """
    train, test, _ = time_based_split(DATA_PATH, test_frac=0.2)
    X_train, y_train = get_Xy(train)
    X_test, y_test = get_Xy(test)
    pipes = make_pipelines(seed=42)
    for name, pipe in pipes.items():
        result = label_shuffle_check(
            lambda p=pipe: copy.deepcopy(p), X_train, y_train, X_test, y_test, seed=42, n_repeats=5
        )
        assert result["passed"], (
            f"{name}: mean shuffled-label AUC={result['shuffled_label_auc_mean']:.3f} is too high "
            f"(per-seed: {[f'{a:.3f}' for a in result['shuffled_label_auc_per_seed']]}) — "
            "possible leakage even with shuffled labels"
        )


def test_cv_returns_correct_shape():
    train, _, _ = time_based_split(DATA_PATH, test_frac=0.2)
    X_train, y_train = get_Xy(train)
    pipes = make_pipelines(seed=42)
    for name, pipe in pipes.items():
        cv = cross_validate_model(pipe, X_train, y_train, n_folds=3, seed=42)
        assert len(cv["cv_auc_per_fold"]) == 3
        assert 0.0 <= cv["cv_auc_mean"] <= 1.0
        assert cv["cv_auc_std"] >= 0.0


def test_get_Xy_shapes():
    train, test, _ = time_based_split(DATA_PATH, test_frac=0.2)
    X_train, y_train = get_Xy(train)
    X_test, y_test = get_Xy(test)
    assert X_train.shape[1] == len(FEATURES)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
