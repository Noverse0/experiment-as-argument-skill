"""Tests for the churn experiment pipeline.

These guard the rigor-critical invariants, not just "does it run":
  - the documented leak column is dropped,
  - duplicates are removed before any split,
  - the time split never lets a test fold precede its train fold,
  - sanity checks land in their expected ranges,
  - the run is deterministic for a fixed seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

from src.data import (
    LEAK_COLUMNS,
    NUMERIC_FEATURES,
    TARGET,
    clean_churn,
    features_and_target,
    load_raw,
)
from src.evaluate import (
    N_SPLITS,
    baseline_floor,
    forward_cv,
    label_shuffle,
    leakage_ceiling,
    overfit_tiny_subset,
)

DATA = "churn.csv"


@pytest.fixture(scope="module")
def raw():
    return load_raw(DATA)


@pytest.fixture(scope="module")
def clean(raw):
    return clean_churn(raw)


def test_leak_column_is_dropped(clean):
    for col in LEAK_COLUMNS:
        assert col not in clean.frame.columns
    assert clean.leak_columns_dropped == LEAK_COLUMNS


def test_duplicates_removed_before_split(raw, clean):
    # raw ships 200 exact duplicate rows on top of unique data.
    assert clean.n_duplicates_dropped == 200
    dedup_subset = [c for c in clean.frame.columns if c != "customer_id"]
    assert clean.frame.duplicated(subset=dedup_subset).sum() == 0


def test_features_exclude_id_and_time(clean):
    X, y = features_and_target(clean.frame)
    assert list(X.columns) == NUMERIC_FEATURES
    assert "customer_id" not in X.columns
    assert "signup_date" not in X.columns
    assert set(np.unique(y)) <= {0, 1}


def test_frame_is_time_sorted(clean):
    dates = clean.frame["signup_date"]
    assert dates.is_monotonic_increasing


def test_time_split_is_forward_only(clean):
    # Every test fold must start strictly after its training fold ends.
    X, _ = features_and_target(clean.frame)
    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    for train_idx, test_idx in splitter.split(X):
        assert train_idx.max() < test_idx.min()


def test_baseline_floor_is_chance(clean):
    res = baseline_floor(clean.frame, seed=0)
    assert 0.45 <= res["roc_auc_mean"] <= 0.55


def test_leakage_ceiling_is_near_perfect(raw):
    # account_status alone must reconstruct the target -> proves it is a leak.
    res = leakage_ceiling(raw, seed=0)
    assert res["roc_auc"] >= 0.99


def test_overfit_tiny_subset(clean):
    res = overfit_tiny_subset(clean.frame, seed=0)
    assert res["train_roc_auc"] >= 0.95


def test_label_shuffle_destroys_signal(clean):
    res = label_shuffle(clean.frame, seed=0)
    assert 0.40 <= res["roc_auc_mean"] <= 0.60


def test_models_beat_baseline(clean):
    res = forward_cv(clean.frame, seed=0)
    for arm in ("logreg", "gboost"):
        assert np.mean(res[arm].roc_auc) > 0.55


def test_determinism_same_seed(clean):
    a = forward_cv(clean.frame, seed=0)
    b = forward_cv(clean.frame, seed=0)
    for arm in ("logreg", "gboost"):
        assert a[arm].roc_auc == b[arm].roc_auc


def test_cleaning_does_not_mutate_target_values(clean):
    assert set(clean.frame[TARGET].unique()) <= {0, 1}
    assert 0.0 < clean.target_rate < 1.0
