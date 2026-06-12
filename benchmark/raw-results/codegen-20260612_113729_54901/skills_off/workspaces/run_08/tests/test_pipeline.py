"""Pytest tests for the churn experiment pipeline."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import load_clean, get_X_y, LEAKED_COLS, TARGET, FEATURE_COLS
from src.models import make_lr, make_gb
from src.evaluate import cv_evaluate, label_shuffle_auc

DATA_CSV = Path(__file__).parent.parent / "churn.csv"


@pytest.fixture(scope="session")
def csv_path(tmp_path_factory):
    if DATA_CSV.exists():
        return str(DATA_CSV)
    out = tmp_path_factory.mktemp("data") / "churn.csv"
    subprocess.run(
        [sys.executable, str(Path(__file__).parent.parent / "make_dataset.py"), "--out", str(out)],
        check=True,
    )
    return str(out)


@pytest.fixture(scope="session")
def cleaned(csv_path):
    return load_clean(csv_path)


# --- Data cleaning ---

def test_duplicates_removed(cleaned):
    df, stats = cleaned
    assert stats["n_dupes_dropped"] == 200
    assert stats["n_clean"] == 4000


def test_leaked_columns_absent(cleaned):
    df, _ = cleaned
    for col in LEAKED_COLS:
        assert col not in df.columns, f"Leaked column {col!r} must be dropped"


def test_target_present(cleaned):
    df, _ = cleaned
    assert TARGET in df.columns


def test_feature_cols_present(cleaned):
    df, _ = cleaned
    for col in FEATURE_COLS:
        assert col in df.columns


def test_no_nulls(cleaned):
    df, _ = cleaned
    assert df.isnull().sum().sum() == 0


def test_churn_rate_plausible(cleaned):
    _, stats = cleaned
    assert 0.10 <= stats["churn_rate"] <= 0.50


def test_sorted_by_time(cleaned):
    df, _ = cleaned
    days = df["days_since_first"].values
    assert (np.diff(days) >= 0).all()


# --- Model basics ---

def test_lr_predicts_probabilities(cleaned):
    df, _ = cleaned
    X, y = get_X_y(df)
    model = make_lr()
    model.fit(X.iloc[:200], y.iloc[:200])
    proba = model.predict_proba(X.iloc[200:210])
    assert proba.shape == (10, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_gb_predicts_probabilities(cleaned):
    df, _ = cleaned
    X, y = get_X_y(df)
    model = make_gb()
    model.fit(X.iloc[:200], y.iloc[:200])
    proba = model.predict_proba(X.iloc[200:210])
    assert proba.shape == (10, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


# --- Sanity / leakage checks ---

def test_label_shuffle_near_chance(cleaned):
    df, _ = cleaned
    X, y = get_X_y(df)
    result = label_shuffle_auc(make_lr(), X, y, n_splits=3)
    assert result["auc_mean"] < 0.55, (
        f"Label-shuffle AUC {result['auc_mean']:.3f} is too high — possible leakage"
    )


def test_lr_beats_majority_baseline(cleaned):
    df, _ = cleaned
    X, y = get_X_y(df)
    result = cv_evaluate(make_lr(), X, y, n_splits=3)
    assert result["auc_mean"] > 0.55, (
        f"LR AUC {result['auc_mean']:.3f} must exceed majority-class baseline (0.5)"
    )


def test_gb_beats_majority_baseline(cleaned):
    df, _ = cleaned
    X, y = get_X_y(df)
    result = cv_evaluate(make_gb(), X, y, n_splits=3)
    assert result["auc_mean"] > 0.55, (
        f"GBT AUC {result['auc_mean']:.3f} must exceed majority-class baseline (0.5)"
    )


# --- CV result structure ---

def test_cv_result_fields(cleaned):
    df, _ = cleaned
    X, y = get_X_y(df)
    result = cv_evaluate(make_lr(), X, y, n_splits=3)
    for key in ("auc_mean", "auc_std", "f1_mean", "f1_std", "n_folds", "auc_per_fold", "f1_per_fold"):
        assert key in result
    assert result["n_folds"] == 3
    assert len(result["auc_per_fold"]) == 3
