"""Pytest tests for the churn prediction pipeline."""
import numpy as np
import pandas as pd
import pytest

from src.data import DROPPED, FEATURES, TARGET, load
from src.evaluate import cv_scores, sanity_checks
from src.pipeline import make_gbm, make_lr

DATA_PATH = "churn.csv"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def test_load_returns_correct_features():
    X, y = load(DATA_PATH)
    assert list(X.columns) == FEATURES


def test_load_drops_leaked_columns():
    X, _ = load(DATA_PATH)
    for col in DROPPED:
        assert col not in X.columns, f"Leaked column still present: {col}"


def test_target_is_binary():
    _, y = load(DATA_PATH)
    assert set(y.unique()).issubset({0, 1})


def test_deduplication_removes_duplicates():
    """The generator appends 200 exact dupe rows; load() must remove them."""
    X, y = load(DATA_PATH)
    # After dedup the dataset should have fewer rows than the raw CSV
    raw = pd.read_csv(DATA_PATH)
    assert len(X) < len(raw), "Deduplication did not reduce row count"
    assert len(X) == len(raw.drop_duplicates()), "Row count after dedup is wrong"


def test_days_since_last_login_absent():
    """Explicitly guard against the disguised target leak."""
    X, _ = load(DATA_PATH)
    assert "days_since_last_login" not in X.columns


def test_no_missing_values():
    X, y = load(DATA_PATH)
    assert X.isnull().sum().sum() == 0
    assert y.isnull().sum() == 0


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

def test_lr_pipeline_fits_and_predicts():
    X, y = load(DATA_PATH)
    pipe = make_lr(seed=0)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_gbm_pipeline_fits_and_predicts():
    X, y = load(DATA_PATH)
    pipe = make_gbm(seed=0)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_pipeline_probabilities_in_unit_interval():
    X, y = load(DATA_PATH)
    for factory in [make_lr, make_gbm]:
        pipe = factory(seed=0)
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_cv_scores_structure():
    X, y = load(DATA_PATH)
    pipe = make_lr(seed=42)
    scores = cv_scores(pipe, X, y, n_splits=3, seeds=[0])
    assert "roc_auc" in scores
    for metric in ["roc_auc", "f1", "accuracy"]:
        assert "mean" in scores[metric]
        assert "std" in scores[metric]
        assert "n" in scores[metric]
        assert scores[metric]["n"] == 3  # 1 seed × 3 folds


def test_cv_roc_auc_above_baseline():
    """Both models must beat a majority-class baseline (AUC > 0.5)."""
    X, y = load(DATA_PATH)
    for factory in [make_lr, make_gbm]:
        pipe = factory(seed=0)
        scores = cv_scores(pipe, X, y, n_splits=3, seeds=[0])
        auc = scores["roc_auc"]["mean"]
        assert auc > 0.5, f"{factory.__name__} AUC {auc:.4f} not above baseline"


def test_cv_reproducible_with_same_seed():
    """Same pipeline, same seed → identical mean AUC."""
    X, y = load(DATA_PATH)
    s1 = cv_scores(make_lr(seed=7), X, y, n_splits=3, seeds=[7])
    s2 = cv_scores(make_lr(seed=7), X, y, n_splits=3, seeds=[7])
    assert s1["roc_auc"]["mean"] == s2["roc_auc"]["mean"]


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def test_sanity_checks_pass_for_gbm():
    X, y = load(DATA_PATH)
    result = sanity_checks(make_gbm, X, y)
    assert result["overfit_ok"], f"GBM failed overfit check: {result}"
    assert result["shuffle_ok"], f"GBM failed shuffle check: {result}"


def test_sanity_checks_pass_for_lr():
    X, y = load(DATA_PATH)
    # LR is a low-capacity linear model; with 3 noisy features and 50 training
    # samples it won't memorise like GBM. 0.65 confirms the pipeline works.
    result = sanity_checks(make_lr, X, y, overfit_threshold=0.65)
    assert result["overfit_ok"], f"LR failed overfit check: {result}"
    assert result["shuffle_ok"], f"LR failed shuffle check: {result}"


def test_label_shuffle_degrades_performance():
    """Shuffled labels should yield AUC near 0.5 (< 0.6), confirming no leakage."""
    X, y = load(DATA_PATH)
    result = sanity_checks(make_gbm, X, y)
    assert result["label_shuffle_auc"] < 0.6, (
        f"Shuffle AUC {result['label_shuffle_auc']:.4f} is too high — "
        "possible feature leakage"
    )
