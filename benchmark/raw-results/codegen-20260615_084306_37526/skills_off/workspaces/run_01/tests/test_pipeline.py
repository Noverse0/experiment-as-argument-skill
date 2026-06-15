"""Pytest tests for the churn experiment pipeline."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import LEAKY_FEATURES, load_and_preprocess, temporal_split
from src.evaluate import cv_evaluate, evaluate_held_out, majority_class_auc
from src.models import make_gradient_boosting, make_logistic_regression


@pytest.fixture
def sample_csv(tmp_path):
    """Synthetic churn CSV matching the real schema, with causal signal and duplicates."""
    n = 300
    rng = np.random.default_rng(1)

    tenure = rng.integers(1, 72, n)
    spend = rng.gamma(2.0, 30.0, n).round(2)
    tickets = rng.poisson(1.2, n)
    logit = -1.2 - 0.03 * tenure + 0.01 * spend + 0.45 * tickets
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    # Disguised leak (mirrors make_dataset.py)
    days_login = np.where(
        churn == 1,
        rng.integers(30, 90, n),
        rng.integers(1, 15, n),
    )
    dates = pd.date_range("2023-01-01", periods=n, freq="D").strftime("%Y-%m-%d")

    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "signup_date": dates,
        "tenure_months": tenure,
        "monthly_spend": spend,
        "support_tickets": tickets,
        "days_since_last_login": days_login,
        "churned": churn,
    })
    # Add 20 exact duplicates
    dup = df.sample(20, random_state=1)
    df = pd.concat([df, dup], ignore_index=True)

    path = tmp_path / "churn.csv"
    df.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def test_deduplication(sample_csv):
    X, y, meta = load_and_preprocess(sample_csv)
    assert meta["n_duplicates_removed"] == 20
    assert meta["n_total"] == 300


def test_no_leaky_features(sample_csv):
    X, y, meta = load_and_preprocess(sample_csv)
    for col in X.columns:
        assert col not in LEAKY_FEATURES, f"Leaky feature '{col}' present in X"
    assert "days_since_last_login" not in X.columns


def test_customer_id_excluded(sample_csv):
    X, y, meta = load_and_preprocess(sample_csv)
    assert "customer_id" not in X.columns


def test_target_not_in_features(sample_csv):
    X, y, meta = load_and_preprocess(sample_csv)
    assert "churned" not in X.columns


def test_days_since_signup_is_numeric(sample_csv):
    X, y, meta = load_and_preprocess(sample_csv)
    assert "days_since_signup" in X.columns
    assert pd.api.types.is_numeric_dtype(X["days_since_signup"])


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------

def test_temporal_split_sizes(sample_csv):
    X, y, meta = load_and_preprocess(sample_csv)
    X_train, X_test, y_train, y_test = temporal_split(X, y, test_size=0.2)
    n = len(X)
    assert len(X_train) == int(n * 0.8)
    assert len(X_test) == n - int(n * 0.8)
    assert len(X_train) + len(X_test) == n


def test_temporal_split_order(sample_csv):
    """Train rows must be temporally earlier than test rows."""
    X, y, meta = load_and_preprocess(sample_csv)
    X_train, X_test, y_train, y_test = temporal_split(X, y, test_size=0.2)
    assert X_train["days_since_signup"].max() <= X_test["days_since_signup"].max()
    assert X_train["days_since_signup"].min() <= X_test["days_since_signup"].min()


# ---------------------------------------------------------------------------
# Preprocessing: no leakage across split
# ---------------------------------------------------------------------------

def test_scaler_fits_on_train_only(sample_csv):
    """StandardScaler mean_ must match the training set mean, not the full dataset."""
    X, y, meta = load_and_preprocess(sample_csv)
    X_train, X_test, y_train, y_test = temporal_split(X, y, test_size=0.2)

    model = make_logistic_regression()
    model.fit(X_train, y_train)

    scaler = model.named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, X_train.mean().values, rtol=1e-5)


# ---------------------------------------------------------------------------
# Model outputs
# ---------------------------------------------------------------------------

def test_lr_probabilities_in_unit_interval(sample_csv):
    X, y, meta = load_and_preprocess(sample_csv)
    X_train, X_test, y_train, y_test = temporal_split(X, y, test_size=0.2)
    model = make_logistic_regression()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_gb_probabilities_in_unit_interval(sample_csv):
    X, y, meta = load_and_preprocess(sample_csv)
    X_train, X_test, y_train, y_test = temporal_split(X, y, test_size=0.2)
    model = make_gradient_boosting()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


# ---------------------------------------------------------------------------
# Baseline floor
# ---------------------------------------------------------------------------

def test_models_beat_majority_baseline(sample_csv):
    """Both models must beat a constant-score baseline (AUC > 0.5)."""
    X, y, meta = load_and_preprocess(sample_csv)
    X_train, X_test, y_train, y_test = temporal_split(X, y, test_size=0.2)

    baseline = majority_class_auc(y_test)
    lr_m = evaluate_held_out(make_logistic_regression, X_train, y_train, X_test, y_test)
    gb_m = evaluate_held_out(make_gradient_boosting, X_train, y_train, X_test, y_test)

    assert lr_m["roc_auc"] > baseline, (
        f"LR AUC {lr_m['roc_auc']:.3f} did not beat baseline {baseline:.3f}"
    )
    assert gb_m["roc_auc"] > baseline, (
        f"GB AUC {gb_m['roc_auc']:.3f} did not beat baseline {baseline:.3f}"
    )


# ---------------------------------------------------------------------------
# CV utility
# ---------------------------------------------------------------------------

def test_cv_evaluate_keys_and_counts(sample_csv):
    X, y, meta = load_and_preprocess(sample_csv)
    X_train, _, y_train, _ = temporal_split(X, y, test_size=0.2)

    result = cv_evaluate(make_logistic_regression, X_train, y_train, n_splits=3, seeds=(42,))

    assert "roc_auc_mean" in result
    assert "roc_auc_std" in result
    assert "n_evals" in result
    assert result["n_evals"] == 3  # 3 folds × 1 seed
    assert 0.0 <= result["roc_auc_mean"] <= 1.0


def test_cv_evaluate_auc_above_chance(sample_csv):
    """CV AUC must be above 0.5 with a causally-generated dataset."""
    X, y, meta = load_and_preprocess(sample_csv)
    X_train, _, y_train, _ = temporal_split(X, y, test_size=0.2)

    result = cv_evaluate(make_logistic_regression, X_train, y_train, n_splits=3, seeds=(42,))
    assert result["roc_auc_mean"] > 0.5
