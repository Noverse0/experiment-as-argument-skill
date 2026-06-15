"""Tests for the churn experiment pipeline."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.data_pipeline import FEATURES, TARGET, deduplicate, get_X_y, time_based_split
from src.evaluate import baseline_score, label_shuffle_roc_auc, run_cv, score, summarise_cv
from src.models import get_models


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture()
def small_df():
    """Minimal deterministic dataframe with known duplicate rows."""
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="3D").strftime("%Y-%m-%d"),
        "tenure_months": rng.integers(1, 72, n),
        "monthly_spend": rng.gamma(2, 30, n).round(2),
        "support_tickets": rng.poisson(1.2, n),
        "days_since_last_login": rng.integers(1, 90, n),
        "churned": rng.integers(0, 2, n),
    })
    # Append 10 exact duplicates (mirrors the real dataset's 200 dups)
    dup = df.sample(n=10, random_state=0)
    return pd.concat([df, dup], ignore_index=True)


# ── data_pipeline tests ──────────────────────────────────────────────────────

def test_deduplicate_removes_exact_dups(small_df):
    clean = deduplicate(small_df)
    assert len(clean) == 100  # 10 duplicates removed


def test_deduplicate_is_idempotent(small_df):
    once = deduplicate(small_df)
    twice = deduplicate(once)
    assert len(once) == len(twice)


def test_leak_feature_absent_from_features():
    """days_since_last_login must NOT be in the feature set (target leak)."""
    assert "days_since_last_login" not in FEATURES


def test_id_column_absent_from_features():
    assert "customer_id" not in FEATURES


def test_signup_date_absent_from_features():
    """signup_date is temporal — used only for split ordering."""
    assert "signup_date" not in FEATURES


def test_get_X_y_columns(small_df):
    X, y = get_X_y(small_df)
    assert list(X.columns) == FEATURES
    assert y.name == TARGET
    assert len(X) == len(y)


def test_time_based_split_no_overlap(small_df):
    clean = deduplicate(small_df)
    train, test = time_based_split(clean, test_frac=0.2)
    assert len(train) + len(test) == len(clean)
    # After sorting by signup_date, every train date <= every test date
    assert train["signup_date"].max() <= test["signup_date"].min()


def test_time_based_split_sizes(small_df):
    clean = deduplicate(small_df)
    train, test = time_based_split(clean, test_frac=0.2)
    # Allow ±1 row for rounding
    assert abs(len(test) / len(clean) - 0.2) < 0.02


# ── models tests ──────────────────────────────────────────────────────────────

def test_get_models_returns_both():
    models = get_models()
    assert "LogisticRegression" in models
    assert "GradientBoosting" in models


def test_models_have_predict_proba():
    for name, model in get_models().items():
        assert hasattr(model, "predict_proba"), f"{name} must expose predict_proba"


# ── evaluate tests ────────────────────────────────────────────────────────────

@pytest.fixture()
def train_test_data(small_df):
    clean = deduplicate(small_df)
    train, test = time_based_split(clean, test_frac=0.2)
    return get_X_y(train), get_X_y(test)


@pytest.fixture()
def large_train_test_data():
    """500-row fixture: large enough for AUC estimates to be reliable."""
    rng = np.random.default_rng(7)
    n = 500
    tenure = rng.integers(1, 72, n)
    spend = rng.gamma(2.0, 30.0, n).round(2)
    tickets = rng.poisson(1.2, n)
    signup = pd.date_range("2023-01-01", periods=n, freq="2D").strftime("%Y-%m-%d")
    logit = -1.2 - 0.03 * tenure + 0.01 * spend + 0.45 * tickets
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "signup_date": signup,
        "tenure_months": tenure,
        "monthly_spend": spend,
        "support_tickets": tickets,
        "days_since_last_login": rng.integers(1, 90, n),
        "churned": churn,
    })
    train, test = time_based_split(df, test_frac=0.2)
    return get_X_y(train), get_X_y(test)


def test_score_returns_valid_metrics(train_test_data):
    (X_tr, y_tr), (X_te, y_te) = train_test_data
    model = list(get_models().values())[0]
    m = clone(model)
    metrics = score(m, X_tr, y_tr, X_te, y_te)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_run_cv_length(train_test_data):
    (X_tr, y_tr), _ = train_test_data
    model = list(get_models().values())[0]
    results = run_cv(model, X_tr, y_tr, n_splits=3, seeds=(0,))
    assert len(results) == 3  # 3 folds × 1 seed


def test_summarise_cv_keys(train_test_data):
    (X_tr, y_tr), _ = train_test_data
    model = list(get_models().values())[0]
    raw = run_cv(model, X_tr, y_tr, n_splits=3, seeds=(0,))
    s = summarise_cv(raw)
    for key in ("roc_auc_mean", "roc_auc_std", "f1_mean", "f1_std", "n_evals"):
        assert key in s


def test_baseline_is_near_random(train_test_data):
    (X_tr, y_tr), (X_te, y_te) = train_test_data
    b = baseline_score(y_tr, y_te)
    # Majority-class baseline AUC is 0.5 (constant predictor)
    assert abs(b["roc_auc"] - 0.5) < 0.05


def test_label_shuffle_drops_auc(large_train_test_data):
    """Real AUC must be substantially higher than shuffled-label null AUC (gap > 0.05)."""
    (X_tr, y_tr), (X_te, y_te) = large_train_test_data
    model = list(get_models().values())[0]
    sh_mean, _ = label_shuffle_roc_auc(model, X_tr, y_tr, X_te, y_te)
    real_auc = score(clone(model), X_tr, y_tr, X_te, y_te)["roc_auc"]
    gap = real_auc - sh_mean
    assert gap > 0.05, (
        f"Real AUC={real_auc:.3f}  shuffled-null={sh_mean:.3f}  gap={gap:.3f} — "
        "model may not be learning real signal."
    )


def test_models_beat_baseline(large_train_test_data):
    """Both models must beat the majority-class baseline (needs enough rows for reliable AUC)."""
    (X_tr, y_tr), (X_te, y_te) = large_train_test_data
    baseline = baseline_score(y_tr, y_te)
    for name, model in get_models().items():
        m = clone(model)
        metrics = score(m, X_tr, y_tr, X_te, y_te)
        assert metrics["roc_auc"] > baseline["roc_auc"], (
            f"{name} AUC {metrics['roc_auc']:.3f} did not beat baseline {baseline['roc_auc']:.3f}"
        )
