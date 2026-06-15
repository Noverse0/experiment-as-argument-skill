import numpy as np
import pandas as pd
import pytest

from src.data import (
    DATE_COL,
    FEATURE_COLS,
    TARGET_COL,
    load_and_clean,
    temporal_split,
)
from src.pipeline import make_gb_pipeline, make_lr_pipeline


def _make_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "customer_id": np.arange(n),
        "signup_date": dates,
        "tenure_months": rng.integers(1, 72, n),
        "monthly_spend": rng.gamma(2, 30, n).round(2),
        "support_tickets": rng.poisson(1.2, n),
        "days_since_last_login": rng.integers(1, 90, n),
        "churned": rng.integers(0, 2, n),
    })


# --- data layer ---

def test_dedup_removes_exact_duplicates(tmp_path):
    df = _make_df(100)
    with_dups = pd.concat([df, df.sample(15, random_state=0)], ignore_index=True)
    csv = tmp_path / "churn.csv"
    with_dups.to_csv(csv, index=False)
    loaded = load_and_clean(str(csv))
    assert len(loaded) == 100


def test_temporal_split_no_future_in_train(tmp_path):
    df = _make_df(100)
    csv = tmp_path / "churn.csv"
    df.to_csv(csv, index=False)
    loaded = load_and_clean(str(csv))
    train, test = temporal_split(loaded, test_frac=0.2)
    assert train[DATE_COL].max() <= test[DATE_COL].min()


def test_temporal_split_sizes(tmp_path):
    df = _make_df(100)
    csv = tmp_path / "churn.csv"
    df.to_csv(csv, index=False)
    loaded = load_and_clean(str(csv))
    train, test = temporal_split(loaded, test_frac=0.2)
    assert len(train) + len(test) == 100
    assert len(test) == pytest.approx(20, abs=1)


def test_feature_cols_exclude_leak_and_id():
    assert "days_since_last_login" not in FEATURE_COLS, "leak must not be a feature"
    assert "customer_id" not in FEATURE_COLS
    assert "signup_date" not in FEATURE_COLS
    assert TARGET_COL not in FEATURE_COLS
    assert len(FEATURE_COLS) == 3


# --- pipeline layer ---

def _synthetic_Xy(n=120, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(n) * 0.3 > 0).astype(int)
    return X, y


def test_lr_pipeline_valid_probabilities():
    X, y = _synthetic_Xy()
    pipe = make_lr_pipeline(random_state=42)
    pipe.fit(X[:90], y[:90])
    probs = pipe.predict_proba(X[90:])
    assert probs.shape == (30, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert probs.min() >= 0.0 and probs.max() <= 1.0


def test_gb_pipeline_valid_probabilities():
    X, y = _synthetic_Xy()
    pipe = make_gb_pipeline(random_state=42)
    pipe.fit(X[:90], y[:90])
    probs = pipe.predict_proba(X[90:])
    assert probs.shape == (30, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_scaler_fitted_on_train_only():
    """Scaler statistics must reflect train data, not contaminated by test."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((80, 3))  # mean ~0, std ~1
    y_train = (X_train[:, 0] > 0).astype(int)
    X_test = rng.standard_normal((20, 3)) * 10 + 100  # very different scale

    pipe = make_lr_pipeline(random_state=42)
    pipe.fit(X_train, y_train)
    scaler = pipe.named_steps["scaler"]

    assert abs(scaler.mean_[0]) < 1.0, "scaler mean should reflect train (≈0)"
    assert abs(scaler.mean_[0] - 100.0) > 5.0, "scaler must not be contaminated by test"


def test_gb_overfits_tiny_subset():
    """Gradient boosting must reach high accuracy on tiny data (pipeline sanity)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3))
    y = (X[:, 0] > 0).astype(int)
    pipe = make_gb_pipeline(random_state=42)
    pipe.fit(X, y)
    acc = (pipe.predict(X) == y).mean()
    assert acc > 0.9, f"expected near-perfect train accuracy, got {acc:.2f}"


def test_lr_beats_majority_class():
    """LR must beat the majority-class baseline on a separable problem."""
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import roc_auc_score

    X, y = _synthetic_Xy(n=300)
    split = 240
    pipe = make_lr_pipeline(random_state=42)
    pipe.fit(X[:split], y[:split])
    lr_auc = roc_auc_score(y[split:], pipe.predict_proba(X[split:])[:, 1])

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X[:split], y[:split])
    base_auc = roc_auc_score(y[split:], dummy.predict_proba(X[split:])[:, 1])

    assert lr_auc > base_auc, f"LR ({lr_auc:.3f}) should beat baseline ({base_auc:.3f})"
