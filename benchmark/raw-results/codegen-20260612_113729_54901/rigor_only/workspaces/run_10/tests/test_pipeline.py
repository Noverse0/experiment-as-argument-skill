"""Tests for data loading, feature engineering, and model pipelines."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import (
    LEAKAGE_COLS,
    load_and_clean,
    make_features,
    make_gb_pipeline,
    make_lr_pipeline,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    churned = rng.integers(0, 2, n)
    return pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="3D").strftime("%Y-%m-%d"),
        "tenure_months": rng.integers(1, 72, n),
        "monthly_spend": rng.gamma(2.0, 30.0, n).round(2),
        "support_tickets": rng.poisson(1.2, n),
        "account_status": np.where(churned == 1, "closed", "active"),
        "churned": churned,
    })


def _save(df: pd.DataFrame, path) -> str:
    p = str(path)
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# load_and_clean
# ---------------------------------------------------------------------------

def test_deduplication_removes_exact_duplicates(tmp_path):
    df = _make_df(50)
    with_dups = pd.concat([df, df.sample(10, random_state=0)], ignore_index=True)
    cleaned = load_and_clean(_save(with_dups, tmp_path / "dup.csv"))
    assert len(cleaned) == 50


def test_deduplication_preserves_all_unique_rows(tmp_path):
    df = _make_df(80)
    cleaned = load_and_clean(_save(df, tmp_path / "clean.csv"))
    assert len(cleaned) == 80


def test_signup_date_parsed_as_datetime(tmp_path):
    df = _make_df(30)
    cleaned = load_and_clean(_save(df, tmp_path / "dates.csv"))
    assert pd.api.types.is_datetime64_any_dtype(cleaned["signup_date"])


# ---------------------------------------------------------------------------
# make_features
# ---------------------------------------------------------------------------

def test_leakage_columns_absent_from_features(tmp_path):
    df = load_and_clean(_save(_make_df(), tmp_path / "df.csv"))
    X, _ = make_features(df)
    for col in LEAKAGE_COLS:
        assert col not in X.columns, f"Leakage column '{col}' must not be in X"


def test_customer_id_absent_from_features(tmp_path):
    df = load_and_clean(_save(_make_df(), tmp_path / "df.csv"))
    X, _ = make_features(df)
    assert "customer_id" not in X.columns


def test_target_absent_from_features(tmp_path):
    df = load_and_clean(_save(_make_df(), tmp_path / "df.csv"))
    X, _ = make_features(df)
    assert "churned" not in X.columns


def test_expected_feature_columns(tmp_path):
    df = load_and_clean(_save(_make_df(), tmp_path / "df.csv"))
    X, _ = make_features(df)
    assert set(X.columns) == {"tenure_months", "monthly_spend", "support_tickets", "signup_age_days"}


def test_signup_age_days_monotone_when_sorted(tmp_path):
    df = load_and_clean(_save(_make_df(200), tmp_path / "df.csv"))
    df = df.sort_values("signup_date").reset_index(drop=True)
    X, _ = make_features(df)
    assert (X["signup_age_days"].diff().dropna() >= 0).all()


def test_feature_row_count_matches_input(tmp_path):
    df = load_and_clean(_save(_make_df(60), tmp_path / "df.csv"))
    X, y = make_features(df)
    assert len(X) == len(y) == 60


# ---------------------------------------------------------------------------
# Pipelines: fit / predict
# ---------------------------------------------------------------------------

@pytest.fixture
def xy(tmp_path):
    df = load_and_clean(_save(_make_df(150), tmp_path / "df.csv"))
    return make_features(df)


@pytest.mark.parametrize("factory", [make_lr_pipeline, make_gb_pipeline])
def test_pipeline_predict_proba_shape(xy, factory):
    X, y = xy
    pipe = factory()
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)


@pytest.mark.parametrize("factory", [make_lr_pipeline, make_gb_pipeline])
def test_pipeline_probabilities_sum_to_one(xy, factory):
    X, y = xy
    pipe = factory()
    pipe.fit(X, y)
    assert np.allclose(pipe.predict_proba(X).sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# Sanity: label-shuffle should hurt performance
# ---------------------------------------------------------------------------

def test_label_shuffle_degrades_auc(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    churned = rng.integers(0, 2, n)
    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="2D").strftime("%Y-%m-%d"),
        "tenure_months": rng.integers(1, 72, n),
        "monthly_spend": rng.gamma(2.0, 30.0, n).round(2),
        "support_tickets": rng.poisson(1.2, n),
        "account_status": np.where(churned == 1, "closed", "active"),
        "churned": churned,
    })
    cleaned = load_and_clean(_save(df, tmp_path / "shuf.csv"))
    X, y = make_features(cleaned)

    split = int(0.6 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    y_shuffled = y_train.sample(frac=1, random_state=42).values
    pipe = make_lr_pipeline()
    pipe.fit(X_train, y_shuffled)
    auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    assert auc < 0.70, f"Shuffled-label AUC {auc:.3f} is suspiciously high — possible leakage"
