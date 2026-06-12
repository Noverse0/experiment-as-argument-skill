"""Tests for data loading, deduplication, splitting, and feature engineering."""
import pandas as pd
import numpy as np
import pytest

from src.pipeline import (
    load_and_clean,
    temporal_split,
    featurise,
    get_features_target,
    make_lr_pipeline,
    make_gb_pipeline,
    TARGET,
)


@pytest.fixture
def sample_df():
    """Minimal synthetic DataFrame mirroring the churn dataset schema."""
    n = 100
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=n, freq="7D")
    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "signup_date": dates,
        "tenure_months": rng.integers(1, 72, n),
        "monthly_spend": rng.uniform(10, 200, n).round(2),
        "support_tickets": rng.integers(0, 5, n),
        "account_status": ["active"] * n,
        "churned": rng.integers(0, 2, n),
    })
    return df


def test_load_and_clean_drops_duplicates(tmp_path, sample_df):
    # Add 10 duplicate rows.
    df_with_dups = pd.concat([sample_df, sample_df.iloc[:10]], ignore_index=True)
    csv = tmp_path / "test.csv"
    df_with_dups.to_csv(csv, index=False)

    cleaned = load_and_clean(str(csv))
    assert len(cleaned) == len(sample_df), "Duplicates should be removed"


def test_temporal_split_ordering(sample_df):
    train, test = temporal_split(sample_df, test_frac=0.2)
    assert len(train) + len(test) == len(sample_df)
    # All train dates must be <= all test dates.
    assert train["signup_date"].max() <= test["signup_date"].min()


def test_temporal_split_size(sample_df):
    train, test = temporal_split(sample_df, test_frac=0.2)
    assert len(test) == pytest.approx(20, abs=1)


def test_featurise_drops_leaky_columns(sample_df):
    out = featurise(sample_df)
    assert "account_status" not in out.columns
    assert "customer_id" not in out.columns
    assert "signup_date" not in out.columns


def test_featurise_adds_signup_day(sample_df):
    out = featurise(sample_df)
    assert "signup_day" in out.columns
    assert out["signup_day"].dtype in [np.int64, np.int32, "int64"]


def test_get_features_target_no_target_in_X(sample_df):
    X, y = get_features_target(sample_df)
    assert TARGET not in X.columns
    assert y.name == TARGET
    assert len(X) == len(y)


def test_lr_pipeline_fits_and_predicts(sample_df):
    train, test = temporal_split(sample_df, test_frac=0.2)
    X_train, y_train = get_features_target(train)
    X_test, _ = get_features_target(test)

    pipe = make_lr_pipeline(random_state=0)
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_gb_pipeline_fits_and_predicts(sample_df):
    train, test = temporal_split(sample_df, test_frac=0.2)
    X_train, y_train = get_features_target(train)
    X_test, _ = get_features_target(test)

    pipe = make_gb_pipeline(random_state=0)
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_no_data_leakage_across_split(sample_df):
    """Verify that after dedup, no row appears in both train and test."""
    train, test = temporal_split(sample_df, test_frac=0.2)
    train_ids = set(train["customer_id"])
    test_ids = set(test["customer_id"])
    assert train_ids.isdisjoint(test_ids), "customer_ids must not overlap"


def test_featurise_signup_day_monotone(sample_df):
    """signup_day must be monotonically non-decreasing when rows are time-sorted."""
    df_sorted = sample_df.sort_values("signup_date").reset_index(drop=True)
    out = featurise(df_sorted)
    assert (out["signup_day"].diff().dropna() >= 0).all()
