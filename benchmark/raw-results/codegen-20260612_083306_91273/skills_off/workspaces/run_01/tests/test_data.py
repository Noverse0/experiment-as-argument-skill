"""Tests for data loading, deduplication, and splitting."""
import pandas as pd
import numpy as np
import pytest

from src.data import (
    clean_data,
    dedup_data,
    feature_engineer,
    time_split,
    LEAK_COLS,
    ID_COLS,
    TARGET,
)


def _make_df(n=100, seed=0, with_dups=True):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "signup_date": pd.date_range("2023-01-01", periods=n).strftime("%Y-%m-%d"),
            "tenure_months": rng.integers(1, 72, n),
            "monthly_spend": rng.gamma(2, 30, n).round(2),
            "support_tickets": rng.poisson(1.2, n),
            "account_status": rng.choice(["active", "closed"], n),
            "churned": rng.integers(0, 2, n),
        }
    )
    if with_dups:
        dup = df.sample(n=10, random_state=seed)
        df = pd.concat([df, dup], ignore_index=True)
    return df


def test_clean_drops_leak_cols():
    df = _make_df()
    cleaned = clean_data(df)
    for col in LEAK_COLS + ID_COLS:
        assert col not in cleaned.columns


def test_clean_keeps_target():
    df = _make_df()
    cleaned = clean_data(df)
    assert TARGET in cleaned.columns


def test_dedup_removes_duplicates():
    df = _make_df(n=100, with_dups=True)
    n_before = len(df)
    deduped = dedup_data(df)
    assert len(deduped) < n_before
    assert len(deduped) == len(deduped.drop_duplicates())


def test_dedup_no_change_if_clean():
    df = _make_df(n=50, with_dups=False)
    deduped = dedup_data(df)
    assert len(deduped) == 50


def test_feature_engineer_creates_signup_day():
    df = clean_data(_make_df())
    df = dedup_data(df)
    fe = feature_engineer(df)
    assert "signup_day" in fe.columns
    assert "signup_date" not in fe.columns
    assert fe["signup_day"].dtype in (int, np.int64, np.int32, "int64")


def test_time_split_no_index_overlap():
    df = clean_data(_make_df(n=200, with_dups=False))
    df = dedup_data(df)
    df = feature_engineer(df)
    X_train, X_test, y_train, y_test = time_split(df)
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_time_split_respects_fraction():
    df = clean_data(_make_df(n=200, with_dups=False))
    df = dedup_data(df)
    df = feature_engineer(df)
    X_train, X_test, _, _ = time_split(df, test_frac=0.2)
    total = len(X_train) + len(X_test)
    assert abs(len(X_test) / total - 0.2) < 0.02


def test_account_status_not_in_features():
    """The perfect leak column must not survive data preparation."""
    df = _make_df()
    cleaned = clean_data(df)
    assert "account_status" not in cleaned.columns
