"""Tests for data loading, deduplication, and splitting."""
import pandas as pd
import pytest

from src.data import FEATURES, LEAK_COLS, get_xy, load_data, temporal_split


def _make_df(n: int = 5):
    return pd.DataFrame({
        "customer_id": range(1, n + 1),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="ME").strftime("%Y-%m-%d"),
        "tenure_months": [12] * n,
        "monthly_spend": [50.0] * n,
        "support_tickets": [1] * n,
        "days_since_last_login": [5] * n,
        "churned": [0, 1] * (n // 2) + [0] * (n % 2),
    })


def test_dedup_removes_duplicate_customer_ids(tmp_path):
    df = _make_df(5)
    dup = df.iloc[[0, 1]]  # same customer_ids as rows 0 and 1
    combined = pd.concat([df, dup], ignore_index=True)
    csv = tmp_path / "churn.csv"
    combined.to_csv(csv, index=False)

    loaded = load_data(str(csv))
    assert len(loaded) == 5  # duplicates removed


def test_dedup_no_duplicate_customer_ids_after_load(tmp_path):
    df = _make_df(4)
    csv = tmp_path / "churn.csv"
    df.to_csv(csv, index=False)
    loaded = load_data(str(csv))
    assert loaded["customer_id"].duplicated().sum() == 0


def test_leak_cols_not_in_features():
    for col in LEAK_COLS:
        assert col not in FEATURES, f"Leak column '{col}' must not be in FEATURES"


def test_temporal_split_no_time_leakage():
    df = _make_df(10)
    train, test = temporal_split(df, train_frac=0.8)
    assert train["signup_date"].max() <= test["signup_date"].min()


def test_temporal_split_sizes():
    df = _make_df(10)
    train, test = temporal_split(df, train_frac=0.8)
    assert len(train) == 8
    assert len(test) == 2


def test_get_xy_returns_only_clean_features():
    df = _make_df(5)
    X, y = get_xy(df)
    assert list(X.columns) == FEATURES
    assert y.name == "churned"
    assert len(X) == len(y)


def test_get_xy_no_leak_columns():
    df = _make_df(5)
    X, _ = get_xy(df)
    for col in LEAK_COLS:
        assert col not in X.columns
