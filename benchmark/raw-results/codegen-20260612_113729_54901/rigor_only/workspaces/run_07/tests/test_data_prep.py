"""Tests for data loading, deduplication, and splitting."""

import pandas as pd
import pytest

from src.data_prep import (
    FEATURES,
    TARGET,
    churn_rate,
    get_X_y,
    load_and_clean,
    time_split,
)


@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 1],
        "signup_date": pd.to_datetime(["2023-01-01", "2023-06-01", "2023-03-01",
                                       "2023-09-01", "2023-01-01"]),
        "tenure_months": [12, 6, 18, 3, 12],
        "monthly_spend": [50.0, 80.0, 40.0, 120.0, 50.0],
        "support_tickets": [1, 2, 0, 3, 1],
        "account_status": ["active", "closed", "active", "closed", "active"],
        "churned": [0, 1, 0, 1, 0],
    })


def test_deduplication_removes_exact_dupes(raw_df, tmp_path):
    csv = tmp_path / "test.csv"
    raw_df.to_csv(csv, index=False)
    df = load_and_clean(str(csv))
    assert len(df) == 4  # row 5 is an exact duplicate of row 1


def test_leaky_column_not_in_features():
    assert "account_status" not in FEATURES


def test_id_column_not_in_features():
    assert "customer_id" not in FEATURES


def test_time_split_is_chronological(raw_df, tmp_path):
    csv = tmp_path / "test.csv"
    raw_df.to_csv(csv, index=False)
    df = load_and_clean(str(csv))
    train, test = time_split(df, train_frac=0.75)
    assert train["signup_date"].max() <= test["signup_date"].min()


def test_time_split_sizes(raw_df, tmp_path):
    csv = tmp_path / "test.csv"
    raw_df.to_csv(csv, index=False)
    df = load_and_clean(str(csv))
    train, test = time_split(df, train_frac=0.75)
    assert len(train) + len(test) == len(df)
    assert len(train) > 0
    assert len(test) > 0


def test_get_X_y_shapes(raw_df, tmp_path):
    csv = tmp_path / "test.csv"
    raw_df.to_csv(csv, index=False)
    df = load_and_clean(str(csv))
    X, y = get_X_y(df)
    assert X.shape[1] == len(FEATURES)
    assert len(y) == len(X)
    assert y.name == TARGET


def test_churn_rate_bounds(raw_df, tmp_path):
    csv = tmp_path / "test.csv"
    raw_df.to_csv(csv, index=False)
    df = load_and_clean(str(csv))
    rate = churn_rate(df)
    assert 0.0 <= rate <= 1.0
