import pandas as pd
import numpy as np
import pytest

from src.data import dedup, engineer_features, time_split


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 1, 2],
        "signup_date": ["2023-01-01", "2023-06-01", "2023-12-01", "2023-01-01", "2023-06-01"],
        "tenure_months": [12, 6, 3, 12, 6],
        "monthly_spend": [50.0, 80.0, 30.0, 50.0, 80.0],
        "support_tickets": [1, 2, 0, 1, 2],
        "account_status": ["active", "closed", "active", "active", "closed"],
        "churned": [0, 1, 0, 0, 1],
    })


def test_dedup_removes_duplicates(sample_df):
    cleaned, n_removed = dedup(sample_df)
    assert n_removed == 2
    assert len(cleaned) == 3


def test_dedup_returns_exact_count(sample_df):
    _, n_removed = dedup(sample_df)
    assert n_removed == len(sample_df) - len(sample_df.drop_duplicates())


def test_dedup_does_not_remove_non_duplicates():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
    })
    cleaned, n_removed = dedup(df)
    assert n_removed == 0
    assert len(cleaned) == 3


def test_engineer_features_drops_leaker(sample_df):
    df, _ = dedup(sample_df)
    result = engineer_features(df)
    assert "account_status" not in result.columns, "account_status is a leaker and must be dropped"


def test_engineer_features_drops_id(sample_df):
    df, _ = dedup(sample_df)
    result = engineer_features(df)
    assert "customer_id" not in result.columns


def test_engineer_features_drops_signup_date(sample_df):
    df, _ = dedup(sample_df)
    result = engineer_features(df)
    assert "signup_date" not in result.columns


def test_engineer_features_adds_days_since_start(sample_df):
    df, _ = dedup(sample_df)
    result = engineer_features(df)
    assert "days_since_start" in result.columns
    assert result["days_since_start"].min() == 0


def test_engineer_features_days_monotone(sample_df):
    df, _ = dedup(sample_df)
    result = engineer_features(df)
    assert (result["days_since_start"] >= 0).all()


def test_time_split_ordering(sample_df):
    df, _ = dedup(sample_df)
    df = engineer_features(df)
    train, test = time_split(df, train_frac=0.67)
    assert train["days_since_start"].max() <= test["days_since_start"].min()


def test_time_split_sizes(sample_df):
    df, _ = dedup(sample_df)
    df = engineer_features(df)
    train, test = time_split(df, train_frac=0.67)
    assert len(train) + len(test) == len(df)
    assert len(train) == int(len(df) * 0.67)


def test_time_split_no_overlap(sample_df):
    df, _ = dedup(sample_df)
    df = engineer_features(df)
    train, test = time_split(df)
    train_idx = set(train.index)
    test_idx = set(test.index)
    assert train_idx.isdisjoint(test_idx)
