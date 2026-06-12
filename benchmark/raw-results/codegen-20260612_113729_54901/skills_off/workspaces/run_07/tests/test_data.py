"""Tests for data loading, deduplication, and splitting."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data import get_features_target, load_data, temporal_split


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "customer_id": range(1, 11),
        "signup_date": [
            "2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01",
            "2023-06-01", "2023-07-01", "2023-08-01", "2023-09-01", "2023-10-01",
        ],
        "tenure_months": [12, 24, 6, 36, 18, 3, 48, 12, 6, 24],
        "monthly_spend": [50.0, 75.0, 30.0, 100.0, 60.0, 25.0, 90.0, 45.0, 35.0, 80.0],
        "support_tickets": [1, 2, 0, 3, 1, 0, 2, 1, 0, 2],
        "account_status": ["active", "closed", "active", "closed", "active",
                           "active", "closed", "active", "active", "closed"],
        "churned": [0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
    })
    path = tmp_path / "test_churn.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def csv_with_dupes(tmp_path):
    base = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "signup_date": ["2023-01-01", "2023-02-01", "2023-03-01"],
        "tenure_months": [12, 24, 6],
        "monthly_spend": [50.0, 75.0, 30.0],
        "support_tickets": [1, 2, 0],
        "account_status": ["active", "closed", "active"],
        "churned": [0, 1, 0],
    })
    duped = pd.concat([base, base.iloc[[0]]], ignore_index=True)
    path = tmp_path / "dupes.csv"
    duped.to_csv(path, index=False)
    return str(path)


def test_drops_account_status(sample_csv):
    df, _ = load_data(sample_csv)
    assert "account_status" not in df.columns


def test_drops_customer_id(sample_csv):
    df, _ = load_data(sample_csv)
    assert "customer_id" not in df.columns


def test_converts_signup_date_to_days(sample_csv):
    df, _ = load_data(sample_csv)
    assert "signup_date" not in df.columns
    assert "days_since_start" in df.columns
    assert df["days_since_start"].min() == 0


def test_deduplication_count(csv_with_dupes):
    df, n_dupes = load_data(csv_with_dupes)
    assert n_dupes == 1
    assert len(df) == 3


def test_no_dupes_in_clean_csv(sample_csv):
    df, n_dupes = load_data(sample_csv)
    assert n_dupes == 0


def test_sorted_by_days_since_start(sample_csv):
    df, _ = load_data(sample_csv)
    assert list(df["days_since_start"]) == sorted(df["days_since_start"])


def test_temporal_split_no_overlap(sample_csv):
    df, _ = load_data(sample_csv)
    train, test = temporal_split(df, train_frac=0.8)
    assert train["days_since_start"].max() <= test["days_since_start"].min()


def test_temporal_split_covers_all_rows(sample_csv):
    df, _ = load_data(sample_csv)
    train, test = temporal_split(df, train_frac=0.8)
    assert len(train) + len(test) == len(df)


def test_get_features_target_separates_correctly(sample_csv):
    df, _ = load_data(sample_csv)
    X, y = get_features_target(df)
    assert "churned" not in X.columns
    assert y.name == "churned"
    assert len(X) == len(y)
