"""Tests for data loading, deduplication, and splitting."""
import numpy as np
import pandas as pd
import pytest

from src.data import load_and_clean, temporal_split


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 1],
        "signup_date": [
            "2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-01-01"
        ],
        "tenure_months": [12, 6, 24, 3, 12],
        "monthly_spend": [50.0, 30.0, 80.0, 20.0, 50.0],
        "support_tickets": [1, 2, 0, 3, 1],
        "account_status": ["active", "closed", "active", "closed", "active"],
        "churned": [0, 1, 0, 1, 0],
    })
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_account_status_dropped(sample_csv):
    X, y, meta = load_and_clean(sample_csv)
    assert "account_status" not in X.columns


def test_customer_id_dropped(sample_csv):
    X, y, meta = load_and_clean(sample_csv)
    assert "customer_id" not in X.columns


def test_duplicates_removed(sample_csv):
    X, y, meta = load_and_clean(sample_csv)
    assert meta["n_dupes_removed"] == 1
    assert len(X) == 4


def test_temporal_features_extracted(sample_csv):
    X, y, meta = load_and_clean(sample_csv)
    assert "signup_year" in X.columns
    assert "signup_month" in X.columns
    assert "signup_dayofyear" in X.columns
    assert "signup_date" not in X.columns


def test_y_is_binary(sample_csv):
    X, y, meta = load_and_clean(sample_csv)
    assert set(y.unique()).issubset({0, 1})


def test_metadata_contains_expected_keys(sample_csv):
    X, y, meta = load_and_clean(sample_csv)
    for key in ("n_rows", "n_dupes_removed", "churn_rate", "features"):
        assert key in meta


def test_temporal_split_preserves_total_rows(sample_csv):
    X, y, meta = load_and_clean(sample_csv)
    X_train, X_test, y_train, y_test = temporal_split(X, y, test_frac=0.25)
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)


def test_temporal_split_is_chronological(sample_csv):
    X, y, meta = load_and_clean(sample_csv)
    X_train, X_test, y_train, y_test = temporal_split(X, y, test_frac=0.25)
    # After load_and_clean rows are sorted by signup_date, so train indices
    # must all be smaller than test indices
    assert X_train.index.max() < X_test.index.min()


def test_temporal_split_respects_frac(sample_csv):
    X, y, meta = load_and_clean(sample_csv)
    n = len(X)
    for frac in [0.2, 0.5]:
        X_train, X_test, _, _ = temporal_split(X, y, test_frac=frac)
        expected_test = n - int(n * (1 - frac))
        assert len(X_test) == expected_test
