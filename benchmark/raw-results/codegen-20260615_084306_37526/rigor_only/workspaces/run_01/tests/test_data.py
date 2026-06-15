"""Tests for data loading and preparation."""
import pandas as pd
import pytest

from src.data import FEATURES, TARGET, load_and_prepare


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 2],
        "signup_date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-02-01"],
        "tenure_months": [12, 24, 6, 36, 24],
        "monthly_spend": [50.0, 75.0, 30.0, 100.0, 75.0],
        "support_tickets": [1, 2, 0, 3, 2],
        "days_since_last_login": [5, 60, 3, 45, 60],
        "churned": [0, 1, 0, 1, 1],
    })
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_deduplication_removes_exact_duplicates(sample_csv):
    X, y, info = load_and_prepare(sample_csv)
    assert info["n_dropped_duplicates"] == 1
    assert len(X) == 4


def test_leak_column_dropped(sample_csv):
    X, y, info = load_and_prepare(sample_csv)
    assert "days_since_last_login" not in X.columns


def test_identifier_dropped(sample_csv):
    X, y, info = load_and_prepare(sample_csv)
    assert "customer_id" not in X.columns


def test_all_features_present(sample_csv):
    X, y, info = load_and_prepare(sample_csv)
    for f in FEATURES:
        assert f in X.columns, f"Missing feature: {f}"


def test_target_is_binary(sample_csv):
    X, y, info = load_and_prepare(sample_csv)
    assert set(y.unique()).issubset({0, 1})


def test_output_lengths_match(sample_csv):
    X, y, info = load_and_prepare(sample_csv)
    assert len(X) == len(y)


def test_signup_month_in_range(sample_csv):
    X, y, info = load_and_prepare(sample_csv)
    assert X["signup_month"].between(1, 12).all()
