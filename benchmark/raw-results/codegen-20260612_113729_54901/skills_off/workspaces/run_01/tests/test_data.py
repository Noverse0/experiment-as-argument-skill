"""Tests for data loading and cleaning."""
import pytest
import pandas as pd
import numpy as np

from src.data import load_and_clean, get_features_and_target, class_balance_report


@pytest.fixture
def raw_csv(tmp_path):
    """Small CSV that reproduces all three planted traps at a miniature scale."""
    rows = [
        # customer_id, signup_date,    tenure, spend,  tickets, account_status, churned
        (1, "2023-01-15", 10, 50.0, 0, "active", 0),
        (2, "2023-03-01", 5,  80.0, 2, "closed", 1),
        (3, "2023-06-10", 20, 60.0, 1, "active", 0),
        (4, "2023-09-01", 3,  110.0, 3, "closed", 1),
        # exact duplicate of row 1 — simulates the planted duplicates
        (1, "2023-01-15", 10, 50.0, 0, "active", 0),
    ]
    df = pd.DataFrame(rows, columns=[
        "customer_id", "signup_date", "tenure_months", "monthly_spend",
        "support_tickets", "account_status", "churned",
    ])
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_deduplication_removes_planted_duplicate(raw_csv):
    df = load_and_clean(raw_csv)
    assert len(df) == 4  # 5 rows - 1 duplicate


def test_leaky_account_status_dropped(raw_csv):
    df = load_and_clean(raw_csv)
    assert "account_status" not in df.columns


def test_customer_id_dropped(raw_csv):
    df = load_and_clean(raw_csv)
    assert "customer_id" not in df.columns


def test_signup_date_converted_to_numeric(raw_csv):
    df = load_and_clean(raw_csv)
    assert "signup_date" not in df.columns
    assert "signup_days" in df.columns
    assert pd.api.types.is_integer_dtype(df["signup_days"])


def test_rows_sorted_by_signup_days(raw_csv):
    df = load_and_clean(raw_csv)
    assert df["signup_days"].is_monotonic_increasing


def test_target_not_in_features(raw_csv):
    df = load_and_clean(raw_csv)
    X, y = get_features_and_target(df)
    assert "churned" not in X.columns
    assert y.name == "churned"


def test_features_and_target_aligned(raw_csv):
    df = load_and_clean(raw_csv)
    X, y = get_features_and_target(df)
    assert len(X) == len(y)
    assert list(X.index) == list(y.index)


def test_class_balance_report():
    y = pd.Series([0, 1, 1, 0, 1])
    report = class_balance_report(y)
    assert report["n_samples"] == 5
    assert abs(report["positive_rate"] - 0.6) < 1e-6
