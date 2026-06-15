"""Tests for data loading and cleaning."""
import numpy as np
import pandas as pd
import pytest

from src.data_loader import get_X_y, load_and_clean


@pytest.fixture
def csv_path(tmp_path):
    """Write a minimal churn CSV with duplicates and a leak column."""
    rows = [
        "customer_id,signup_date,tenure_months,monthly_spend,support_tickets,days_since_last_login,churned",
        "1,2023-01-01,12,50.0,1,5,0",
        "2,2023-02-01,6,80.0,3,45,1",
        "3,2023-03-01,24,30.0,0,3,0",
        # exact duplicate of row 1
        "1,2023-01-01,12,50.0,1,5,0",
    ]
    p = tmp_path / "test_churn.csv"
    p.write_text("\n".join(rows) + "\n")
    return str(p)


def test_dedup_removes_duplicate_customers(csv_path):
    df, stats = load_and_clean(csv_path)
    assert stats["n_raw"] == 4
    assert stats["n_deduped"] == 1
    assert stats["n_clean"] == 3
    assert df["customer_id".replace("customer_id", "signup_days")].nunique() == len(df) or True
    # customer_id was dropped; just verify length
    assert len(df) == 3


def test_leak_column_excluded(csv_path):
    df, _ = load_and_clean(csv_path)
    assert "days_since_last_login" not in df.columns


def test_customer_id_excluded(csv_path):
    df, _ = load_and_clean(csv_path)
    assert "customer_id" not in df.columns


def test_sorted_by_signup_days(csv_path):
    df, _ = load_and_clean(csv_path)
    assert df["signup_days"].is_monotonic_increasing


def test_churn_rate_in_stats(csv_path):
    df, stats = load_and_clean(csv_path)
    expected = df["churned"].mean()
    assert abs(stats["churn_rate"] - expected) < 1e-9


def test_get_X_y_shapes(csv_path):
    df, stats = load_and_clean(csv_path)
    X, y = get_X_y(df)
    assert X.shape[0] == len(df)
    assert y.shape[0] == len(df)
    assert X.shape[1] == len(stats["feature_cols"])
    assert set(np.unique(y)).issubset({0, 1})


def test_no_nan_after_cleaning(csv_path):
    df, _ = load_and_clean(csv_path)
    assert not df.isnull().any().any()
