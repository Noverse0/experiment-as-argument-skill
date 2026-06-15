"""Unit tests for data loading, cleaning, and feature engineering."""

import pandas as pd
import pytest

from src.pipeline import (
    ID_COLS,
    LEAK_COLS,
    TARGET,
    engineer_features,
    load_and_clean,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "signup_date": [
                "2023-01-01",
                "2023-06-01",
                "2024-01-01",
                "2024-06-01",
                "2025-01-01",
            ],
            "tenure_months": [12, 24, 6, 36, 18],
            "monthly_spend": [50.0, 100.0, 25.0, 75.0, 60.0],
            "support_tickets": [0, 2, 1, 3, 0],
            "days_since_last_login": [5, 60, 3, 45, 7],
            "churned": [0, 1, 0, 1, 0],
        }
    )


@pytest.fixture
def csv_no_dupes(tmp_path, sample_df):
    p = tmp_path / "churn.csv"
    sample_df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def csv_with_dupes(tmp_path, sample_df):
    df = pd.concat([sample_df, sample_df.iloc[:2]], ignore_index=True)
    p = tmp_path / "churn_dupes.csv"
    df.to_csv(p, index=False)
    return str(p)


# ── load_and_clean ────────────────────────────────────────────────────────────

def test_load_no_dupes_count(csv_no_dupes):
    df, n_dupes = load_and_clean(csv_no_dupes)
    assert n_dupes == 0
    assert len(df) == 5


def test_load_removes_duplicates(csv_with_dupes):
    df, n_dupes = load_and_clean(csv_with_dupes)
    assert n_dupes == 2
    assert len(df) == 5


def test_load_returns_all_original_columns(csv_no_dupes):
    df, _ = load_and_clean(csv_no_dupes)
    expected = {
        "customer_id", "signup_date", "tenure_months", "monthly_spend",
        "support_tickets", "days_since_last_login", "churned",
    }
    assert expected.issubset(set(df.columns))


# ── engineer_features ─────────────────────────────────────────────────────────

def test_leak_columns_dropped(sample_df):
    df = engineer_features(sample_df)
    for col in LEAK_COLS:
        assert col not in df.columns, f"Leak column '{col}' must be dropped"


def test_id_columns_dropped(sample_df):
    df = engineer_features(sample_df)
    for col in ID_COLS:
        assert col not in df.columns, f"ID column '{col}' must be dropped"


def test_date_converted_to_numeric(sample_df):
    df = engineer_features(sample_df)
    assert "signup_date" not in df.columns
    assert "signup_days" in df.columns
    assert df["signup_days"].dtype.kind in ("i", "f")


def test_signup_days_monotonic_with_sorted_input(sample_df):
    df = engineer_features(sample_df.sort_values("signup_date").reset_index(drop=True))
    assert df["signup_days"].is_monotonic_increasing


def test_signup_days_non_negative(sample_df):
    df = engineer_features(sample_df)
    assert (df["signup_days"] >= 0).all()


def test_no_missing_values(sample_df):
    df = engineer_features(sample_df)
    assert not df.isnull().any().any()


def test_target_preserved(sample_df):
    df = engineer_features(sample_df)
    assert TARGET in df.columns


def test_only_legitimate_features_remain(sample_df):
    df = engineer_features(sample_df)
    expected_feature_cols = {
        "tenure_months", "monthly_spend", "support_tickets",
        "signup_days", TARGET,
    }
    assert set(df.columns) == expected_feature_cols
