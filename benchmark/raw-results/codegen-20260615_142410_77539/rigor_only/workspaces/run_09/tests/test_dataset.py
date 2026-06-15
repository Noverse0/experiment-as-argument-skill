"""Tests for data loading and preprocessing."""
import pytest
import pandas as pd
import numpy as np
from src.dataset import (
    load_churn_data,
    deduplicate,
    split_by_time,
    get_features_and_target
)


@pytest.fixture
def sample_data():
    """Create a small sample churn dataset for testing."""
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "signup_date": [
            "2023-01-01",
            "2023-02-01",
            "2023-03-01",
            "2023-04-01",
            "2023-05-01"
        ],
        "tenure_months": [10, 20, 30, 40, 50],
        "monthly_spend": [100.0, 200.0, 150.0, 250.0, 300.0],
        "support_tickets": [1, 2, 3, 2, 1],
        "days_since_last_login": [5, 10, 15, 20, 25],
        "churned": [0, 1, 0, 1, 0]
    })


def test_load_churn_data(tmp_path):
    """Test loading churn data from CSV."""
    csv_file = tmp_path / "test_churn.csv"
    df = pd.DataFrame({
        "customer_id": [1, 2],
        "signup_date": ["2023-01-01", "2023-02-01"],
        "tenure_months": [10, 20],
        "monthly_spend": [100.0, 200.0],
        "support_tickets": [1, 2],
        "days_since_last_login": [5, 10],
        "churned": [0, 1]
    })
    df.to_csv(csv_file, index=False)

    loaded = load_churn_data(str(csv_file))
    assert len(loaded) == 2
    assert "churned" in loaded.columns
    assert pd.api.types.is_datetime64_any_dtype(loaded["signup_date"])


def test_deduplicate(sample_data):
    """Test deduplication logic."""
    df_dup = pd.concat([sample_data, sample_data.iloc[0:2]], ignore_index=True)
    df_clean, removed = deduplicate(df_dup)

    assert len(df_dup) == 7
    assert len(df_clean) == 5
    assert removed == 2


def test_split_by_time(sample_data):
    """Test time-based split respects chronological order."""
    train, test = split_by_time(sample_data, train_ratio=0.8, seed=42)

    assert len(train) + len(test) == len(sample_data)
    assert len(train) == 4
    assert len(test) == 1

    # Verify dates are in the right order (train should have earlier dates)
    max_train_date = train["signup_date"].max()
    min_test_date = test["signup_date"].min()
    assert max_train_date <= min_test_date


def test_get_features_and_target(sample_data):
    """Test feature/target extraction with leakage handling."""
    X, y = get_features_and_target(sample_data, drop_leakage=True)

    # With drop_leakage=True, days_since_last_login should be gone
    assert "days_since_last_login" not in X.columns
    assert "customer_id" not in X.columns
    assert "signup_date" not in X.columns
    assert "churned" not in X.columns

    # Should have the honest features
    expected_features = ["tenure_months", "monthly_spend", "support_tickets"]
    assert list(X.columns) == expected_features

    assert len(X) == len(y)
    assert len(y) == 5


def test_get_features_with_leakage(sample_data):
    """Test feature extraction includes leakage feature when requested."""
    X, y = get_features_and_target(sample_data, drop_leakage=False)

    assert "days_since_last_login" in X.columns
    assert len(X.columns) == 4
