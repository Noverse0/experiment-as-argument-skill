"""Tests for preprocessing module."""
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from src.preprocessing import (
    load_data,
    get_features_and_target,
    time_based_split,
    get_class_distribution,
    HONEST_FEATURES,
    TARGET,
)


@pytest.fixture
def sample_df():
    """Create a small sample dataframe for testing."""
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "signup_date": ["2023-01-15", "2023-06-10", "2023-10-20", "2023-11-05", "2023-12-01"],
        "tenure_months": [10, 5, 2, 1, 1],
        "monthly_spend": [50.0, 30.0, 100.0, 20.0, 15.0],
        "support_tickets": [2, 1, 5, 0, 1],
        "days_since_last_login": [5, 10, 50, 100, 80],
        "churned": [0, 0, 1, 1, 1],
    })


def test_load_data():
    """Test that data loads correctly."""
    data_file = Path(__file__).parent.parent / "churn.csv"
    df = load_data(str(data_file))
    assert len(df) > 0
    assert "churned" in df.columns
    assert "tenure_months" in df.columns


def test_get_features_and_target(sample_df):
    """Test feature/target extraction."""
    X, y = get_features_and_target(sample_df)

    assert X.shape == (5, 3)
    assert list(X.columns) == HONEST_FEATURES
    assert "days_since_last_login" not in X.columns
    assert len(y) == 5
    assert list(y) == [0, 0, 1, 1, 1]


def test_time_based_split(sample_df):
    """Test time-based split."""
    train, test = time_based_split(sample_df, test_month=10)

    assert len(train) == 2
    assert len(test) == 3

    train_months = pd.to_datetime(train["signup_date"]).dt.month
    test_months = pd.to_datetime(test["signup_date"]).dt.month

    assert all(train_months < 10)
    assert all(test_months >= 10)


def test_get_class_distribution(sample_df):
    """Test class distribution computation."""
    _, y = get_features_and_target(sample_df)
    dist = get_class_distribution(y)

    assert dist["n_samples"] == 5
    assert dist["n_churn"] == 3
    assert dist["n_no_churn"] == 2
    assert abs(dist["churn_rate"] - 0.6) < 0.01


def test_honest_features_excludes_leak():
    """Verify that HONEST_FEATURES does not include the leak."""
    assert "days_since_last_login" not in HONEST_FEATURES
    assert "tenure_months" in HONEST_FEATURES
    assert "monthly_spend" in HONEST_FEATURES
    assert "support_tickets" in HONEST_FEATURES
