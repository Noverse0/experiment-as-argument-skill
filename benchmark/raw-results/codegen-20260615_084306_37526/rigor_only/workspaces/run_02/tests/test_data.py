"""Tests for data loading and cleaning."""
import pandas as pd
import numpy as np
import pytest
from src.data import clean_data, get_X_y, FEATURES, TARGET, LEAK_COLS


@pytest.fixture
def sample_df():
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 1, 2],  # rows 3,4 are dupes of rows 0,1
        "signup_date": ["2023-01-01"] * 5,
        "tenure_months": [12, 24, 36, 12, 24],
        "monthly_spend": [50.0, 80.0, 120.0, 50.0, 80.0],
        "support_tickets": [1, 2, 0, 1, 2],
        "days_since_last_login": [5, 60, 3, 5, 60],
        "churned": [0, 1, 0, 0, 1],
    })
    return df


def test_clean_data_removes_duplicates(sample_df):
    cleaned, stats = clean_data(sample_df)
    assert len(cleaned) == 3
    assert stats["n_duplicates_removed"] == 2
    assert stats["n_after_dedup"] == 3


def test_clean_data_resets_index(sample_df):
    cleaned, _ = clean_data(sample_df)
    assert list(cleaned.index) == list(range(len(cleaned)))


def test_clean_data_no_dupes_unchanged():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
    })
    cleaned, stats = clean_data(df)
    assert stats["n_duplicates_removed"] == 0
    assert len(cleaned) == 3


def test_get_X_y_returns_correct_features(sample_df):
    cleaned, _ = clean_data(sample_df)
    X, y = get_X_y(cleaned)
    assert list(X.columns) == FEATURES
    assert y.name == TARGET


def test_leak_col_not_in_features():
    """days_since_last_login must not appear in the feature set."""
    for col in LEAK_COLS:
        assert col not in FEATURES, f"Leak column '{col}' found in FEATURES"


def test_customer_id_not_in_features():
    assert "customer_id" not in FEATURES


def test_get_X_y_correct_length(sample_df):
    cleaned, _ = clean_data(sample_df)
    X, y = get_X_y(cleaned)
    assert len(X) == len(y) == len(cleaned)
