"""Tests for data pipeline: load, dedupe, split, preprocess."""
import pytest
import pandas as pd
import numpy as np
from src.pipeline import (
    load_and_clean,
    deduplicate,
    time_based_split,
    check_no_leakage,
    preprocess,
    check_class_balance,
)


@pytest.fixture
def sample_df():
    """Create a tiny sample dataset for testing."""
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 1, 2],
            "signup_date": [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-01",
                "2023-01-02",
            ],
            "tenure_months": [10, 20, 30, 10, 20],
            "monthly_spend": [100.0, 200.0, 300.0, 100.0, 200.0],
            "support_tickets": [1, 2, 3, 1, 2],
            "account_status": ["active", "closed", "active", "active", "closed"],
            "churned": [0, 1, 0, 0, 1],
        }
    )
    # Remove account_status before returning (it's dropped in load_and_clean)
    return df.drop(columns=["account_status"])


def test_load_and_clean_drops_account_status(tmp_path):
    """account_status must be dropped to prevent leakage."""
    df = pd.DataFrame(
        {
            "customer_id": [1, 2],
            "signup_date": ["2023-01-01", "2023-01-02"],
            "tenure_months": [10, 20],
            "monthly_spend": [100.0, 200.0],
            "support_tickets": [1, 2],
            "account_status": ["active", "closed"],
            "churned": [0, 1],
        }
    )
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)

    result = load_and_clean(str(csv_path))
    assert "account_status" not in result.columns
    assert "churned" in result.columns
    assert len(result) == 2


def test_deduplicate_removes_exact_duplicates(sample_df):
    """Exact duplicates must be removed."""
    before = len(sample_df)
    result = deduplicate(sample_df)
    after = len(result)
    # sample_df has 5 rows, with rows 0,3 identical and 1,4 identical
    # so 2 exact duplicates = 3 unique rows
    assert after == 3
    assert before - after == 2


def test_time_based_split(sample_df):
    """Split should be ordered by signup_date."""
    train, test = time_based_split(sample_df, train_ratio=0.6)
    # 5 rows total, 60% train = 3 rows, 40% test = 2 rows
    assert len(train) == 3
    assert len(test) == 2
    # Train should have earlier signup dates
    assert max(train["signup_date"]) <= min(test["signup_date"])


def test_check_no_leakage(sample_df):
    """Same customer in both train/test should raise error."""
    # Create a case where the same customer_id is in both
    train = sample_df.iloc[:2].copy()
    test = sample_df.iloc[2:].copy()

    # Manually inject a duplicate customer_id
    test.loc[test.index[0], "customer_id"] = train.iloc[0]["customer_id"]

    with pytest.raises(ValueError, match="Data leakage"):
        check_no_leakage(train, test)


def test_check_no_leakage_pass(sample_df):
    """With disjoint customer_ids, should pass."""
    train = sample_df.iloc[:2].copy()
    test = sample_df.iloc[2:].copy()
    # Make customer_ids disjoint
    test["customer_id"] = [100, 101, 102]
    # No error should be raised
    check_no_leakage(train, test)


def test_preprocess_shapes(sample_df):
    """Preprocess should return properly shaped arrays."""
    # Remove duplicates and split
    sample_df = deduplicate(sample_df)
    train, test = time_based_split(sample_df, train_ratio=0.6)

    X_train, X_test, y_train, y_test = preprocess(train, test)

    # Check shapes
    assert X_train.shape[0] == len(train)
    assert X_test.shape[0] == len(test)
    assert X_train.shape[1] == 3  # tenure, spend, tickets
    assert X_test.shape[1] == 3
    assert len(y_train) == len(train)
    assert len(y_test) == len(test)


def test_preprocess_scales_features(sample_df):
    """Features should be scaled (mean ~0, std ~1 for multi-sample sets)."""
    # Use more data to ensure meaningful std calculation
    extended_df = pd.concat([sample_df] * 5, ignore_index=True)
    extended_df["customer_id"] = range(len(extended_df))

    train, test = time_based_split(extended_df, train_ratio=0.6)

    X_train, X_test, y_train, y_test = preprocess(train, test)

    # Check that train features are approximately scaled
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)

    # Mean should be close to 0
    assert np.allclose(train_mean, 0, atol=1e-10)
    # Std should be close to 1
    assert np.allclose(train_std, 1, atol=1e-10)


def test_class_balance(sample_df):
    """Class balance check should run without error."""
    sample_df = deduplicate(sample_df)
    train, test = time_based_split(sample_df, train_ratio=0.6)
    X_train, X_test, y_train, y_test = preprocess(train, test)

    # Should not raise
    check_class_balance(y_train, y_test)
