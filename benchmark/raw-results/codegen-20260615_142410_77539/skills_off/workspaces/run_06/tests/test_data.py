"""Tests for data loading and preprocessing."""
import pytest
import pandas as pd
import tempfile
import os
from src.data import load_and_deduplicate, time_based_split, prepare_features


@pytest.fixture
def sample_churn_csv():
    """Create a temporary CSV with duplicate rows for testing."""
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 2, 4],
            "signup_date": [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-02",
                "2023-01-04",
            ],
            "tenure_months": [12, 24, 6, 24, 18],
            "monthly_spend": [100.0, 200.0, 50.0, 200.0, 150.0],
            "support_tickets": [1, 2, 0, 2, 1],
            "days_since_last_login": [5, 10, 3, 10, 7],
            "churned": [0, 1, 0, 1, 0],
        }
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        path = f.name
    yield path
    os.unlink(path)


def test_load_and_deduplicate(sample_churn_csv):
    """Test that duplicates are removed correctly."""
    df, n_dup = load_and_deduplicate(sample_churn_csv)

    # Started with 5 rows, 1 duplicate (row 3 is dup of row 1), ends with 4
    assert len(df) == 4
    assert n_dup == 1

    # Check that duplicate is removed (only one row with customer_id=2, tenure=24, etc.)
    row_2 = df[df["customer_id"] == 2]
    assert len(row_2) == 1


def test_time_based_split(sample_churn_csv):
    """Test that time-based split respects temporal ordering."""
    df, _ = load_and_deduplicate(sample_churn_csv)
    train, test = time_based_split(df, train_frac=0.7)

    # 4 rows * 0.7 = 2.8 → 2 train, 2 test
    assert len(train) == 2
    assert len(test) == 2

    # Train should have earlier dates than test
    max_train_date = train["signup_date"].max()
    min_test_date = test["signup_date"].min()
    assert max_train_date <= min_test_date


def test_prepare_features(sample_churn_csv):
    """Test that features are extracted and scaled correctly."""
    df, _ = load_and_deduplicate(sample_churn_csv)
    train, test = time_based_split(df, train_frac=0.7)

    X_train, X_test, y_train, y_test = prepare_features(train, test)

    # Check shapes
    assert X_train.shape[0] == len(train)
    assert X_train.shape[1] == 3  # 3 features
    assert X_test.shape[0] == len(test)
    assert y_train.shape[0] == len(train)
    assert y_test.shape[0] == len(test)

    # Check that X is scaled (mean ~0, std ~1)
    assert abs(X_train.mean()) < 0.1
    assert abs(X_train.std() - 1.0) < 0.1
