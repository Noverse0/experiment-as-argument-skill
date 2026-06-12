"""Tests for data loading, deduplication, and splitting."""
import pytest
import pandas as pd
import numpy as np
from src.data import load_and_deduplicate, time_based_split, select_features, preprocess, prepare_data


@pytest.fixture
def sample_csv(tmp_path):
    """Create a temporary CSV file with sample data."""
    data = {
        "customer_id": [1, 2, 3, 1, 2],  # Last two are duplicates
        "signup_date": ["2023-01-01", "2023-01-15", "2023-02-01", "2023-01-01", "2023-01-15"],
        "tenure_months": [10, 20, 30, 10, 20],
        "monthly_spend": [50.0, 100.0, 150.0, 50.0, 100.0],
        "support_tickets": [1, 2, 3, 1, 2],
        "account_status": ["active", "closed", "active", "active", "closed"],
        "churned": [0, 1, 0, 0, 1],
    }
    df = pd.DataFrame(data)
    csv_file = tmp_path / "test.csv"
    df.to_csv(csv_file, index=False)
    return str(csv_file)


def test_deduplication(sample_csv):
    """Test that exact duplicates are removed."""
    df = load_and_deduplicate(sample_csv)
    assert len(df) == 3, "Should remove 2 duplicate rows"


def test_deduplication_preserves_data(sample_csv):
    """Test that deduplication preserves unique rows."""
    df = load_and_deduplicate(sample_csv)
    assert set(df["customer_id"].values) == {1, 2, 3}


def test_time_based_split(sample_csv):
    """Test that time-based split respects temporal ordering."""
    df = load_and_deduplicate(sample_csv)
    train, test = time_based_split(df, date_col="signup_date", train_fraction=0.6)

    # Check split preserves all rows
    assert len(train) + len(test) == len(df)

    # Check temporal ordering: all train dates should be <= test dates
    if len(test) > 0 and len(train) > 0:
        max_train_date = pd.to_datetime(train["signup_date"]).max()
        min_test_date = pd.to_datetime(test["signup_date"]).min()
        assert max_train_date <= min_test_date, "Train and test time windows should not overlap"


def test_select_features():
    """Test that feature selection excludes leaky columns."""
    df = pd.DataFrame({
        "tenure_months": [10, 20],
        "monthly_spend": [50.0, 100.0],
        "support_tickets": [1, 2],
        "account_status": ["active", "closed"],
        "customer_id": [1, 2],
        "signup_date": ["2023-01-01", "2023-01-15"],
        "churned": [0, 1],
    })

    features = select_features(df)
    assert list(features.columns) == ["tenure_months", "monthly_spend", "support_tickets"]
    assert "account_status" not in features.columns, "account_status should be excluded (leak)"
    assert "customer_id" not in features.columns, "customer_id should be excluded (identifier)"


def test_preprocess_scaling():
    """Test that preprocessing scales features correctly."""
    X_train = pd.DataFrame({
        "tenure_months": [10, 20, 30],
        "monthly_spend": [50.0, 100.0, 150.0],
        "support_tickets": [1, 2, 3],
    })
    X_test = pd.DataFrame({
        "tenure_months": [15, 25],
        "monthly_spend": [75.0, 125.0],
        "support_tickets": [1, 2],
    })

    X_train_scaled, X_test_scaled = preprocess(X_train, X_test)

    # Check scaling: train should have mean ~0 and std ~1
    assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(X_train_scaled.std(axis=0), 1)

    # Check shapes
    assert X_train_scaled.shape == (3, 3)
    assert X_test_scaled.shape == (2, 3)


def test_preprocess_test_uses_train_scaler():
    """Test that test set is scaled using train statistics (no data leakage)."""
    X_train = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0],
    })
    X_test = pd.DataFrame({
        "feat1": [10.0, 100.0],  # Very different from train
    })

    X_train_scaled, X_test_scaled = preprocess(X_train, X_test)

    # Train should be centered at 0
    assert np.isclose(X_train_scaled.mean(), 0)

    # Test should NOT be centered at 0 (because scaler was fit on train)
    test_mean = X_test_scaled.mean()
    assert not np.isclose(test_mean, 0), "Test data should be transformed with train scaler, not re-centered"


def test_prepare_data_full_pipeline(sample_csv):
    """Test the full data preparation pipeline."""
    X_train, X_test, y_train, y_test = prepare_data(sample_csv)

    # Check shapes
    assert X_train.shape[1] == 3, "Should have 3 features"
    assert X_test.shape[1] == 3
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]

    # Check that y values are binary
    assert set(y_train).issubset({0, 1})
    assert set(y_test).issubset({0, 1})

    # Check that train and test do not have same indices (no overlap)
    # This is ensured by time-based split


def test_no_nan_in_output(sample_csv):
    """Test that the pipeline produces no NaN values."""
    X_train, X_test, y_train, y_test = prepare_data(sample_csv)

    assert not np.isnan(X_train).any(), "X_train should have no NaNs"
    assert not np.isnan(X_test).any(), "X_test should have no NaNs"
