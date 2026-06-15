"""Tests for data loading and preprocessing."""
import pytest
import pandas as pd
import numpy as np
from src.data import (
    load_and_validate, find_duplicates, deduplicate_before_split,
    split_train_test, preprocess, get_class_distribution, FEATURE_NAMES
)


def create_dummy_csv(tmp_path):
    """Create a minimal dummy churn CSV for testing."""
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "signup_date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
        "tenure_months": [6, 12, 24, 36, 48],
        "monthly_spend": [50.0, 75.0, 100.0, 125.0, 150.0],
        "support_tickets": [0, 1, 2, 3, 4],
        "days_since_last_login": [5, 10, 15, 20, 25],
        "churned": [0, 0, 1, 1, 1],
    })
    csv_path = tmp_path / "test_churn.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_load_and_validate(tmp_path):
    """Test CSV loading."""
    csv_path = create_dummy_csv(tmp_path)
    df = load_and_validate(str(csv_path))
    assert len(df) == 5
    assert "churned" in df.columns


def test_load_and_validate_missing_columns(tmp_path):
    """Test that missing columns raise error."""
    bad_df = pd.DataFrame({"customer_id": [1, 2], "churned": [0, 1]})
    bad_path = tmp_path / "bad.csv"
    bad_df.to_csv(bad_path, index=False)

    with pytest.raises(ValueError):
        load_and_validate(str(bad_path))


def test_deduplicate_before_split(tmp_path):
    """Test deduplication."""
    csv_path = create_dummy_csv(tmp_path)
    df = load_and_validate(str(csv_path))

    # Add one duplicate
    dup = df.iloc[0:1].copy()
    df = pd.concat([df, dup], ignore_index=True)
    assert len(df) == 6

    df_dedup, removed = deduplicate_before_split(df)
    assert len(df_dedup) == 5
    assert removed == 1


def test_split_train_test(tmp_path):
    """Test stratified train/test split."""
    csv_path = create_dummy_csv(tmp_path)
    df = load_and_validate(str(csv_path))
    X_train, X_test, y_train, y_test = split_train_test(df, test_size=0.4, random_state=42)

    assert len(X_train) + len(X_test) == len(df)
    assert len(X_train) == 3
    assert len(X_test) == 2
    assert all(col in X_train.columns for col in FEATURE_NAMES)


def test_preprocess_with_scaling(tmp_path):
    """Test preprocessing with scaling."""
    csv_path = create_dummy_csv(tmp_path)
    df = load_and_validate(str(csv_path))
    X_train, X_test, y_train, y_test = split_train_test(df, test_size=0.4, random_state=42)

    X_train_scaled, X_test_scaled, scaler = preprocess(X_train, X_test, use_scaling=True)

    assert scaler is not None
    assert X_train_scaled.shape[1] == len(FEATURE_NAMES)
    # Scaled train should have mean ~0, std ~1 per feature
    np.testing.assert_array_almost_equal(X_train_scaled.mean(axis=0), 0, decimal=1)


def test_preprocess_without_scaling(tmp_path):
    """Test preprocessing without scaling."""
    csv_path = create_dummy_csv(tmp_path)
    df = load_and_validate(str(csv_path))
    X_train, X_test, y_train, y_test = split_train_test(df, test_size=0.4, random_state=42)

    X_train_unscaled, X_test_unscaled, scaler = preprocess(X_train, X_test, use_scaling=False)

    assert scaler is None
    assert X_train_unscaled.shape == (len(X_train), len(FEATURE_NAMES))


def test_get_class_distribution(tmp_path):
    """Test class distribution calculation."""
    csv_path = create_dummy_csv(tmp_path)
    df = load_and_validate(str(csv_path))
    y = df["churned"]

    dist = get_class_distribution(y)
    assert dist["class_0_count"] == 2
    assert dist["class_1_count"] == 3
    assert dist["class_1_rate"] == pytest.approx(0.6, abs=0.01)
    assert dist["total"] == 5
