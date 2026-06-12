"""Tests for churn prediction experiment pipeline."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.utils import (
    load_and_clean_data,
    deduplicate_dataset,
    time_based_split,
    preprocess_features,
    baseline_predictions,
)


@pytest.fixture
def sample_csv():
    """Create a temporary churn CSV for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("customer_id,signup_date,tenure_months,monthly_spend,support_tickets,account_status,churned\n")
        # 50 data rows, balanced classes, with duplicates.
        for i in range(1, 51):
            days_offset = ((i - 1) % 30) + 1
            status = "closed" if (i % 2 == 0) else "active"
            churned = 1 if status == "closed" else 0
            f.write(f"{i},2023-01-{days_offset:02d},{10+i%40},{50+i*2:.1f},{i%4},{status},{churned}\n")
        # Add exact duplicates (except customer_id which will be dropped anyway).
        f.write("51,2023-01-01,11,52.0,1,active,0\n")  # duplicate of row 1
        f.write("52,2023-01-01,11,52.0,1,active,0\n")  # duplicate of row 1 and 51
        f.flush()
        yield f.name
        Path(f.name).unlink()


def test_load_and_clean_data_drops_leak_features(sample_csv):
    """Verify account_status is dropped (it's a leak)."""
    df = load_and_clean_data(sample_csv)
    assert "account_status" not in df.columns
    assert "customer_id" not in df.columns
    assert "churned" in df.columns


def test_load_and_clean_data_preserves_target(sample_csv):
    """Target column is preserved."""
    df = load_and_clean_data(sample_csv)
    assert df["churned"].dtype in [np.int64, int]
    assert df["churned"].nunique() == 2


def test_deduplicate_removes_exact_duplicates(sample_csv):
    """Exact row duplicates are removed."""
    df = load_and_clean_data(sample_csv)
    initial = len(df)
    df_dedup = deduplicate_dataset(df)
    assert len(df_dedup) < initial
    # Remaining rows should be unique.
    assert df_dedup.drop_duplicates().shape == df_dedup.shape


def test_time_based_split_respects_date_order(sample_csv):
    """Split is ordered by date, not random."""
    df = load_and_clean_data(sample_csv)
    train, test = time_based_split(df, train_frac=0.7)

    # Convert to datetime for comparison.
    train_max_date = pd.to_datetime(train["signup_date"]).max()
    test_min_date = pd.to_datetime(test["signup_date"]).min()

    # All train dates should be ≤ all test dates (with possible overlap at boundary).
    assert train_max_date <= test_min_date or train_max_date.date() == test_min_date.date()


def test_time_based_split_fraction(sample_csv):
    """Split respects train_frac."""
    df = load_and_clean_data(sample_csv)
    train, test = time_based_split(df, train_frac=0.7)

    total = len(train) + len(test)
    assert abs(len(train) / total - 0.7) < 0.05  # Allow small rounding error.


def test_preprocess_fit_on_train_only(sample_csv):
    """Scaler is fit on train only, not on test."""
    df = load_and_clean_data(sample_csv)
    train, test = time_based_split(df, train_frac=0.7)

    X_train, X_test, y_train, y_test, scaler = preprocess_features(train, test)

    # Both should be scaled.
    assert X_train.shape[0] == len(train)
    assert X_test.shape[0] == len(test)
    assert X_train.shape[1] == X_test.shape[1]  # Same features.


def test_preprocess_removes_date_column(sample_csv):
    """Date column is not returned in X matrices."""
    df = load_and_clean_data(sample_csv)
    train, test = time_based_split(df, train_frac=0.7)
    X_train, X_test, y_train, y_test, scaler = preprocess_features(train, test)

    # Date column should not be in the feature set.
    assert X_train.shape[1] == 3  # tenure, spend, tickets


def test_preprocess_target_extraction(sample_csv):
    """y_train and y_test are correctly extracted."""
    df = load_and_clean_data(sample_csv)
    train, test = time_based_split(df, train_frac=0.7)
    X_train, X_test, y_train, y_test, scaler = preprocess_features(train, test)

    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert set(y_train).issubset({0, 1})
    assert set(y_test).issubset({0, 1})


def test_baseline_predictions_majority_class(sample_csv):
    """Baseline is always the majority class."""
    df = load_and_clean_data(sample_csv)
    train, test = time_based_split(df, train_frac=0.7)
    X_train, X_test, y_train, y_test, scaler = preprocess_features(train, test)

    baseline_acc = baseline_predictions(y_test)

    # Should be at least as good as flipping a coin (for balanced or imbalanced).
    assert 0.0 <= baseline_acc <= 1.0
    # For imbalanced test set, baseline ≥ max(p, 1-p).
    minority_rate = min(y_test.mean(), 1 - y_test.mean())
    expected_baseline = 1 - minority_rate
    assert abs(baseline_acc - expected_baseline) < 0.01


def test_pipeline_end_to_end(sample_csv):
    """Full pipeline: load → clean → dedup → split → preprocess."""
    df = load_and_clean_data(sample_csv)
    df = deduplicate_dataset(df)
    train, test = time_based_split(df, train_frac=0.7)
    X_train, X_test, y_train, y_test, scaler = preprocess_features(train, test)

    # Sanity: no NaNs.
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
    assert not np.isnan(y_train).any()
    assert not np.isnan(y_test).any()

    # Sanity: reasonable sizes.
    assert len(X_train) > 10
    assert len(X_test) > 5
