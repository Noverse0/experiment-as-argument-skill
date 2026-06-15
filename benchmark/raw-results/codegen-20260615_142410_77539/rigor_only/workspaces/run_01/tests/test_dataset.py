"""Tests for dataset loading and splitting."""
import pandas as pd
import numpy as np
import pytest
from src.dataset import (
    load_and_deduplicate,
    time_based_split,
    get_feature_columns,
    prepare_features,
)


def test_load_and_deduplicate():
    """Test deduplication removes exact duplicates."""
    df = load_and_deduplicate("churn.csv")
    assert len(df) > 0
    assert 'churned' in df.columns
    assert 'signup_date' in df.columns

    # Should have removed some duplicates
    initial = df.attrs['initial_rows']
    removed = df.attrs['duplicates_removed']
    assert removed > 0, "Expected to find and remove duplicate rows"
    assert len(df) == initial - removed


def test_time_based_split():
    """Test time-based split sorts by signup_date."""
    df = load_and_deduplicate("churn.csv")
    train, test = time_based_split(df, test_fraction=0.2)

    total = len(train) + len(test)
    assert total == len(df), "Split should preserve all rows"

    # Check split ratio
    expected_train = int(len(df) * 0.8)
    assert len(train) == expected_train, f"Expected ~{expected_train} train rows"

    # Check temporal order: train dates should be earlier than test dates
    train_dates = pd.to_datetime(train['signup_date'])
    test_dates = pd.to_datetime(test['signup_date'])
    assert train_dates.max() <= test_dates.min(), "Train dates should precede test dates"


def test_no_duplicates_across_split():
    """Test that exact duplicates don't straddle train/test boundary."""
    df = load_and_deduplicate("churn.csv")
    train, test = time_based_split(df, test_fraction=0.2)

    # Combine train and test, find any exact duplicates
    combined = pd.concat([train, test], ignore_index=True)
    duplicates = combined[combined.duplicated(subset=train.columns, keep=False)]

    # There should be no duplicates after dedup + time split
    assert len(duplicates) == 0, "Duplicates should not cross train/test boundary"


def test_get_feature_columns():
    """Test feature column selection."""
    features = get_feature_columns()
    assert 'tenure_months' in features
    assert 'monthly_spend' in features
    assert 'support_tickets' in features
    # Leak should not be included
    assert 'days_since_last_login' not in features


def test_prepare_features():
    """Test feature and target extraction."""
    df = load_and_deduplicate("churn.csv")
    train, _ = time_based_split(df)

    X, y = prepare_features(train)

    assert X.shape[0] == len(train)
    assert X.shape[1] == len(get_feature_columns())
    assert y.shape[0] == len(train)
    assert set(np.unique(y)) <= {0, 1}  # Binary target


def test_feature_dtypes():
    """Test features are properly typed for sklearn."""
    df = load_and_deduplicate("churn.csv")
    train, _ = time_based_split(df)

    X, y = prepare_features(train)

    assert X.dtype == np.float32
    assert y.dtype == np.int32


def test_no_nan_features():
    """Test no NaN values in features."""
    df = load_and_deduplicate("churn.csv")
    train, _ = time_based_split(df)

    X, y = prepare_features(train)

    assert not np.any(np.isnan(X)), "Features should not contain NaN"
    assert not np.any(np.isnan(y)), "Target should not contain NaN"


def test_class_imbalance():
    """Test that target is imbalanced (as expected for churn)."""
    df = load_and_deduplicate("churn.csv")
    train, _ = time_based_split(df)

    X, y = prepare_features(train)
    churn_rate = y.mean()

    # Churn should be minority class
    assert 0 < churn_rate < 0.5, f"Expected minority churn (got {churn_rate:.2%})"
