"""Tests for dataset loading, splitting, and leak detection."""
import pytest
import numpy as np
import pandas as pd
from src.dataset import (
    load_data,
    check_duplicates,
    remove_duplicates,
    detect_leaks,
    prepare_features,
    time_based_split,
    get_split,
)


def test_load_data():
    """CSV loads successfully."""
    df = load_data("churn.csv")
    assert len(df) > 0
    assert "churned" in df.columns
    assert "tenure_months" in df.columns
    assert "days_since_last_login" in df.columns


def test_check_duplicates():
    """Duplicate detection works."""
    df = load_data("churn.csv")
    dup_count = check_duplicates(df)
    # Dataset includes 200 exact duplicates by design
    assert dup_count == 200


def test_remove_duplicates():
    """Duplicates are removed correctly."""
    df = load_data("churn.csv")
    before = len(df)
    df_deduped = remove_duplicates(df)
    after = len(df_deduped)
    assert after == before - 200
    # No duplicates remain
    assert df_deduped.duplicated().sum() == 0


def test_detect_leaks():
    """Leak detection identifies days_since_last_login."""
    df = load_data("churn.csv")
    leaks = detect_leaks(df)
    # Should detect days_since_last_login as a leak
    assert len(leaks) > 0
    assert any("days_since_last_login" in leak for leak in leaks)


def test_prepare_features_clean():
    """Feature preparation drops leaked features when requested."""
    df = load_data("churn.csv")
    X, names = prepare_features(df, drop_leaks=True)
    # Should have 3 clean features
    assert list(names) == ["tenure_months", "monthly_spend", "support_tickets"]
    assert X.shape[1] == 3
    assert "days_since_last_login" not in names


def test_prepare_features_with_leaks():
    """Feature preparation can include leaked features."""
    df = load_data("churn.csv")
    X, names = prepare_features(df, drop_leaks=False)
    # Should have 3 clean + 1 leaked = 4 features
    assert X.shape[1] == 4
    assert "days_since_last_login" in names


def test_time_based_split():
    """Time-based split respects temporal order."""
    df = load_data("churn.csv")
    train, test = time_based_split(df, temporal_col="signup_date", train_ratio=0.8)

    assert len(train) + len(test) == len(df)
    assert len(train) > len(test)

    # Train signup dates should be before test
    max_train_date = train["signup_date"].max()
    min_test_date = test["signup_date"].min()
    assert max_train_date <= min_test_date


def test_get_split_deterministic():
    """Same parameters produce identical splits."""
    X1_train, X1_test, y1_train, y1_test, _ = get_split("churn.csv", drop_leaks=True)
    X2_train, X2_test, y2_train, y2_test, _ = get_split("churn.csv", drop_leaks=True)

    # Exact same splits
    assert np.allclose(X1_train.values, X2_train.values)
    assert np.allclose(X1_test.values, X2_test.values)
    assert np.array_equal(y1_train, y2_train)
    assert np.array_equal(y1_test, y2_test)


def test_get_split_shapes():
    """Split produces valid train/test sets."""
    X_train, X_test, y_train, y_test, feature_names = get_split("churn.csv")

    # Basic shape checks
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    assert X_train.shape[1] == X_test.shape[1]
    assert len(feature_names) == X_train.shape[1]

    # Train should be ~80% of total
    total = len(X_train) + len(X_test)
    assert 0.75 < len(X_train) / total < 0.85


def test_no_leakage_across_split():
    """Exact duplicates don't straddle train/test."""
    # Load with full dataset (including duplicates)
    df = load_data("churn.csv")

    # Split without dedup (bad)
    train_bad, test_bad = time_based_split(df, train_ratio=0.8)

    # Check if any duplicates straddle
    train_set = set(map(tuple, train_bad.drop(columns=["customer_id", "signup_date"]).values))
    test_set = set(map(tuple, test_bad.drop(columns=["customer_id", "signup_date"]).values))
    overlap = train_set & test_set

    # With duplicates, some exact matches will exist (by design of dataset)
    # After dedup and split, they shouldn't
    df_dedup = remove_duplicates(df)
    train_good, test_good = time_based_split(df_dedup, train_ratio=0.8)

    train_set_good = set(map(tuple, train_good.drop(columns=["customer_id", "signup_date"]).values))
    test_set_good = set(map(tuple, test_good.drop(columns=["customer_id", "signup_date"]).values))
    overlap_good = train_set_good & test_set_good

    # After dedup, overlap should be zero (no exact duplicates)
    assert len(overlap_good) == 0
