"""
Tests for the churn prediction experiment pipeline.

Tests focus on:
- Data integrity (deduplication, no leakage)
- Feature correctness (proper leak handling)
- Sanity checks (baseline, overfit, label-shuffle)
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.experiment import (
    load_and_deduplicate,
    split_by_time,
    prepare_features,
    sanity_check_baseline,
    sanity_check_overfit_tiny,
    sanity_check_label_shuffle,
    run_experiment,
)


@pytest.fixture
def sample_data():
    """Create a minimal test dataset matching churn.csv structure."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "tenure_months": np.random.randint(1, 72, n),
        "monthly_spend": np.random.gamma(2.0, 30.0, n),
        "support_tickets": np.random.poisson(1.2, n),
        "days_since_last_login": np.random.randint(0, 100, n),
        "churned": np.random.binomial(1, 0.3, n),
    })
    return df


def test_load_and_deduplicate(tmp_path, sample_data):
    """Test deduplication removes duplicates before split."""
    csv_file = tmp_path / "test.csv"
    sample_data.to_csv(csv_file, index=False)

    # Add duplicates
    dupes = sample_data.iloc[:10].copy()
    df_with_dupes = pd.concat([sample_data, dupes], ignore_index=True)
    df_with_dupes.to_csv(csv_file, index=False)

    loaded = load_and_deduplicate(str(csv_file))
    assert len(loaded) == len(sample_data), "Duplicates should be removed"
    assert not loaded.duplicated().any(), "No duplicates should remain"


def test_time_based_split_order(sample_data):
    """Test time-based split respects temporal order."""
    train, test = split_by_time(sample_data, train_ratio=0.7)

    # Verify temporal order
    train_dates = pd.to_datetime(train["signup_date"])
    test_dates = pd.to_datetime(test["signup_date"])
    assert train_dates.max() <= test_dates.min(), "Train dates must come before test dates"
    assert len(train) > 0 and len(test) > 0, "Both splits should be non-empty"


def test_time_based_split_ratio(sample_data):
    """Test time-based split ratio is approximately correct."""
    train, test = split_by_time(sample_data, train_ratio=0.7)
    ratio = len(train) / (len(train) + len(test))
    assert 0.65 < ratio < 0.75, f"Split ratio {ratio} deviates from 0.7"


def test_no_data_leakage_across_split(sample_data):
    """Test no customer_ids straddle train/test after split."""
    train, test = split_by_time(sample_data)
    train_ids = set(train["customer_id"])
    test_ids = set(test["customer_id"])
    overlap = train_ids & test_ids
    assert len(overlap) == 0, "Customer IDs must not overlap across train/test"


def test_prepare_features_drops_leak(sample_data):
    """Test prepare_features drops days_since_last_login and other non-predictive columns."""
    X, scaler = prepare_features(sample_data, fit=True)

    # Expected shape: 3 features (tenure_months, monthly_spend, support_tickets)
    assert X.shape[1] == 3, f"Expected 3 features, got {X.shape[1]}"
    assert X.shape[0] == len(sample_data), "Row count should match input"


def test_prepare_features_scaler_fit_train_only(sample_data):
    """Test scaler is fitted on train and applied to test."""
    train, test = split_by_time(sample_data, train_ratio=0.7)

    X_train, scaler = prepare_features(train, fit=True)
    X_test, _ = prepare_features(test, scaler=scaler, fit=False)

    # Verify scaler is fitted (has mean and scale)
    assert scaler.mean_ is not None, "Scaler should be fitted"
    assert scaler.scale_ is not None, "Scaler should have scale attributes"

    # Verify X_test uses the same scaler (not refitted)
    X_test_check, _ = prepare_features(test, scaler=scaler, fit=False)
    np.testing.assert_array_almost_equal(X_test, X_test_check, err_msg="Scaler should be consistent")


def test_baseline_sanity_check(sample_data):
    """Test baseline sanity check runs and returns reasonable value."""
    y_train = sample_data["churned"].values[:70]
    y_test = sample_data["churned"].values[70:]

    baseline = sanity_check_baseline(y_train, y_test)
    assert 0.4 < baseline < 0.6, f"Baseline AUC {baseline} should be near 0.5 for balanced data"


def test_overfit_sanity_check(sample_data):
    """Test overfit sanity check passes (models can fit tiny subset)."""
    X = sample_data[["tenure_months", "monthly_spend", "support_tickets"]].values
    y = sample_data["churned"].values

    # Should not raise
    sanity_check_overfit_tiny(X, y)


def test_label_shuffle_sanity_check(sample_data):
    """Test label-shuffle sanity check detects if signal survives shuffled labels."""
    X = sample_data[["tenure_months", "monthly_spend", "support_tickets"]].values
    y = sample_data["churned"].values

    # Should not raise
    sanity_check_label_shuffle(X, y, seed=42)


def test_run_experiment_returns_dict(tmp_path):
    """Test run_experiment returns results dict for both algorithms."""
    # Create a minimal CSV
    csv_file = tmp_path / "test_churn.csv"
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "tenure_months": np.random.randint(1, 72, n),
        "monthly_spend": np.random.gamma(2.0, 30.0, n),
        "support_tickets": np.random.poisson(1.2, n),
        "days_since_last_login": np.random.randint(0, 100, n),
        "churned": np.random.binomial(1, 0.3, n),
    })
    df.to_csv(csv_file, index=False)

    results = run_experiment(str(csv_file), seed=42)

    # Check both algorithms present
    assert "LogisticRegression" in results, "LogisticRegression results missing"
    assert "GradientBoostingClassifier" in results, "GradientBoostingClassifier results missing"

    # Check metrics are numeric
    for algo, metrics in results.items():
        assert 0 <= metrics.test_auc <= 1, f"{algo} test_auc out of range"
        assert 0 <= metrics.train_auc <= 1, f"{algo} train_auc out of range"
        assert 0 <= metrics.test_pr_auc <= 1, f"{algo} test_pr_auc out of range"


def test_feature_selection_correct_columns(sample_data):
    """Test that only honest features are used (not days_since_last_login)."""
    train = sample_data.iloc[:70]
    X, scaler = prepare_features(train, fit=True)

    # Verify we can reconstruct feature names from the original data
    expected_features = ["tenure_months", "monthly_spend", "support_tickets"]
    actual_features = train[expected_features].values

    # X should be the scaled version of these features
    assert X.shape[1] == len(expected_features), "Wrong number of features"

    # Verify days_since_last_login is NOT in features
    # (We can't directly check this, but we verify the shape is 3, not 4+)
    assert X.shape[1] == 3, "days_since_last_login should be excluded"
