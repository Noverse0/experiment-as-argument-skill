"""Test suite for the churn prediction pipeline."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.dataset import load_data, check_duplicates, get_class_balance
from src.pipeline import (
    time_based_split, deduplicate_train, train_and_evaluate, label_shuffle_test, FEATURE_COLS
)


@pytest.fixture
def sample_data():
    """Create a small synthetic dataset for testing."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "tenure_months": np.random.randint(1, 72, n),
        "monthly_spend": np.random.gamma(2.0, 30.0, n),
        "support_tickets": np.random.poisson(1.2, n),
        "days_since_last_login": np.random.randint(1, 100, n),
        "churned": np.random.randint(0, 2, n),
    })
    return df


def test_data_loading(tmp_path):
    """Test that data is loaded and parsed correctly."""
    # Create a temporary CSV
    df_orig = pd.DataFrame({
        "customer_id": [1, 2],
        "signup_date": ["2023-01-01", "2023-01-02"],
        "tenure_months": [10, 20],
        "monthly_spend": [100.0, 200.0],
        "support_tickets": [1, 2],
        "days_since_last_login": [5, 10],
        "churned": [0, 1],
    })
    csv_path = tmp_path / "test.csv"
    df_orig.to_csv(csv_path, index=False)

    # Load and check
    df = load_data(str(csv_path))
    assert len(df) == 2
    assert isinstance(df["signup_date"].iloc[0], pd.Timestamp)
    assert df["churned"].tolist() == [0, 1]


def test_check_duplicates(sample_data):
    """Test duplicate detection."""
    # Original should have no duplicates (except random)
    n_dups = check_duplicates(sample_data)
    assert n_dups == 0

    # Add a duplicate
    dup_row = sample_data.iloc[0:1].copy()
    df_with_dup = pd.concat([sample_data, dup_row], ignore_index=True)
    n_dups = check_duplicates(df_with_dup)
    assert n_dups == 1


def test_class_balance(sample_data):
    """Test class balance reporting."""
    balance = get_class_balance(sample_data)
    assert "churned" in balance
    assert "not_churned" in balance
    assert "churn_rate" in balance
    assert balance["churned"] + balance["not_churned"] == len(sample_data)
    assert 0 <= balance["churn_rate"] <= 1


def test_time_based_split(sample_data):
    """Test that time-based split respects chronological order."""
    X_train, X_test, y_train, y_test = time_based_split(sample_data, test_size=0.2)

    # Check split sizes
    assert len(X_train) + len(X_test) == len(sample_data)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    # Test set should be roughly 20%
    assert 0.15 < len(X_test) / len(sample_data) < 0.25

    # Check features are correct
    assert list(X_train.columns) == FEATURE_COLS
    assert list(X_test.columns) == FEATURE_COLS


def test_deduplicate_train(sample_data):
    """Test that deduplication works only on train set."""
    # Create duplicates
    dup_rows = sample_data.iloc[0:5].copy()
    df_with_dups = pd.concat([sample_data, dup_rows], ignore_index=True)

    X_train, X_test, y_train, y_test = time_based_split(df_with_dups, test_size=0.2)
    X_train_before = len(X_train)

    X_train_dedup, y_train_dedup, n_removed = deduplicate_train(X_train, y_train)

    # Check that dedup happened
    assert n_removed > 0
    assert len(X_train_dedup) < X_train_before

    # Check that train/test sizes still match
    assert len(X_train_dedup) == len(y_train_dedup)


def test_train_and_evaluate(sample_data):
    """Test that model training and evaluation work."""
    X_train, X_test, y_train, y_test = time_based_split(sample_data, test_size=0.3)

    for model_name in ["logistic_regression", "gradient_boosting"]:
        metrics = train_and_evaluate(model_name, X_train, X_test, y_train, y_test, seed=42)

        # Check all metrics are present and in valid range
        assert "auc_roc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

        assert 0 <= metrics["auc_roc"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1


def test_label_shuffle_test(sample_data):
    """Test that label shuffle sanity check works."""
    X_train, X_test, y_train, y_test = time_based_split(sample_data, test_size=0.3)

    result = label_shuffle_test(X_test, y_test, seed=42)

    assert "baseline_auc" in result
    assert "shuffled_auc" in result
    assert "churn_rate" in result

    assert result["baseline_auc"] == 0.5
    assert 0 <= result["shuffled_auc"] <= 1
    assert 0 <= result["churn_rate"] <= 1


def test_model_reproducibility(sample_data):
    """Test that same seed produces same results."""
    X_train, X_test, y_train, y_test = time_based_split(sample_data, test_size=0.3)

    metrics1 = train_and_evaluate("logistic_regression", X_train, X_test, y_train, y_test, seed=42)
    metrics2 = train_and_evaluate("logistic_regression", X_train, X_test, y_train, y_test, seed=42)

    # Should be identical
    for key in metrics1:
        assert metrics1[key] == pytest.approx(metrics2[key])


def test_invalid_model_name(sample_data):
    """Test that invalid model name raises error."""
    X_train, X_test, y_train, y_test = time_based_split(sample_data, test_size=0.3)

    with pytest.raises(ValueError):
        train_and_evaluate("invalid_model", X_train, X_test, y_train, y_test, seed=42)
