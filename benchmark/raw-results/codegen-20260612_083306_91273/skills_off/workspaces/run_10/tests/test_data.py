"""Tests for data loading and preprocessing."""
import pytest
import pandas as pd
import numpy as np
from src.data import load_and_clean, stratified_split, preprocess


@pytest.fixture
def sample_df():
    """Create a small sample dataset for testing."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "signup_date": pd.date_range("2023-01-01", periods=n).strftime("%Y-%m-%d"),
        "tenure_months": np.random.randint(1, 72, n),
        "monthly_spend": np.random.gamma(2.0, 30.0, n).round(2),
        "support_tickets": np.random.poisson(1.2, n),
        "account_status": np.random.choice(["active", "closed"], n),
        "churned": np.random.randint(0, 2, n),
    })
    return df


def test_load_and_clean_drops_leaky_columns(tmp_path):
    """Verify that account_status and customer_id are dropped."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "signup_date": pd.date_range("2023-01-01", periods=n).strftime("%Y-%m-%d"),
        "tenure_months": np.random.randint(1, 72, n),
        "monthly_spend": np.random.gamma(2.0, 30.0, n).round(2),
        "support_tickets": np.random.poisson(1.2, n),
        "account_status": np.random.choice(["active", "closed"], n),
        "churned": np.random.randint(0, 2, n),
    })

    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)

    cleaned = load_and_clean(str(csv_path))

    assert "account_status" not in cleaned.columns
    assert "customer_id" not in cleaned.columns
    assert "days_since_signup" in cleaned.columns
    assert "churned" in cleaned.columns


def test_load_and_clean_removes_duplicates(tmp_path):
    """Verify that exact duplicates are removed."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "signup_date": pd.date_range("2023-01-01", periods=n).strftime("%Y-%m-%d"),
        "tenure_months": np.random.randint(1, 72, n),
        "monthly_spend": np.random.gamma(2.0, 30.0, n).round(2),
        "support_tickets": np.random.poisson(1.2, n),
        "account_status": np.random.choice(["active", "closed"], n),
        "churned": np.random.randint(0, 2, n),
    })

    # Add duplicates
    df_with_dups = pd.concat([df, df.iloc[:10]], ignore_index=True)
    assert len(df_with_dups) == len(df) + 10

    csv_path = tmp_path / "test.csv"
    df_with_dups.to_csv(csv_path, index=False)

    cleaned = load_and_clean(str(csv_path))

    assert len(cleaned) == len(df)


def test_stratified_split_preserves_churn_rate():
    """Verify stratified split maintains class balance in all splits."""
    np.random.seed(42)
    n = 100
    # Create imbalanced dataset: 30% churn
    df = pd.DataFrame({
        "tenure_months": np.random.randint(1, 72, n),
        "monthly_spend": np.random.gamma(2.0, 30.0, n).round(2),
        "support_tickets": np.random.poisson(1.2, n),
        "days_since_signup": np.random.randint(0, 900, n),
        "churned": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df, random_state=42)

    overall_rate = df["churned"].mean()
    train_rate = y_train.mean()
    val_rate = y_val.mean()
    test_rate = y_test.mean()

    # Rates should be similar (within tolerance due to randomness)
    assert abs(train_rate - overall_rate) < 0.1
    assert abs(val_rate - overall_rate) < 0.1
    assert abs(test_rate - overall_rate) < 0.1


def test_stratified_split_sizes():
    """Verify split sizes are approximately correct."""
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "tenure_months": np.random.randint(1, 72, n),
        "monthly_spend": np.random.gamma(2.0, 30.0, n).round(2),
        "support_tickets": np.random.poisson(1.2, n),
        "days_since_signup": np.random.randint(0, 900, n),
        "churned": np.random.randint(0, 2, n),
    })

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        df, train_size=0.6, val_size=0.2, random_state=42
    )

    total = len(df)
    assert len(X_train) == pytest.approx(total * 0.6, abs=5)
    assert len(X_val) == pytest.approx(total * 0.2, abs=5)
    assert len(X_test) == pytest.approx(total * 0.2, abs=5)


def test_preprocess_scales_numerical_features():
    """Verify that numerical features are standardized."""
    np.random.seed(42)
    X_train = pd.DataFrame({
        "tenure_months": np.random.randint(1, 72, 100),
        "monthly_spend": np.random.gamma(2.0, 30.0, 100),
    })
    X_val = X_train.copy()
    X_test = X_train.copy()

    X_train_scaled, X_val_scaled, X_test_scaled = preprocess(X_train, X_val, X_test)

    # Check that scaled features have mean ~0 and std ~1
    for col in ["tenure_months", "monthly_spend"]:
        assert abs(X_train_scaled[col].mean()) < 0.1
        assert abs(X_train_scaled[col].std() - 1.0) < 0.1


def test_preprocess_aligns_categorical_columns():
    """Verify categorical encoding produces same columns across splits."""
    np.random.seed(42)
    n = 100
    X_train = pd.DataFrame({
        "tenure_months": np.random.randint(1, 72, n),
        "category": np.random.choice(["A", "B", "C"], n),
    })
    X_val = X_train.copy()
    X_test = X_train.copy()

    X_train_scaled, X_val_scaled, X_test_scaled = preprocess(X_train, X_val, X_test)

    # All splits should have the same columns
    assert set(X_train_scaled.columns) == set(X_val_scaled.columns) == set(X_test_scaled.columns)
    # Should have categorical columns (one-hot encoded, minus one for drop_first)
    assert any("category" in col for col in X_train_scaled.columns)
