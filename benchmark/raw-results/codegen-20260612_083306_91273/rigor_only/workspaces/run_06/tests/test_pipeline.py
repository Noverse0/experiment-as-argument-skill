"""Tests for data pipeline: leak detection, split correctness."""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.pipeline import (
    load_and_clean, select_features, preprocess, prepare_split,
    time_based_split, check_target_balance
)


@pytest.fixture
def sample_df():
    """Create a small test dataset."""
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "signup_date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"],
        "tenure_months": [10, 20, 30, 40, 50],
        "monthly_spend": [100, 150, 200, 250, 300],
        "support_tickets": [1, 2, 3, 4, 5],
        "account_status": ["active", "closed", "active", "closed", "active"],
        "churned": [0, 1, 0, 1, 0],
    })


def test_load_and_clean_removes_duplicates(tmp_path):
    """Deduplication removes exact duplicates."""
    df = pd.DataFrame({
        "customer_id": [1, 2, 1],
        "churned": [0, 1, 0],
    })
    csv_file = tmp_path / "test.csv"
    df.to_csv(csv_file, index=False)

    cleaned = load_and_clean(str(csv_file))
    assert len(cleaned) == 2


def test_select_features_excludes_leaked_and_index_cols(sample_df):
    """Feature selection excludes account_status (leaked), customer_id (index), signup_date (temporal)."""
    features = select_features(sample_df)
    assert list(features.columns) == ["tenure_months", "monthly_spend", "support_tickets"]
    assert "account_status" not in features.columns
    assert "customer_id" not in features.columns
    assert "signup_date" not in features.columns


def test_preprocess_scales_correctly(sample_df):
    """Scaler fitted on train, applied to both."""
    X_train = select_features(sample_df.iloc[:3])
    X_test = select_features(sample_df.iloc[3:])

    X_train_scaled, X_test_scaled, scaler = preprocess(X_train, X_test)

    # Check shapes
    assert X_train_scaled.shape == (3, 3)
    assert X_test_scaled.shape == (2, 3)

    # Check train is centered/scaled (approximately)
    assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-10)


def test_time_based_split_respects_temporal_order(sample_df):
    """Time-based split orders by signup_date: older→train, newer→test."""
    df_train, df_test = time_based_split(sample_df, test_size=0.4)

    # Check that all train dates are <= all test dates
    train_max_date = df_train["signup_date"].max()
    test_min_date = df_test["signup_date"].min()
    assert train_max_date <= test_min_date

    # Check split ratio
    assert len(df_train) == 3
    assert len(df_test) == 2


def test_no_data_leakage_in_split():
    """Prepare split does not leak train data into test via preprocessing."""
    df = pd.DataFrame({
        "customer_id": range(100),
        "signup_date": pd.date_range("2023-01-01", periods=100),
        "tenure_months": np.random.randint(1, 60, 100),
        "monthly_spend": np.random.gamma(2, 30, 100),
        "support_tickets": np.random.poisson(1.5, 100),
        "account_status": ["active"] * 100,
        "churned": np.random.randint(0, 2, 100),
    })
    csv_file = "/tmp/test_leak.csv"
    df.to_csv(csv_file, index=False)

    split = prepare_split(csv_file, test_size=0.2)

    # Check that scaler was fit on train only
    # Re-fit scaler on train and compare
    scaler_independent = StandardScaler()
    X_train_independent = scaler_independent.fit_transform(split["X_train"])

    assert np.allclose(X_train_independent, split["X_train"], atol=1e-10)


def test_train_test_no_temporal_overlap():
    """Train/test split via time has no temporal overlap."""
    df = pd.DataFrame({
        "customer_id": range(200),
        "signup_date": pd.date_range("2023-01-01", periods=200),
        "tenure_months": np.random.randint(1, 60, 200),
        "monthly_spend": np.random.gamma(2, 30, 200),
        "support_tickets": np.random.poisson(1.5, 200),
        "account_status": ["active"] * 200,
        "churned": np.random.randint(0, 2, 200),
    })
    csv_file = "/tmp/test_temporal.csv"
    df.to_csv(csv_file, index=False)

    df_train, df_test = time_based_split(df, test_size=0.3)

    # No row should be in both
    train_ids = set(df_train["customer_id"].values)
    test_ids = set(df_test["customer_id"].values)
    assert len(train_ids & test_ids) == 0

    # No date overlap
    train_dates = set(df_train["signup_date"].values)
    test_dates = set(df_test["signup_date"].values)
    assert len(train_dates & test_dates) == 0


def test_target_balance_computed(sample_df):
    """Check target balance calculation."""
    y = sample_df["churned"].values
    rate = check_target_balance(y)
    assert rate == 0.4


def test_full_pipeline_shape():
    """Full pipeline produces correct shapes."""
    df = pd.DataFrame({
        "customer_id": range(50),
        "signup_date": pd.date_range("2023-01-01", periods=50),
        "tenure_months": np.random.randint(1, 60, 50),
        "monthly_spend": np.random.gamma(2, 30, 50),
        "support_tickets": np.random.poisson(1.5, 50),
        "account_status": ["active"] * 50,
        "churned": np.random.randint(0, 2, 50),
    })
    csv_file = "/tmp/test_full.csv"
    df.to_csv(csv_file, index=False)

    split = prepare_split(csv_file, test_size=0.2)

    assert split["X_train"].shape[0] == 40
    assert split["X_test"].shape[0] == 10
    assert split["X_train"].shape[1] == 3
    assert split["X_test"].shape[1] == 3
    assert len(split["y_train"]) == 40
    assert len(split["y_test"]) == 10
