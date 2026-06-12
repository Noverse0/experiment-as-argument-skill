"""Tests for data loading, deduplication, and splitting."""
import numpy as np
import pandas as pd
import pytest

from src.data import (
    LEAKED_COLS,
    NUMERIC_FEATURES,
    build_features,
    deduplicate,
    prepare,
    time_split,
)


@pytest.fixture
def raw_df():
    """Minimal synthetic dataframe mimicking the churn CSV schema."""
    rng = np.random.default_rng(42)
    n = 100
    tenure = rng.integers(1, 72, n)
    spend = rng.uniform(10, 200, n).round(2)
    tickets = rng.integers(0, 5, n)
    dates = pd.date_range("2023-01-01", periods=n, freq="7D")
    churn = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "signup_date": dates,
            "tenure_months": tenure,
            "monthly_spend": spend,
            "support_tickets": tickets,
            "account_status": np.where(churn == 1, "closed", "active"),
            "churned": churn,
        }
    )
    return df


def test_deduplicate_removes_exact_duplicates(raw_df):
    df_with_dupes = pd.concat([raw_df, raw_df.iloc[:10]], ignore_index=True)
    cleaned, n_removed = deduplicate(df_with_dupes)
    assert n_removed == 10
    assert len(cleaned) == len(raw_df)


def test_deduplicate_no_change_when_clean(raw_df):
    cleaned, n_removed = deduplicate(raw_df)
    assert n_removed == 0
    assert len(cleaned) == len(raw_df)


def test_time_split_preserves_chronological_order(raw_df):
    train, test = time_split(raw_df, train_frac=0.80)
    assert train["signup_date"].max() <= test["signup_date"].min()


def test_time_split_sizes(raw_df):
    train, test = time_split(raw_df, train_frac=0.80)
    total = len(raw_df)
    assert len(train) == int(total * 0.80)
    assert len(test) == total - int(total * 0.80)


def test_time_split_no_overlap(raw_df):
    train, test = time_split(raw_df, train_frac=0.80)
    train_ids = set(train["customer_id"])
    test_ids = set(test["customer_id"])
    assert train_ids.isdisjoint(test_ids), "Train and test share customer_ids"


def test_leaked_cols_not_in_numeric_features():
    for col in LEAKED_COLS:
        assert col not in NUMERIC_FEATURES, f"{col} must not be in NUMERIC_FEATURES"


def test_account_status_not_in_features():
    assert "account_status" not in NUMERIC_FEATURES


def test_build_features_scaler_fit_on_train_only(raw_df):
    train, test = time_split(raw_df)
    X_train, X_test, y_train, y_test, scaler = build_features(train, test)
    # Scaler was fit on train: train mean should be near 0, std near 1
    assert np.abs(X_train.mean(axis=0)).max() < 1e-10
    assert np.abs(X_train.std(axis=0) - 1.0).max() < 1e-10
    # Test is only transformed — its mean may differ from 0
    # (but we just check no leakage: not re-fitted on test)
    assert X_test.shape[1] == len(NUMERIC_FEATURES)


def test_build_features_shapes(raw_df):
    train, test = time_split(raw_df)
    X_train, X_test, y_train, y_test, _ = build_features(train, test)
    assert X_train.shape[0] == len(train)
    assert X_test.shape[0] == len(test)
    assert X_train.shape[1] == len(NUMERIC_FEATURES)
    assert y_train.shape == (len(train),)
    assert y_test.shape == (len(test),)


def test_prepare_e2e(tmp_path):
    """Integration: prepare() runs end-to-end on the actual generated CSV."""
    import subprocess
    csv = tmp_path / "churn.csv"
    subprocess.run(["python3", "make_dataset.py", "--out", str(csv)], check=True)
    data = prepare(str(csv))
    assert data["n_dupes_removed"] == 200
    assert data["X_train"].shape[1] == len(NUMERIC_FEATURES)
    assert "account_status" not in data["feature_names"]
    assert "customer_id" not in data["feature_names"]
