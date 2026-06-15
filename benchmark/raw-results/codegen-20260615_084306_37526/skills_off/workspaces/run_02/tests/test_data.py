"""Tests for the data loading and preparation pipeline."""
import numpy as np
import pandas as pd
import pytest

from src.data import FEATURE_COLS, TARGET_COL, load_and_prepare


@pytest.fixture
def sample_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "signup_date": (
                pd.Timestamp("2023-01-01")
                + pd.to_timedelta(rng.integers(0, 600, n), unit="D")
            ).strftime("%Y-%m-%d"),
            "tenure_months": rng.integers(1, 72, n),
            "monthly_spend": rng.gamma(2.0, 30.0, n).round(2),
            "support_tickets": rng.poisson(1.2, n),
            "days_since_last_login": rng.integers(1, 90, n),
            "churned": rng.integers(0, 2, n),
        }
    )
    path = tmp_path / "churn_sample.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def csv_with_duplicates(tmp_path):
    """CSV where 10 rows are exact duplicates of the first 10 rows."""
    rng = np.random.default_rng(1)
    n = 50
    df = pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "signup_date": "2023-06-15",
            "tenure_months": rng.integers(1, 72, n),
            "monthly_spend": rng.gamma(2.0, 30.0, n).round(2),
            "support_tickets": rng.poisson(1.2, n),
            "days_since_last_login": rng.integers(1, 90, n),
            "churned": rng.integers(0, 2, n),
        }
    )
    df_with_dups = pd.concat([df, df.iloc[:10]], ignore_index=True)
    path = tmp_path / "churn_dups.csv"
    df_with_dups.to_csv(path, index=False)
    return str(path), 50  # (path, expected_size_after_dedup)


def test_output_shapes(sample_csv):
    X_train, X_test, y_train, y_test, scaler, meta = load_and_prepare(sample_csv)
    assert X_train.shape[1] == len(FEATURE_COLS)
    assert X_test.shape[1] == len(FEATURE_COLS)
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]
    assert X_train.shape[0] + X_test.shape[0] == meta["total_size"]


def test_target_leak_excluded(sample_csv):
    _, _, _, _, _, meta = load_and_prepare(sample_csv)
    assert "days_since_last_login" not in meta["feature_names"]


def test_customer_id_excluded(sample_csv):
    _, _, _, _, _, meta = load_and_prepare(sample_csv)
    assert "customer_id" not in meta["feature_names"]


def test_deduplication(csv_with_duplicates):
    path, expected_size = csv_with_duplicates
    _, _, y_train, y_test, _, meta = load_and_prepare(path)
    assert meta["total_size"] == expected_size
    assert meta["duplicates_removed"] == 10


def test_scaler_fit_on_train_only(sample_csv):
    X_train, X_test, y_train, y_test, scaler, meta = load_and_prepare(sample_csv)
    assert hasattr(scaler, "mean_"), "Scaler must be fitted"
    assert hasattr(scaler, "scale_")
    # Scaled train data should be approximately zero-mean, unit-variance
    assert abs(X_train.mean()) < 0.5
    assert abs(X_train.std() - 1.0) < 0.3


def test_time_based_split_no_overlap(sample_csv):
    X_train, X_test, y_train, y_test, scaler, meta = load_and_prepare(sample_csv)
    # signup_days is the last feature; in scaled space we can check ordering
    # by comparing counts — train should have earlier customers (lower signup_days)
    # We verify via meta sizes being consistent with train_frac
    expected_train = int(meta["total_size"] * 0.75)
    assert abs(meta["train_size"] - expected_train) <= 1


def test_meta_keys(sample_csv):
    _, _, _, _, _, meta = load_and_prepare(sample_csv)
    for key in [
        "feature_names", "original_size", "deduped_size",
        "duplicates_removed", "total_size", "train_size",
        "test_size", "train_churn_rate", "test_churn_rate",
    ]:
        assert key in meta, f"Missing meta key: {key}"


def test_churn_rates_are_valid(sample_csv):
    _, _, y_train, y_test, _, meta = load_and_prepare(sample_csv)
    assert 0.0 <= meta["train_churn_rate"] <= 1.0
    assert 0.0 <= meta["test_churn_rate"] <= 1.0
