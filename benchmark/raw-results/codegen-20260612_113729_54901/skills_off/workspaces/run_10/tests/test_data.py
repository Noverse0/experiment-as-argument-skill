import numpy as np
import pandas as pd
import pytest

from src.data import FEATURE_COLS, TARGET_COL, TEMPORAL_COL, load_and_clean, time_based_split


@pytest.fixture
def sample_df():
    np.random.seed(0)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="10D")
    return pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "signup_date": dates.strftime("%Y-%m-%d"),
        "tenure_months": np.random.randint(1, 72, n),
        "monthly_spend": np.random.gamma(2, 30, n).round(2),
        "support_tickets": np.random.poisson(1.2, n),
        "account_status": "active",
        TARGET_COL: np.random.randint(0, 2, n),
    })


def test_dedup_removes_exact_duplicates(tmp_path, sample_df):
    dup = sample_df.sample(10, random_state=0)
    df_with_dups = pd.concat([sample_df, dup], ignore_index=True)
    csv_path = tmp_path / "test.csv"
    df_with_dups.to_csv(csv_path, index=False)

    df, meta = load_and_clean(str(csv_path))

    assert meta["n_duplicates_dropped"] == 10
    assert meta["n_raw"] == 110
    assert meta["n_deduped"] == 100
    assert len(df) == 100


def test_clean_data_has_no_duplicates(tmp_path, sample_df):
    csv_path = tmp_path / "clean.csv"
    sample_df.to_csv(csv_path, index=False)
    df, meta = load_and_clean(str(csv_path))
    assert meta["n_duplicates_dropped"] == 0
    assert not df.duplicated().any()


def test_churn_rate_in_meta(tmp_path, sample_df):
    csv_path = tmp_path / "cr.csv"
    sample_df.to_csv(csv_path, index=False)
    _, meta = load_and_clean(str(csv_path))
    assert 0.0 <= meta["churn_rate"] <= 1.0


def test_time_split_sizes(sample_df):
    X_train, X_test, y_train, y_test, _ = time_based_split(sample_df, test_frac=0.2)
    assert len(X_train) + len(X_test) == len(sample_df)
    assert len(X_test) == pytest.approx(len(sample_df) * 0.2, abs=1)


def test_time_split_temporal_order(sample_df):
    """All training signup dates must precede all test signup dates."""
    df = sample_df.copy()
    df[TEMPORAL_COL] = pd.to_datetime(df[TEMPORAL_COL])
    df_sorted = df.sort_values(TEMPORAL_COL).reset_index(drop=True)
    X_train, X_test, _, _, _ = time_based_split(df, test_frac=0.2)

    split_idx = len(X_train)
    train_max_date = df_sorted[TEMPORAL_COL].iloc[split_idx - 1]
    test_min_date = df_sorted[TEMPORAL_COL].iloc[split_idx]
    assert train_max_date <= test_min_date


def test_no_leaky_or_id_features_in_split(sample_df):
    X_train, X_test, _, _, _ = time_based_split(sample_df)
    for col in ["account_status", "customer_id", "signup_date"]:
        assert col not in X_train.columns, f"Leaky/ID column '{col}' found in X_train"
        assert col not in X_test.columns, f"Leaky/ID column '{col}' found in X_test"


def test_feature_columns_complete(sample_df):
    X_train, X_test, _, _, _ = time_based_split(sample_df)
    for col in FEATURE_COLS:
        assert col in X_train.columns
        assert col in X_test.columns


def test_target_not_in_features(sample_df):
    X_train, X_test, _, _, _ = time_based_split(sample_df)
    assert TARGET_COL not in X_train.columns
    assert TARGET_COL not in X_test.columns
