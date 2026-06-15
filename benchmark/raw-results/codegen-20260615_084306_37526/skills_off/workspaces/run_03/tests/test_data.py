import numpy as np
import pandas as pd
import pytest

from src.data import load_and_prepare, time_split, get_Xy, FEATURE_COLS, TARGET


@pytest.fixture
def sample_csv(tmp_path):
    np.random.seed(0)
    n = 100
    df = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="3D").strftime("%Y-%m-%d"),
        "tenure_months": np.random.randint(1, 72, n),
        "monthly_spend": np.random.gamma(2.0, 30.0, n).round(2),
        "support_tickets": np.random.poisson(1.2, n),
        "days_since_last_login": np.random.randint(1, 90, n),
        "churned": np.random.randint(0, 2, n),
    })
    dup = df.sample(n=10, random_state=0)
    df_with_dups = pd.concat([df, dup], ignore_index=True)
    csv = tmp_path / "test.csv"
    df_with_dups.to_csv(csv, index=False)
    return str(csv)


def test_deduplication_removes_exact_duplicates(sample_csv):
    df, n_dup = load_and_prepare(sample_csv)
    assert n_dup == 10
    assert len(df) == 100


def test_no_leak_column_in_features():
    assert "days_since_last_login" not in FEATURE_COLS


def test_customer_id_not_in_features():
    assert "customer_id" not in FEATURE_COLS


def test_signup_date_not_in_features():
    assert "signup_date" not in FEATURE_COLS


def test_temporal_split_train_before_test(sample_csv):
    df, _ = load_and_prepare(sample_csv)
    train, test = time_split(df, test_frac=0.2)
    assert train["signup_date"].max() <= test["signup_date"].min()


def test_time_split_covers_all_rows(sample_csv):
    df, _ = load_and_prepare(sample_csv)
    train, test = time_split(df, test_frac=0.2)
    assert len(train) + len(test) == len(df)


def test_time_split_fraction(sample_csv):
    df, _ = load_and_prepare(sample_csv)
    train, test = time_split(df, test_frac=0.2)
    actual_frac = len(test) / len(df)
    assert abs(actual_frac - 0.2) < 0.05


def test_get_Xy_feature_count(sample_csv):
    df, _ = load_and_prepare(sample_csv)
    X, y = get_Xy(df)
    assert X.shape[1] == len(FEATURE_COLS)


def test_get_Xy_target_is_binary(sample_csv):
    df, _ = load_and_prepare(sample_csv)
    X, y = get_Xy(df)
    assert set(np.unique(y)).issubset({0, 1})


def test_get_Xy_shapes_consistent(sample_csv):
    df, _ = load_and_prepare(sample_csv)
    X, y = get_Xy(df)
    assert X.shape[0] == y.shape[0] == len(df)
