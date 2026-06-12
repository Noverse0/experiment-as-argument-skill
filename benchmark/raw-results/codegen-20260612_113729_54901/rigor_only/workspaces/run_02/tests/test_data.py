import pandas as pd
import numpy as np
import pytest

from src.data import load_and_clean, get_X_y, FEATURE_COLS, TARGET


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 1],  # row 4 duplicates row 1
        "signup_date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-01-01"],
        "tenure_months": [12, 24, 6, 12],
        "monthly_spend": [50.0, 80.0, 30.0, 50.0],
        "support_tickets": [2, 1, 3, 2],
        "account_status": ["closed", "active", "closed", "closed"],
        "churned": [1, 0, 1, 1],
    })
    path = tmp_path / "churn_test.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_deduplication_removes_exact_dups(sample_csv):
    df = load_and_clean(sample_csv)
    assert len(df) == 3


def test_account_status_dropped(sample_csv):
    df = load_and_clean(sample_csv)
    assert "account_status" not in df.columns, "Leakage column must be removed"


def test_customer_id_dropped(sample_csv):
    df = load_and_clean(sample_csv)
    assert "customer_id" not in df.columns


def test_signup_date_dropped(sample_csv):
    df = load_and_clean(sample_csv)
    assert "signup_date" not in df.columns


def test_target_preserved(sample_csv):
    df = load_and_clean(sample_csv)
    assert TARGET in df.columns


def test_feature_cols_present(sample_csv):
    df = load_and_clean(sample_csv)
    for col in FEATURE_COLS:
        assert col in df.columns


def test_get_X_y_shapes(sample_csv):
    df = load_and_clean(sample_csv)
    X, y = get_X_y(df)
    assert X.shape == (3, len(FEATURE_COLS))
    assert y.shape == (3,)
    assert X.dtype == float


def test_no_leakage_col_in_X(sample_csv):
    df = load_and_clean(sample_csv)
    X, _ = get_X_y(df)
    # X should only have 3 numeric columns (no room for account_status encoding)
    assert X.shape[1] == 3
