"""Tests for data loading and cleaning."""
import subprocess
import sys

import numpy as np
import pytest

from src.data import (
    FEATURE_COLS,
    TARGET_COL,
    load_and_clean,
    prepare_arrays,
    train_test_split_temporal,
)


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory):
    out = tmp_path_factory.mktemp("data") / "churn.csv"
    subprocess.run(
        [sys.executable, "make_dataset.py", "--out", str(out)],
        check=True,
    )
    return str(out)


@pytest.fixture(scope="module")
def cleaned_df(csv_path):
    df, _ = load_and_clean(csv_path)
    return df


def test_account_status_removed(csv_path):
    """account_status is a target-derived leak column and must be dropped."""
    df, _ = load_and_clean(csv_path)
    assert "account_status" not in df.columns


def test_customer_id_removed(csv_path):
    df, _ = load_and_clean(csv_path)
    assert "customer_id" not in df.columns


def test_signup_date_removed(csv_path):
    df, _ = load_and_clean(csv_path)
    assert "signup_date" not in df.columns


def test_signup_days_engineered(cleaned_df):
    assert "signup_days" in cleaned_df.columns
    assert cleaned_df["signup_days"].min() == 0


def test_duplicates_removed(csv_path):
    """The generator injects 200 exact duplicates; all must be removed."""
    _, n_removed = load_and_clean(csv_path)
    assert n_removed >= 200, f"Expected >=200 duplicates removed, got {n_removed}"


def test_no_remaining_duplicates(cleaned_df):
    assert cleaned_df.duplicated().sum() == 0


def test_feature_and_target_cols_present(cleaned_df):
    for col in FEATURE_COLS + [TARGET_COL]:
        assert col in cleaned_df.columns, f"Missing column: {col}"


def test_target_is_binary(cleaned_df):
    assert set(cleaned_df[TARGET_COL].unique()).issubset({0, 1})


def test_prepare_arrays_sorted_by_signup_days(cleaned_df):
    X, y, _ = prepare_arrays(cleaned_df)
    signup_idx = FEATURE_COLS.index("signup_days")
    days = X[:, signup_idx]
    assert (np.diff(days) >= 0).all(), "X must be sorted by signup_days"


def test_prepare_arrays_shape(cleaned_df):
    X, y, meta = prepare_arrays(cleaned_df)
    assert X.shape[1] == len(FEATURE_COLS)
    assert X.shape[0] == y.shape[0]
    assert meta["n_total"] == len(y)


def test_train_test_split_temporal_ordering(cleaned_df):
    X, y, _ = prepare_arrays(cleaned_df)
    X_tr, X_te, y_tr, y_te = train_test_split_temporal(X, y, test_frac=0.20)
    signup_idx = FEATURE_COLS.index("signup_days")
    # Train set max signup_days must be <= test set min signup_days (temporal ordering)
    assert X_tr[:, signup_idx].max() <= X_te[:, signup_idx].min()


def test_train_test_split_sizes(cleaned_df):
    X, y, meta = prepare_arrays(cleaned_df)
    X_tr, X_te, y_tr, y_te = train_test_split_temporal(X, y, test_frac=0.20)
    assert len(X_tr) + len(X_te) == meta["n_total"]
    assert len(X_tr) > len(X_te)
