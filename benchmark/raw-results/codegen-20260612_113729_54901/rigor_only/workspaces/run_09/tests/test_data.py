"""Tests for data loading, leakage prevention, and splitting."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from make_dataset import make
from src.data import load_and_clean, temporal_split


@pytest.fixture
def raw_csv(tmp_path):
    df = make(seed=42, n=500)
    path = tmp_path / "churn.csv"
    df.to_csv(path, index=False)
    return str(path), df


def test_drops_account_status(raw_csv):
    path, _ = raw_csv
    df, audit = load_and_clean(path)
    assert "account_status" not in df.columns, (
        "account_status must be dropped — it is a perfect target leak "
        "('closed' iff churned=1)"
    )


def test_drops_customer_id(raw_csv):
    path, _ = raw_csv
    df, audit = load_and_clean(path)
    assert "customer_id" not in df.columns


def test_removes_planted_duplicates(raw_csv):
    path, raw_df = raw_csv
    n_unique = len(raw_df.drop_duplicates())
    df, audit = load_and_clean(path)
    # After dropping customer_id and account_status, the dedup should still catch
    # rows identical across remaining columns
    assert audit["duplicates_removed"] > 0, "Expected duplicate rows to be found"
    assert len(df) == n_unique or len(df) <= n_unique


def test_signup_date_converted_to_numeric(raw_csv):
    path, _ = raw_csv
    df, _ = load_and_clean(path)
    assert "signup_date" not in df.columns
    assert "signup_days" in df.columns
    assert df["signup_days"].dtype in [np.int64, np.int32, int, np.float64]


def test_temporal_split_ordering(raw_csv):
    path, _ = raw_csv
    df, _ = load_and_clean(path)
    train, test = temporal_split(df, test_frac=0.2)
    assert train["signup_days"].max() <= test["signup_days"].min(), (
        "All training signup_days must precede all test signup_days"
    )


def test_temporal_split_sizes(raw_csv):
    path, _ = raw_csv
    df, _ = load_and_clean(path)
    train, test = temporal_split(df, test_frac=0.2)
    assert len(train) + len(test) == len(df)
    assert abs(len(test) / len(df) - 0.2) < 0.02


def test_no_index_overlap_after_split(raw_csv):
    path, _ = raw_csv
    df, _ = load_and_clean(path)
    train, test = temporal_split(df, test_frac=0.2)
    assert set(train.index).isdisjoint(set(test.index))


def test_target_column_present(raw_csv):
    path, _ = raw_csv
    df, _ = load_and_clean(path)
    assert "churned" in df.columns
    assert set(df["churned"].unique()).issubset({0, 1})


def test_account_status_matches_target_in_raw(raw_csv):
    """Confirm the leak exists in the raw data, so the test above is meaningful."""
    _, raw_df = raw_csv
    perfect_match = (raw_df["account_status"] == "closed") == (raw_df["churned"] == 1)
    assert perfect_match.all(), "account_status should be a perfect function of churned"
