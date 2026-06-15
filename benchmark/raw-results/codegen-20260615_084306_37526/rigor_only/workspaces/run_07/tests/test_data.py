"""Tests for data loading and splitting logic."""
import pandas as pd
import numpy as np
import pytest

from src.data import (
    FEATURES,
    LEAKY_FEATURES,
    TARGET,
    load_and_clean,
    temporal_split,
)


@pytest.fixture
def sample_df():
    """Minimal synthetic dataframe that mirrors the real schema."""
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({
        "customer_id": np.arange(n),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="7D").strftime("%Y-%m-%d"),
        "tenure_months": rng.integers(1, 72, n),
        "monthly_spend": rng.uniform(10, 200, n).round(2),
        "support_tickets": rng.integers(0, 5, n),
        "days_since_last_login": rng.integers(1, 90, n),
        "churned": rng.integers(0, 2, n),
    })
    # Append 5 exact duplicates.
    dup = df.sample(5, random_state=0)
    return pd.concat([df, dup], ignore_index=True)


def test_load_removes_duplicates(tmp_path, sample_df):
    path = str(tmp_path / "test.csv")
    sample_df.to_csv(path, index=False)
    clean_df, audit = load_and_clean(path)
    assert audit["n_duplicates_removed"] == 5
    assert audit["n_clean"] == 100
    assert len(clean_df) == 100


def test_load_reports_target_rate(tmp_path, sample_df):
    path = str(tmp_path / "test.csv")
    sample_df.to_csv(path, index=False)
    clean_df, audit = load_and_clean(path)
    expected = float(clean_df[TARGET].mean())
    assert abs(audit["target_rate"] - expected) < 1e-9


def test_temporal_split_no_time_overlap(sample_df):
    # Remove duplicates first as the real pipeline does.
    df = sample_df.drop_duplicates()
    df, _ = df.copy(), None  # just use the frame
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    train, test = temporal_split(df, test_frac=0.2)
    assert train["signup_date"].max() <= test["signup_date"].min(), (
        "Train set contains dates that are more recent than the earliest test date — time leak"
    )


def test_temporal_split_sizes(sample_df):
    df = sample_df.drop_duplicates()
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    train, test = temporal_split(df, test_frac=0.2)
    total = len(df)
    assert len(train) + len(test) == total
    assert abs(len(test) / total - 0.2) < 0.05


def test_features_exclude_leaky():
    for f in LEAKY_FEATURES:
        assert f not in FEATURES, f"{f} is a leaky feature and must not appear in FEATURES"


def test_features_exclude_id():
    assert "customer_id" not in FEATURES


def test_temporal_split_preserves_all_rows(sample_df):
    df = sample_df.drop_duplicates().copy()
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    train, test = temporal_split(df, test_frac=0.2)
    assert len(train) + len(test) == len(df)
