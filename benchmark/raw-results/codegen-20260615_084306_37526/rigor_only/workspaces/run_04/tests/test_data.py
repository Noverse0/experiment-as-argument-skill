"""Tests for data loading, deduplication, and feature selection."""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.data import FEATURES, LEAKY_COLS, TARGET, load_and_prepare


def _write_mock_csv(path: str, n: int = 60, n_dups: int = 5, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "signup_date": "2023-06-01",
        "tenure_months": rng.integers(1, 72, n),
        "monthly_spend": rng.uniform(20.0, 200.0, n).round(2),
        "support_tickets": rng.integers(0, 6, n),
        "days_since_last_login": rng.integers(1, 90, n),
        "churned": rng.integers(0, 2, n),
    })
    dups = df.sample(n=n_dups, random_state=seed)
    pd.concat([df, dups], ignore_index=True).to_csv(path, index=False)


@pytest.fixture
def mock_csv(tmp_path):
    path = str(tmp_path / "test_churn.csv")
    _write_mock_csv(path, n=60, n_dups=5)
    return path


def test_deduplication_removes_planted_dups(mock_csv):
    _, _, _, stats = load_and_prepare(mock_csv)
    assert stats["n_dupes_dropped"] == 5
    assert stats["n_clean"] == 60


def test_leaky_col_absent_from_X(mock_csv):
    X, _, _, _ = load_and_prepare(mock_csv)
    for col in LEAKY_COLS:
        assert col not in X.columns, f"{col} must not appear in X"


def test_all_features_present_in_X(mock_csv):
    X, _, _, _ = load_and_prepare(mock_csv)
    for col in FEATURES:
        assert col in X.columns


def test_leaky_col_present_in_X_audit(mock_csv):
    _, _, X_audit, _ = load_and_prepare(mock_csv)
    for col in LEAKY_COLS:
        assert col in X_audit.columns


def test_X_y_lengths_match(mock_csv):
    X, y, _, stats = load_and_prepare(mock_csv)
    assert len(X) == len(y)
    assert len(X) == stats["n_clean"]


def test_no_missing_values(mock_csv):
    X, y, _, _ = load_and_prepare(mock_csv)
    assert X.isnull().sum().sum() == 0
    assert y.isnull().sum() == 0


def test_churn_rate_is_fraction(mock_csv):
    _, _, _, stats = load_and_prepare(mock_csv)
    assert 0.0 <= stats["churn_rate_clean"] <= 1.0


def test_stats_counts_consistent(mock_csv):
    _, _, _, stats = load_and_prepare(mock_csv)
    assert stats["n_raw"] == stats["n_clean"] + stats["n_dupes_dropped"]
