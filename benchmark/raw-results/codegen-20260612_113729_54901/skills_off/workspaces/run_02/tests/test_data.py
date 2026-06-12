"""Tests for data loading, cleaning, and feature extraction."""

import pandas as pd
import pytest

from src.data import TARGET, FEATURES, DATE_COL, clean, get_X_y, load


def _make_raw_df(n: int = 20, with_dups: bool = False) -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="30D"),
        "tenure_months": rng.integers(1, 72, n),
        "monthly_spend": rng.uniform(20, 200, n).round(2),
        "support_tickets": rng.integers(0, 5, n),
        "account_status": rng.choice(["active", "closed"], n),
        "churned": rng.integers(0, 2, n),
    })
    if with_dups:
        df = pd.concat([df, df.iloc[:3]], ignore_index=True)
    return df


def test_clean_drops_account_status():
    df, _ = clean(_make_raw_df())
    assert "account_status" not in df.columns


def test_clean_drops_customer_id():
    df, _ = clean(_make_raw_df())
    assert "customer_id" not in df.columns


def test_clean_removes_duplicates():
    raw = _make_raw_df(with_dups=True)
    df, stats = clean(raw)
    assert stats["n_duplicates_removed"] == 3
    assert len(df) == 20


def test_clean_sorts_by_signup_date():
    import numpy as np
    # Shuffle dates deliberately out of order
    raw = _make_raw_df(n=30)
    raw = raw.sample(frac=1, random_state=99).reset_index(drop=True)
    df, _ = clean(raw)
    dates = df[DATE_COL].values
    assert all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))


def test_clean_reports_churn_rate():
    df_raw = _make_raw_df()
    _, stats = clean(df_raw)
    assert 0.0 <= stats["churn_rate"] <= 1.0


def test_clean_no_duplicates_stat_when_none():
    raw = _make_raw_df(with_dups=False)
    _, stats = clean(raw)
    assert stats["n_duplicates_removed"] == 0


def test_get_X_y_returns_correct_features():
    df_raw = _make_raw_df()
    df, _ = clean(df_raw)
    X, y = get_X_y(df)
    assert list(X.columns) == FEATURES
    assert y.name == TARGET


def test_get_X_y_lengths_match():
    df_raw = _make_raw_df(n=15)
    df, _ = clean(df_raw)
    X, y = get_X_y(df)
    assert len(X) == len(y)
