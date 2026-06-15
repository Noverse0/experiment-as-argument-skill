"""Shared pytest fixtures."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def churn_df():
    """Small synthetic DataFrame that mirrors the churn.csv schema."""
    rng = np.random.default_rng(0)
    n = 120
    return pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="3D").strftime("%Y-%m-%d"),
        "tenure_months": rng.integers(1, 72, n),
        "monthly_spend": rng.gamma(2.0, 30.0, n).round(2),
        "support_tickets": rng.poisson(1.2, n),
        "days_since_last_login": rng.integers(1, 90, n),
        "churned": rng.integers(0, 2, n),
    })


@pytest.fixture
def separable_data():
    """Simple two-class dataset where class 1 has higher feature values.

    Both classes are shuffled throughout so any slice contains both labels.
    """
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({
        "f1": np.concatenate([rng.normal(0, 1, n // 2), rng.normal(4, 1, n // 2)]),
        "f2": rng.normal(0, 1, n),
    })
    y = pd.Series(np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int))
    idx = rng.permutation(n)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
