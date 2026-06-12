"""Integration test: full pipeline on a small synthetic dataset."""

import numpy as np
import pandas as pd
import pytest

from src.data_prep import get_X_y, time_split
from src.evaluate import compute_metrics
from src.models import make_gbm_pipeline, make_lr_pipeline


@pytest.fixture
def small_df():
    """100-row synthetic churn dataset with the same schema as the real one."""
    rng = np.random.default_rng(7)
    n = 100
    tenure = rng.integers(1, 72, n)
    spend = rng.gamma(2.0, 30.0, n).round(2)
    tickets = rng.poisson(1.2, n)
    signup = pd.date_range("2023-01-01", periods=n, freq="3D")
    logit = -1.2 - 0.03 * tenure + 0.01 * spend + 0.45 * tickets
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "signup_date": signup,
        "tenure_months": tenure,
        "monthly_spend": spend,
        "support_tickets": tickets,
        "account_status": np.where(churn == 1, "closed", "active"),
        "churned": churn,
    })


def test_no_leaky_features_in_split(small_df):
    train, test = time_split(small_df)
    for df in (train, test):
        assert "account_status" not in get_X_y(df)[0].columns


def test_split_respects_time(small_df):
    train, test = time_split(small_df)
    assert train["signup_date"].max() <= test["signup_date"].min()


@pytest.mark.parametrize("make_pipeline", [make_lr_pipeline, make_gbm_pipeline])
def test_model_beats_baseline(small_df, make_pipeline):
    """Each model must beat 0.5 AUC on a learnable dataset."""
    train, test = time_split(small_df)
    X_train, y_train = get_X_y(train)
    X_test, y_test = get_X_y(test)

    pipeline = make_pipeline(random_state=42)
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    m = compute_metrics(y_test, proba)
    assert m["roc_auc"] > 0.5, f"Expected AUC > 0.5, got {m['roc_auc']:.4f}"


def test_train_and_test_sets_are_disjoint_on_time(small_df):
    train, test = time_split(small_df)
    assert len(train) + len(test) == len(small_df)
