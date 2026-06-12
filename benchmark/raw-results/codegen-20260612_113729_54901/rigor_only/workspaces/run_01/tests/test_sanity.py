"""Tests for the sanity check module."""
import numpy as np
import pandas as pd
import pytest

from src.pipeline import make_lr_pipeline, get_features_target, temporal_split
from src import sanity


@pytest.fixture
def trained_data():
    """Return fitted train/test arrays from a small synthetic dataset."""
    rng = np.random.default_rng(1)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="5D")
    tenure = rng.integers(1, 72, n)
    spend = rng.uniform(10, 200, n).round(2)
    tickets = rng.integers(0, 5, n)
    logit = -1.2 - 0.03 * tenure + 0.01 * spend + 0.45 * tickets
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "signup_date": dates,
        "tenure_months": tenure,
        "monthly_spend": spend,
        "support_tickets": tickets,
        "account_status": np.where(churn == 1, "closed", "active"),
        "churned": churn,
    })
    train_df, test_df = temporal_split(df, test_frac=0.2)
    X_train, y_train = get_features_target(train_df)
    X_test, y_test = get_features_target(test_df)
    return X_train, y_train, X_test, y_test


def test_check_no_leakage_columns_clean(trained_data):
    X_train, _, _, _ = trained_data
    # Should not raise.
    sanity.check_no_leakage_columns(X_train)


def test_check_no_leakage_columns_raises_on_leaky():
    df = pd.DataFrame({"account_status": ["a"], "feat": [1.0]})
    with pytest.raises(ValueError, match="Leaky columns"):
        sanity.check_no_leakage_columns(df)


def test_check_baseline_floor_near_half(trained_data):
    X_train, y_train, X_test, y_test = trained_data
    auc = sanity.check_baseline_floor(X_train, y_train, X_test, y_test)
    assert 0.4 <= auc <= 0.6, "Majority classifier AUC should be near 0.5"


def test_check_overfit_subset(trained_data):
    X_train, y_train, X_test, y_test = trained_data
    pipe = make_lr_pipeline(random_state=0)
    result = sanity.check_overfit_subset(pipe, X_train, y_train, subset_size=20)
    assert isinstance(result, bool)


def test_check_label_shuffle_near_half(trained_data):
    X_train, y_train, X_test, y_test = trained_data
    pipe = make_lr_pipeline(random_state=0)
    pipe.fit(X_train, y_train)
    auc = sanity.check_label_shuffle(pipe, X_train, y_train, X_test, y_test)
    assert 0.3 <= auc <= 0.75, "Shuffled-label AUC should be near random"


def test_run_all_returns_expected_keys(trained_data):
    X_train, y_train, X_test, y_test = trained_data
    pipe = make_lr_pipeline(random_state=0)
    result = sanity.run_all(pipe, X_train, y_train, X_test, y_test)
    for key in ("baseline_auc", "overfit_ok", "shuffle_auc", "warnings"):
        assert key in result
