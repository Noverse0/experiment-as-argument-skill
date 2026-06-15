"""Tests for data loading, feature engineering, and model construction."""
import numpy as np
import pandas as pd
import pytest

from src.pipeline import (
    ID_COLS,
    LEAKY_COLS,
    TARGET,
    build_features,
    load_and_clean,
    make_models,
    temporal_split,
)


def test_leaky_cols_absent_from_features(churn_df):
    X, y, _ = build_features(churn_df)
    for col in LEAKY_COLS:
        assert col not in X.columns, f"Leaky column '{col}' must not appear in X"


def test_id_cols_absent_from_features(churn_df):
    X, _, _ = build_features(churn_df)
    for col in ID_COLS:
        assert col not in X.columns


def test_target_absent_from_features(churn_df):
    X, _, _ = build_features(churn_df)
    assert TARGET not in X.columns


def test_build_features_signup_days_present(churn_df):
    X, _, _ = build_features(churn_df)
    assert "signup_days" in X.columns
    assert X["signup_days"].min() == 0  # ordinal from earliest signup


def test_deduplication_removes_exact_dupes(tmp_path):
    n = 40
    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="3D").strftime("%Y-%m-%d"),
        "tenure_months": np.arange(1, n + 1),
        "monthly_spend": np.arange(1.0, n + 1.0),
        "support_tickets": np.zeros(n, dtype=int),
        "days_since_last_login": np.zeros(n, dtype=int),
        "churned": np.zeros(n, dtype=int),
    })
    duped = pd.concat([df, df.iloc[:10]], ignore_index=True)
    path = tmp_path / "test.csv"
    duped.to_csv(path, index=False)

    cleaned, n_removed = load_and_clean(str(path))
    assert n_removed == 10
    assert len(cleaned) == n


def test_deduplication_no_dupes_unchanged(tmp_path, churn_df):
    path = tmp_path / "test.csv"
    churn_df.to_csv(path, index=False)
    cleaned, n_removed = load_and_clean(str(path))
    assert n_removed == 0
    assert len(cleaned) == len(churn_df)


def test_temporal_split_sizes(churn_df):
    X, y, dates = build_features(churn_df)
    X_train, X_test, y_train, y_test = temporal_split(X, y, dates, train_frac=0.80)
    n = len(X)
    assert len(X_train) == int(n * 0.80)
    assert len(X_test) == n - int(n * 0.80)
    assert len(X_train) + len(X_test) == n


def test_temporal_split_chronological_order():
    """All train signup_dates should come before all test signup_dates."""
    n = 50
    dates = pd.Series(pd.date_range("2023-01-01", periods=n, freq="D"))
    X = pd.DataFrame({"rank": np.arange(n)})  # 'rank' encodes temporal position
    y = pd.Series(np.zeros(n, dtype=int))

    X_train, X_test, _, _ = temporal_split(X, y, dates, train_frac=0.80)
    n_train = int(n * 0.80)

    # The first n_train ranks (0..n_train-1) must be in train
    assert set(X_train["rank"].values) == set(range(n_train))
    assert set(X_test["rank"].values) == set(range(n_train, n))


def test_make_models_returns_expected_keys():
    models = make_models()
    assert "LogisticRegression" in models
    assert "GradientBoosting" in models


def test_pipeline_probabilities_valid(churn_df):
    from sklearn.model_selection import train_test_split

    X, y, _ = build_features(churn_df)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, random_state=0)
    for name, pipe in make_models().items():
        pipe.fit(X_tr, y_tr)
        probs = pipe.predict_proba(X_val)
        assert probs.shape == (len(X_val), 2), f"{name}: wrong shape"
        assert np.all(probs >= 0) and np.all(probs <= 1), f"{name}: probs out of [0,1]"
        assert np.allclose(probs.sum(axis=1), 1.0), f"{name}: rows don't sum to 1"
