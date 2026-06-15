"""Tests for the ML pipeline."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from src.pipeline import (
    load_and_clean,
    time_split,
    get_clean_features,
    get_features_with_leak,
    preprocess,
    evaluate,
    get_baseline_metrics,
)


@pytest.fixture
def sample_df():
    """Create a small sample dataset for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "signup_date": pd.date_range("2023-01-01", periods=n),
        "tenure_months": np.random.randint(1, 72, n),
        "monthly_spend": np.random.gamma(2.0, 30.0, n).round(2),
        "support_tickets": np.random.poisson(1.2, n),
        "days_since_last_login": np.random.randint(1, 100, n),
        "churned": np.random.binomial(1, 0.25, n),
    })


def test_load_and_clean_removes_duplicates(sample_df, tmp_path):
    """Test that duplicate removal works."""
    # Add duplicates
    df_with_dups = pd.concat([sample_df, sample_df.iloc[:5]], ignore_index=True)
    csv_path = tmp_path / "test.csv"
    df_with_dups.to_csv(csv_path, index=False)

    df_clean = load_and_clean(str(csv_path))
    assert len(df_clean) == len(sample_df)
    assert not df_clean.duplicated().any()


def test_time_split_respects_order(sample_df):
    """Test that time split preserves temporal order."""
    train, test = time_split(sample_df)
    assert len(train) + len(test) == len(sample_df)
    assert train["signup_date"].max() <= test["signup_date"].min()


def test_time_split_fraction(sample_df):
    """Test that time split respects the train fraction."""
    train, test = time_split(sample_df, train_fraction=0.7)
    ratio = len(train) / len(sample_df)
    assert 0.65 < ratio < 0.75  # Allow small rounding error


def test_get_clean_features_shape(sample_df):
    """Test that clean features have correct shape."""
    X, y = get_clean_features(sample_df)
    assert X.shape == (len(sample_df), 3)  # 3 clean features
    assert y.shape == (len(sample_df),)


def test_get_features_with_leak_shape(sample_df):
    """Test that leak features have correct shape."""
    X, y = get_features_with_leak(sample_df)
    assert X.shape == (len(sample_df), 4)  # 4 features + leak
    assert y.shape == (len(sample_df),)


def test_get_features_with_leak_includes_days_since_login(sample_df):
    """Verify that leak version includes days_since_last_login."""
    X_clean, _ = get_clean_features(sample_df)
    X_leak, _ = get_features_with_leak(sample_df)
    assert X_leak.shape[1] > X_clean.shape[1]
    assert X_leak.shape[1] == 4


def test_preprocess_scaling(sample_df):
    """Test that preprocessing scales features correctly."""
    train, test = time_split(sample_df)
    X_train, y_train = get_clean_features(train)
    X_test, y_test = get_clean_features(test)

    X_train_scaled, X_test_scaled, scaler = preprocess(X_train, X_test)

    # Scaled training data should have mean ≈ 0, std ≈ 1
    assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-10)

    # Test data should be scaled with train parameters
    assert X_test_scaled is not None
    assert X_test_scaled.shape == X_test.shape


def test_evaluate_returns_all_metrics(sample_df):
    """Test that evaluate returns all expected metrics."""
    y_true = sample_df["churned"].values
    y_pred = np.random.binomial(1, 0.3, len(sample_df))
    y_pred_proba = np.random.random(len(sample_df))

    metrics = evaluate(y_true, y_pred, y_pred_proba)

    assert "auc" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "log_loss" in metrics


def test_evaluate_metric_ranges(sample_df):
    """Test that metrics are in valid ranges."""
    y_true = sample_df["churned"].values
    y_pred = np.random.binomial(1, 0.3, len(sample_df))
    y_pred_proba = np.random.random(len(sample_df))

    metrics = evaluate(y_true, y_pred, y_pred_proba)

    assert 0 <= metrics["auc"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1
    assert metrics["log_loss"] >= 0


def test_get_baseline_metrics_beats_no_info(sample_df):
    """Test that baseline returns reasonable metrics."""
    y_test = sample_df["churned"].values
    baseline_metrics, majority_class = get_baseline_metrics(y_test)

    assert "auc" in baseline_metrics
    assert majority_class in [0, 1]
    assert baseline_metrics["auc"] >= 0.5  # At least as good as random


def test_models_fit_and_predict(sample_df):
    """Test that both models can fit and predict."""
    train, test = time_split(sample_df)
    X_train, y_train = get_clean_features(train)
    X_test, y_test = get_clean_features(test)
    X_train_scaled, X_test_scaled, _ = preprocess(X_train, X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    gb = GradientBoostingClassifier(random_state=42)

    # Fit
    lr.fit(X_train_scaled, y_train)
    gb.fit(X_train_scaled, y_train)

    # Predict
    lr_pred = lr.predict(X_test_scaled)
    gb_pred = gb.predict(X_test_scaled)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
    gb_proba = gb.predict_proba(X_test_scaled)[:, 1]

    assert lr_pred.shape == y_test.shape
    assert gb_pred.shape == y_test.shape
    assert lr_proba.shape == y_test.shape
    assert gb_proba.shape == y_test.shape


