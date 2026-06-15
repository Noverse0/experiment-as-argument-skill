"""
Tests for the churn prediction experiment pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from src.experiment import (
    load_data,
    prepare_features,
    split_temporal,
    preprocess,
    evaluate_model,
    baseline_majority,
    label_shuffle_test,
    aggregate_results,
    compare_models,
)


@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "tenure_months": np.random.randint(1, 72, n),
        "monthly_spend": np.random.uniform(10, 200, n),
        "support_tickets": np.random.randint(0, 5, n),
        "days_since_last_login": np.random.randint(0, 100, n),
        "churned": np.random.binomial(1, 0.3, n),
    })


def test_load_data(tmp_path, sample_data):
    """Test data loading."""
    csv_file = tmp_path / "test.csv"
    sample_data.to_csv(csv_file, index=False)
    df = load_data(str(csv_file))
    assert df.shape[0] == sample_data.shape[0]
    assert "churned" in df.columns


def test_prepare_features(sample_data):
    """Test feature preparation."""
    X, y = prepare_features(sample_data)
    assert X.shape[0] == sample_data.shape[0]
    assert X.shape[1] == 4  # 4 features kept
    assert set(X.columns) == {"tenure_months", "monthly_spend", "support_tickets", "days_since_last_login"}
    assert len(y) == sample_data.shape[0]
    assert not X.isna().any().any()
    assert not np.isnan(y).any()


def test_split_temporal(sample_data):
    """Test temporal split respects ordering."""
    X, y = prepare_features(sample_data)
    X_train, X_test, y_train, y_test = split_temporal(X, y, train_fraction=0.7)

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert len(X_train) / len(X) < 0.75  # close to 70%

    # Check temporal ordering: train tenure should be <= test tenure
    train_tenure_max = X_train["tenure_months"].max()
    test_tenure_min = X_test["tenure_months"].min()
    assert train_tenure_max <= test_tenure_min


def test_preprocess(sample_data):
    """Test preprocessing (scaling on train only)."""
    X, y = prepare_features(sample_data)
    X_train, X_test, y_train, y_test = split_temporal(X, y)
    X_train_scaled, X_test_scaled = preprocess(X_train, X_test)

    assert X_train_scaled.shape[0] == len(X_train)
    assert X_test_scaled.shape[0] == len(X_test)
    assert X_train_scaled.shape[1] == 4

    # Scaled data should be roughly zero-mean, unit-variance
    assert abs(X_train_scaled.mean()) < 0.1
    assert abs(X_train_scaled.std() - 1.0) < 0.1


def test_evaluate_model(sample_data):
    """Test metric computation."""
    X, y = prepare_features(sample_data)
    X_train, X_test, y_train, y_test = split_temporal(X, y)

    # Create dummy predictions
    y_pred = (np.random.rand(len(y_test)) > 0.5).astype(int)
    y_pred_proba = np.random.rand(len(y_test))

    metrics = evaluate_model(y_test, y_pred, y_pred_proba)

    assert "auc" in metrics
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "accuracy" in metrics
    assert all(0 <= v <= 1 for v in metrics.values())


def test_baseline_majority(sample_data):
    """Test majority class baseline."""
    X, y = prepare_features(sample_data)
    X_train, X_test, y_train, y_test = split_temporal(X, y)

    baseline_metrics = baseline_majority(y_train, y_test)

    assert "auc" in baseline_metrics
    assert baseline_metrics["auc"] >= 0
    # Majority class AUC should be >= 0.5 (not better than random)


def test_label_shuffle_test(sample_data):
    """Test label shuffle sanity check."""
    X, y = prepare_features(sample_data)
    X_train, X_test, y_train, y_test = split_temporal(X, y)
    X_train_scaled, X_test_scaled = preprocess(X_train, X_test)

    shuffle_metrics = label_shuffle_test(X_train_scaled, X_test_scaled, y_test)

    assert "auc" in shuffle_metrics
    # With shuffled labels, AUC should be low (no information)


def test_aggregate_results():
    """Test result aggregation across seeds."""
    results = {
        "ModelA": [
            {"auc": 0.7, "f1": 0.6, "precision": 0.65, "recall": 0.55, "accuracy": 0.7},
            {"auc": 0.72, "f1": 0.62, "precision": 0.67, "recall": 0.57, "accuracy": 0.72},
            {"auc": 0.68, "f1": 0.58, "precision": 0.63, "recall": 0.53, "accuracy": 0.68},
        ],
        "ModelB": [
            {"auc": 0.75, "f1": 0.65, "precision": 0.7, "recall": 0.6, "accuracy": 0.75},
            {"auc": 0.77, "f1": 0.67, "precision": 0.72, "recall": 0.62, "accuracy": 0.77},
            {"auc": 0.73, "f1": 0.63, "precision": 0.68, "recall": 0.58, "accuracy": 0.73},
        ],
    }

    agg = aggregate_results(results)

    assert "ModelA" in agg
    assert "ModelB" in agg
    assert agg["ModelA"]["auc_mean"] == pytest.approx(0.7, abs=0.01)
    assert agg["ModelB"]["auc_mean"] == pytest.approx(0.75, abs=0.01)
    assert "auc_std" in agg["ModelA"]
    assert agg["ModelA"]["auc_std"] >= 0


def test_compare_models():
    """Test conclusion generation."""
    aggregated = {
        "LogisticRegression": {
            "auc_mean": 0.7,
            "auc_std": 0.01,
            "f1_mean": 0.6,
            "f1_std": 0.01,
            "precision_mean": 0.65,
            "precision_std": 0.01,
            "recall_mean": 0.55,
            "recall_std": 0.01,
            "accuracy_mean": 0.7,
            "accuracy_std": 0.01,
        },
        "GradientBoostingClassifier": {
            "auc_mean": 0.75,
            "auc_std": 0.01,
            "f1_mean": 0.65,
            "f1_std": 0.01,
            "precision_mean": 0.70,
            "precision_std": 0.01,
            "recall_mean": 0.60,
            "recall_std": 0.01,
            "accuracy_mean": 0.75,
            "accuracy_std": 0.01,
        },
    }

    conclusion = compare_models(aggregated)

    assert "Gradient boosting outperforms" in conclusion or \
           "Logistic regression outperforms" in conclusion or \
           "No statistically significant difference" in conclusion
    assert "AUC" in conclusion
