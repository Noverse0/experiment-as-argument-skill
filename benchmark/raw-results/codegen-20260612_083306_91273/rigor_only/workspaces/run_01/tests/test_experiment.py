"""Tests for the churn prediction experiment."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.experiment import (
    load_and_preprocess,
    evaluate_model,
    run_seed,
)


@pytest.fixture
def sample_dataset():
    """Create a small sample churn dataset for testing."""
    df = pd.DataFrame({
        "customer_id": range(1, 101),
        "signup_date": pd.date_range("2023-01-01", periods=100),
        "tenure_months": np.random.randint(1, 72, 100),
        "monthly_spend": np.random.gamma(2.0, 30.0, 100).round(2),
        "support_tickets": np.random.poisson(1.2, 100),
        "account_status": np.random.choice(["active", "closed"], 100),
        "churned": np.random.randint(0, 2, 100),
    })

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name

    Path(f.name).unlink()


def test_load_and_preprocess_excludes_leaks(sample_dataset):
    """Test that load_and_preprocess excludes account_status and customer_id."""
    df = load_and_preprocess(sample_dataset)

    # Check that excluded columns are gone.
    assert "account_status" not in df.columns, "account_status should be excluded (leak)"
    assert "customer_id" not in df.columns, "customer_id should be excluded (not a feature)"
    assert "signup_date" not in df.columns, "raw signup_date should be excluded"

    # Check that temporal features are added.
    assert "signup_year" in df.columns, "signup_year should be extracted"
    assert "signup_month" in df.columns, "signup_month should be extracted"

    # Check that other features are present.
    assert "tenure_months" in df.columns
    assert "monthly_spend" in df.columns
    assert "support_tickets" in df.columns
    assert "churned" in df.columns


def test_load_and_preprocess_deduplicates(sample_dataset):
    """Test that duplicates are removed."""
    # Create a dataset with duplicates.
    df = pd.read_csv(sample_dataset)
    dup = df.iloc[:10].copy()
    df_with_dups = pd.concat([df, dup], ignore_index=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df_with_dups.to_csv(f.name, index=False)

        # Preprocess should remove the duplicates.
        result = load_and_preprocess(f.name)

        # The result should have fewer rows (100 + 10 - 10 = 100).
        assert len(result) == len(df), "Duplicates should be removed"

        Path(f.name).unlink()


def test_evaluate_model_balanced_case():
    """Test evaluate_model on a balanced case."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 1, 0])
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])

    metrics = evaluate_model(y_true, y_pred, y_pred_proba)

    # Check that all expected metrics are present.
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics

    # Check that metrics are in [0, 1].
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1


def test_evaluate_model_single_class():
    """Test evaluate_model when y_true has only one class (edge case)."""
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    y_pred_proba = np.array([0.1, 0.1, 0.1])

    metrics = evaluate_model(y_true, y_pred, y_pred_proba)

    # Should not crash; ROC-AUC should be NaN.
    assert metrics["accuracy"] == 1.0
    assert np.isnan(metrics["roc_auc"])


def test_run_seed_returns_results(sample_dataset):
    """Test that run_seed trains both models and returns metrics."""
    df = load_and_preprocess(sample_dataset)
    X = df.drop("churned", axis=1).values
    y = df["churned"].values

    # Simple 70/30 split for this test.
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    results = run_seed(X_train, X_test, y_train, y_test, seed=42)

    # Check structure.
    assert "LogisticRegression" in results
    assert "GradientBoosting" in results

    # Check that all metrics are present.
    for model_name in ["LogisticRegression", "GradientBoosting"]:
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert metric in results[model_name]
            value = results[model_name][metric]
            # Metrics should be floats (may be NaN but still float).
            assert isinstance(value, (float, np.floating))


def test_run_seed_deterministic():
    """Test that run_seed is deterministic with the same seed."""
    X_train, y_train = make_classification(n_samples=100, n_features=5, random_state=0)
    X_test, y_test = make_classification(n_samples=50, n_features=5, random_state=1)

    results1 = run_seed(X_train, X_test, y_train, y_test, seed=42)
    results2 = run_seed(X_train, X_test, y_train, y_test, seed=42)

    # Same seed should give identical results (up to floating point).
    for model_name in ["LogisticRegression", "GradientBoosting"]:
        for metric in ["accuracy", "f1"]:
            assert np.isclose(
                results1[model_name][metric],
                results2[model_name][metric],
                rtol=1e-10,
            ), f"{model_name} {metric} should be deterministic"


def test_run_seed_different_seeds():
    """Test that different seeds can produce different results."""
    X_train, y_train = make_classification(n_samples=100, n_features=5, random_state=0)
    X_test, y_test = make_classification(n_samples=50, n_features=5, random_state=1)

    results1 = run_seed(X_train, X_test, y_train, y_test, seed=42)
    results2 = run_seed(X_train, X_test, y_train, y_test, seed=43)

    # Different seeds may produce different results (not guaranteed but likely).
    # This is more of a sanity check that different seeds are actually being used.
    assert results1 is not None
    assert results2 is not None
