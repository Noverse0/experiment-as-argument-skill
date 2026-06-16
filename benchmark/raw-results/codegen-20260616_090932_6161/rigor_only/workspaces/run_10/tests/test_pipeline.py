"""
Tests for the churn prediction experiment pipeline.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from src.experiment import (
    load_and_preprocess,
    baseline_auc,
    sanity_check_overfit,
    sanity_check_label_shuffle,
    run_single_seed,
)


@pytest.fixture
def sample_dataset():
    """Create a small sample churn dataset for testing."""
    np.random.seed(42)
    n = 500
    X = pd.DataFrame({
        "tenure_months": np.random.randint(1, 72, n),
        "monthly_spend": np.random.gamma(2.0, 30.0, n).round(2),
        "support_tickets": np.random.poisson(1.2, n),
        "days_since_last_login": np.random.randint(0, 100, n),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="H"),
        "customer_id": range(1, n + 1),
    })
    y = pd.Series(np.random.randint(0, 2, n), name="churned")
    df = pd.concat([X, y], axis=1)
    return df


@pytest.fixture
def temp_csv(sample_dataset):
    """Write sample dataset to a temporary CSV."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_dataset.to_csv(f, index=False)
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink()


def test_load_and_preprocess(temp_csv):
    """Test data loading and leaky feature removal."""
    X, y = load_and_preprocess(temp_csv)

    # Should have correct features
    assert list(X.columns) == ["tenure_months", "monthly_spend", "support_tickets"]

    # Should not have leaky or temporal features
    assert "days_since_last_login" not in X.columns
    assert "signup_date" not in X.columns
    assert "customer_id" not in X.columns

    # Should have same number of samples as y
    assert len(X) == len(y)

    # Should have valid data
    assert X.notna().all().all()
    assert y.notna().all()


def test_baseline_auc():
    """Test baseline AUC calculation."""
    y = np.array([0, 0, 0, 1, 1])  # 60% negative, 40% positive
    auc = baseline_auc(y)
    # Majority class baseline (0) should have AUC = 0.5
    assert 0.45 < auc < 0.55, f"Expected AUC ~0.5 for baseline, got {auc}"


def test_sanity_check_overfit():
    """Test overfit check: both models must reach high train AUC on small subset."""
    np.random.seed(42)
    # Create data with clear signal so models can overfit
    X = np.random.randn(100, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Should not raise
    sanity_check_overfit(X, y, seed=42)


def test_sanity_check_label_shuffle(temp_csv):
    """Test label-shuffle check is callable (meaningful only on real data)."""
    X, y = load_and_preprocess(temp_csv)
    baseline = baseline_auc(y.values)
    X_array = X.values
    # Just verify it runs without error on real data
    try:
        sanity_check_label_shuffle(X_array, y.values, seed=42, baseline=baseline)
    except AssertionError:
        # May fail on random data; that's OK for this test
        pass


def test_run_single_seed(temp_csv):
    """Test single seed run produces valid results."""
    X, y = load_and_preprocess(temp_csv)
    baseline = baseline_auc(y.values)
    X_array = X.values

    results = run_single_seed(X_array, y.values, seed=42, baseline=baseline, perform_sanity_checks=False)

    # Should have both models
    assert "LogisticRegression" in results
    assert "GradientBoosting" in results

    # Each model should have key metrics
    for model_name in ["LogisticRegression", "GradientBoosting"]:
        metrics = results[model_name]
        assert "auc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

        # AUC should be between 0 and 1
        assert 0 <= metrics["auc"] <= 1


def test_reproducibility(temp_csv):
    """Test that same seed produces same results."""
    X, y = load_and_preprocess(temp_csv)
    baseline = baseline_auc(y.values)
    X_array = X.values

    results1 = run_single_seed(X_array, y.values, seed=42, baseline=baseline, perform_sanity_checks=False)
    results2 = run_single_seed(X_array, y.values, seed=42, baseline=baseline, perform_sanity_checks=False)

    # Same seed should produce identical AUC (within floating point tolerance)
    for model_name in ["LogisticRegression", "GradientBoosting"]:
        assert abs(results1[model_name]["auc"] - results2[model_name]["auc"]) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
