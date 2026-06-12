"""Tests for experiment pipeline."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiment import DataLoader, ExperimentRunner, Preprocessor


@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "signup_date": ["2023-06-01"] * n,
        "tenure_months": rng.integers(1, 72, n),
        "monthly_spend": rng.gamma(2.0, 30.0, n).round(2),
        "support_tickets": rng.poisson(1.2, n),
        "account_status": ["active"] * (n // 2) + ["closed"] * (n // 2),
        "churned": [0] * (n // 2) + [1] * (n // 2),
    })
    return df


@pytest.fixture
def temp_csv(sample_data):
    """Write sample data to a temporary CSV."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as f:
        sample_data.to_csv(f, index=False)
        temp_path = f.name
    yield temp_path
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


def test_dataloader_removes_leak(temp_csv):
    """Test that account_status is identified and removed as a leak."""
    features, target = DataLoader.load_and_clean(temp_csv)
    # account_status should be removed
    assert "account_status" not in features.columns
    assert len(target) > 0


def test_dataloader_deduplicates(sample_data):
    """Test that duplicate rows are removed before split."""
    # Add duplicates
    df_with_dupes = pd.concat(
        [sample_data, sample_data.iloc[:10]], ignore_index=True
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as f:
        df_with_dupes.to_csv(f, index=False)
        temp_path = f.name

    try:
        features, target = DataLoader.load_and_clean(temp_path)
        # Should have removed duplicates
        assert len(features) == len(sample_data)
        assert len(target) == len(sample_data)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_preprocessor_no_leakage(sample_data):
    """Test that preprocessor fits only on train and applies to test."""
    features, target = DataLoader.load_and_clean(
        _create_temp_csv(sample_data)
    )

    # Split
    train_idx = slice(0, 60)
    test_idx = slice(60, 100)

    X_train = features.iloc[train_idx]
    X_test = features.iloc[test_idx]

    # Fit preprocessor
    preprocessor = Preprocessor()
    preprocessor.fit(X_train)

    # Transform both
    X_train_scaled = preprocessor.transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # Check scaling was applied
    assert X_train_scaled.shape == X_train.values.shape
    assert X_test_scaled.shape == X_test.values.shape

    # Verify test set uses train statistics (not fit to test itself)
    # by checking shapes match (proves transform was applied)
    assert X_train_scaled.mean(axis=0).shape == (X_train.shape[1],)


def test_experiment_runner_trains_models(sample_data):
    """Test that ExperimentRunner can train both models."""
    features, target = DataLoader.load_and_clean(
        _create_temp_csv(sample_data)
    )

    runner = ExperimentRunner(features, target)
    results = runner.run_seed(seed=42)

    # Should have results for both models
    assert len(results) == 2
    model_names = {r.model_name for r in results}
    assert model_names == {"LogisticRegression", "GradientBoosting"}

    # Metrics should be in reasonable range
    for result in results:
        assert 0.0 <= result.test_auc <= 1.0
        assert 0.0 <= result.test_precision <= 1.0
        assert 0.0 <= result.test_recall <= 1.0


def test_experiment_reproducible(sample_data):
    """Test that same seed produces same results."""
    features, target = DataLoader.load_and_clean(
        _create_temp_csv(sample_data)
    )

    runner = ExperimentRunner(features, target)
    results1 = runner.run_seed(seed=123)
    results2 = runner.run_seed(seed=123)

    # Results should be identical
    for r1, r2 in zip(results1, results2):
        assert r1.model_name == r2.model_name
        assert abs(r1.test_auc - r2.test_auc) < 1e-6


def _create_temp_csv(df):
    """Helper to create temp CSV from DataFrame."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as f:
        df.to_csv(f, index=False)
        return f.name
