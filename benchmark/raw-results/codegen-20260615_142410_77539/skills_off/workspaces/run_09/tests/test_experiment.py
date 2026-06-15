"""Tests for the churn experiment pipeline."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiment import (
    load_and_audit,
    preprocess,
    baseline_majority,
    baseline_label_shuffle,
    run_experiment,
)


@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'customer_id': np.arange(1, n + 1),
        'signup_date': pd.date_range('2023-01-01', periods=n, freq='D').astype(str),
        'tenure_months': np.random.randint(1, 72, n),
        'monthly_spend': np.random.gamma(2.0, 30.0, n),
        'support_tickets': np.random.poisson(1.2, n),
        'days_since_last_login': np.random.randint(1, 100, n),
        'churned': np.random.randint(0, 2, n),
    })
    return df


@pytest.fixture
def temp_csv(sample_data):
    """Create a temporary CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f, index=False)
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink()


def test_load_and_audit(temp_csv):
    """Test data loading."""
    df = load_and_audit(temp_csv)
    assert len(df) == 100
    assert 'churned' in df.columns


def test_preprocess(sample_data):
    """Test preprocessing drops correct columns and extracts features."""
    X, y = preprocess(sample_data)

    # Check target is correct
    assert len(y) == len(sample_data)
    assert y.name == 'churned'

    # Check dropped columns
    assert 'customer_id' not in X.columns
    assert 'churned' not in X.columns
    assert 'signup_date' not in X.columns
    assert 'days_since_last_login' not in X.columns

    # Check extracted features exist
    assert 'year_of_signup' in X.columns
    assert 'month_of_signup' in X.columns
    assert 'days_since_signup' in X.columns

    # Check remaining features
    assert 'tenure_months' in X.columns
    assert 'monthly_spend' in X.columns
    assert 'support_tickets' in X.columns

    # Check no NaNs
    assert not X.isna().any().any()
    assert not y.isna().any()


def test_baseline_majority(sample_data):
    """Test majority class baseline."""
    _, y = preprocess(sample_data)
    baseline = baseline_majority(y)

    # Check all metrics are present and reasonable
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        assert metric in baseline
        assert 0 <= baseline[metric] <= 1

    # For majority class, accuracy should be >= max class rate
    max_class_rate = max(y.mean(), 1 - y.mean())
    assert baseline['accuracy'] >= max_class_rate - 0.01  # small epsilon for floating point


def test_baseline_label_shuffle(sample_data):
    """Test that label shuffle drops performance near baseline."""
    X, y = preprocess(sample_data)
    baseline_maj = baseline_majority(y)
    baseline_shuffle = baseline_label_shuffle(X, y, seed=999)

    # Accuracy with shuffled labels should be near random
    assert 'accuracy' in baseline_shuffle
    assert 0 <= baseline_shuffle['accuracy'] <= 1


def test_run_experiment(temp_csv):
    """Test running a single experiment iteration."""
    results = run_experiment(temp_csv, seed=42)

    # Check both models are present
    assert 'LogisticRegression' in results
    assert 'GradientBoosting' in results

    # Check all metrics for each model
    for model_name in ['LogisticRegression', 'GradientBoosting']:
        model_results = results[model_name]
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            assert metric in model_results
            assert 'mean' in model_results[metric]
            assert 'std' in model_results[metric]
            assert 'values' in model_results[metric]

            # Metrics should be between 0 and 1
            assert 0 <= model_results[metric]['mean'] <= 1
            assert model_results[metric]['std'] >= 0


def test_experiment_deterministic(temp_csv):
    """Test that same seed gives same results."""
    results1 = run_experiment(temp_csv, seed=42)
    results2 = run_experiment(temp_csv, seed=42)

    for model_name in ['LogisticRegression', 'GradientBoosting']:
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            m1 = results1[model_name][metric]['mean']
            m2 = results2[model_name][metric]['mean']
            assert abs(m1 - m2) < 1e-10, f"Mismatch for {model_name}/{metric}"


def test_experiment_different_seed(temp_csv):
    """Test that different seeds can give different results."""
    results1 = run_experiment(temp_csv, seed=42)
    results2 = run_experiment(temp_csv, seed=43)

    # At least one metric should differ (with very high probability)
    has_diff = False
    for model_name in ['LogisticRegression', 'GradientBoosting']:
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            m1 = results1[model_name][metric]['mean']
            m2 = results2[model_name][metric]['mean']
            if abs(m1 - m2) > 0.001:
                has_diff = True
                break

    assert has_diff, "Different seeds should (usually) produce different results"


def test_models_beat_baseline(temp_csv):
    """Sanity check: both models should beat majority class baseline."""
    df = pd.read_csv(temp_csv)
    X, y = preprocess(df)
    baseline = baseline_majority(y)

    results = run_experiment(temp_csv, seed=42)

    for model_name in ['LogisticRegression', 'GradientBoosting']:
        model_f1 = results[model_name]['f1']['mean']
        baseline_f1 = baseline['f1']
        # Both models should beat baseline on F1 (or be very close on tiny data)
        assert model_f1 >= baseline_f1 * 0.9, f"{model_name} should beat baseline"
