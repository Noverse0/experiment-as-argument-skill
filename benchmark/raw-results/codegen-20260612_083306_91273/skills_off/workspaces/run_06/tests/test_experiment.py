"""Tests for the churn prediction experiment."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiment import (
    detect_duplicates,
    load_data,
    preprocess_data,
    run_experiment,
    run_single_seed,
)
from src.sanity_checks import (
    check_baseline_floor,
    check_label_shuffle,
    check_leakage_ceiling,
    check_overfit_tiny_subset,
)


@pytest.fixture
def sample_data():
    """Create a small sample churn dataset for testing."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'customer_id': np.arange(1, n + 1),
        'signup_date': ['2023-01-01'] * n,
        'tenure_months': np.random.randint(1, 72, n),
        'monthly_spend': np.random.gamma(2.0, 30.0, n).round(2),
        'support_tickets': np.random.poisson(1.2, n),
        'account_status': np.random.choice(['active', 'closed'], n),
        'churned': np.random.choice([0, 1], n),
    })
    return df


def test_load_data(tmp_path):
    """Test loading CSV data."""
    csv_file = tmp_path / "test.csv"
    df_original = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6],
    })
    df_original.to_csv(csv_file, index=False)

    df_loaded = load_data(str(csv_file))
    pd.testing.assert_frame_equal(df_loaded, df_original)


def test_detect_duplicates(sample_data):
    """Test duplicate detection."""
    # Add exact duplicates
    df_with_dups = pd.concat([sample_data, sample_data.iloc[:5]], ignore_index=True)

    n_dups, df_dedup = detect_duplicates(df_with_dups)

    assert n_dups == 5
    assert len(df_dedup) == len(sample_data)


def test_preprocess_data(sample_data):
    """Test that preprocessing drops correct columns."""
    df_processed = preprocess_data(sample_data)

    # Should have exactly 4 columns: 3 features + 1 target
    assert list(df_processed.columns) == [
        'tenure_months', 'monthly_spend', 'support_tickets', 'churned'
    ]
    assert len(df_processed) == len(sample_data)

    # Should not contain leaked/unused columns
    assert 'account_status' not in df_processed.columns
    assert 'customer_id' not in df_processed.columns
    assert 'signup_date' not in df_processed.columns


def test_run_single_seed(sample_data):
    """Test single seed run produces metrics."""
    df_processed = preprocess_data(sample_data)
    metrics = run_single_seed(df_processed, seed=42)

    # Should have metrics for both models
    assert 'lr_auc' in metrics
    assert 'gb_auc' in metrics
    assert 'lr_accuracy' in metrics
    assert 'gb_accuracy' in metrics
    assert 'seed' in metrics

    # Metrics should be between 0 and 1
    assert 0 <= metrics['lr_auc'] <= 1
    assert 0 <= metrics['gb_auc'] <= 1
    assert 0 <= metrics['lr_accuracy'] <= 1
    assert 0 <= metrics['gb_accuracy'] <= 1


def test_baseline_floor(sample_data):
    """Test baseline floor check."""
    df_processed = preprocess_data(sample_data)
    passed = check_baseline_floor(df_processed)
    assert passed is not None


def test_overfit_tiny_subset(sample_data):
    """Test overfit on tiny subset check."""
    df_processed = preprocess_data(sample_data)
    passed = check_overfit_tiny_subset(df_processed)
    assert passed is not None


def test_label_shuffle(sample_data):
    """Test label shuffle check."""
    df_processed = preprocess_data(sample_data)
    passed = check_label_shuffle(df_processed)
    assert passed is not None


def test_leakage_ceiling(sample_data):
    """Test leakage ceiling check."""
    df_processed = preprocess_data(sample_data)
    passed = check_leakage_ceiling(df_processed)
    assert passed is not None


def test_run_experiment(sample_data, tmp_path):
    """Test full experiment with small sample."""
    # Save sample data to CSV
    csv_file = tmp_path / "test_churn.csv"
    sample_data.to_csv(csv_file, index=False)

    # Run experiment with 2 seeds for speed
    result = run_experiment(str(csv_file), n_seeds=2, output_dir=str(tmp_path / "results"))

    # Check result structure
    assert 'models' in result
    assert 'LogisticRegression' in result['models']
    assert 'GradientBoosting' in result['models']
    assert 'n_seeds' in result
    assert result['n_seeds'] == 2

    # Check each model has required metrics
    for model_name in ['LogisticRegression', 'GradientBoosting']:
        model_metrics = result['models'][model_name]
        assert 'auc' in model_metrics
        assert 'mean' in model_metrics['auc']
        assert 'std' in model_metrics['auc']

    # Check results files were written
    results_dir = tmp_path / "results"
    assert (results_dir / "results.json").exists()
    assert (results_dir / "metrics_by_seed.csv").exists()

    # Verify JSON is valid and readable
    with open(results_dir / "results.json") as f:
        loaded_result = json.load(f)
    assert loaded_result['n_seeds'] == 2


def test_results_json_structure(sample_data, tmp_path):
    """Test that results.json has expected structure."""
    csv_file = tmp_path / "test_churn.csv"
    sample_data.to_csv(csv_file, index=False)

    result = run_experiment(str(csv_file), n_seeds=2, output_dir=str(tmp_path / "results"))

    # Check effect size is computed
    assert 'effect_size' in result
    assert 'mean_diff_auc' in result['effect_size']
    assert 'cohens_d' in result['effect_size']
    assert 'overlapping' in result['effect_size']


def test_metrics_are_numeric(sample_data, tmp_path):
    """Test that all metrics are valid numbers."""
    csv_file = tmp_path / "test_churn.csv"
    sample_data.to_csv(csv_file, index=False)

    result = run_experiment(str(csv_file), n_seeds=2, output_dir=str(tmp_path / "results"))

    # Sample and verify metric types
    lr_auc_mean = result['models']['LogisticRegression']['auc']['mean']
    gb_auc_mean = result['models']['GradientBoosting']['auc']['mean']

    assert isinstance(lr_auc_mean, float)
    assert isinstance(gb_auc_mean, float)
    assert not np.isnan(lr_auc_mean)
    assert not np.isnan(gb_auc_mean)
    assert 0 <= lr_auc_mean <= 1
    assert 0 <= gb_auc_mean <= 1
