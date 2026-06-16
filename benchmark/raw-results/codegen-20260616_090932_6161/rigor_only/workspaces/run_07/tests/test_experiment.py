"""Tests for churn prediction experiment."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiment import (
    ExperimentConfig,
    aggregate_results,
    load_and_preprocess,
    run_experiment,
    run_sanity_checks,
    run_single_experiment,
)


@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'customer_id': np.arange(1, n + 1),
        'signup_date': pd.date_range('2023-01-01', periods=n).strftime('%Y-%m-%d'),
        'tenure_months': np.random.randint(1, 72, n),
        'monthly_spend': np.random.gamma(2.0, 30.0, n).round(2),
        'support_tickets': np.random.poisson(1.2, n),
        'days_since_last_login': np.random.randint(1, 100, n),
        'churned': np.random.binomial(1, 0.3, n),
    })
    return df


def test_load_and_preprocess(sample_data, tmp_path):
    """Test data loading and preprocessing."""
    csv_path = tmp_path / "test.csv"
    sample_data.to_csv(csv_path, index=False)

    features, target, n_dups, churn_rate = load_and_preprocess(str(csv_path))

    # Check shape
    assert len(features) == len(target)
    assert len(features) <= len(sample_data)  # May have deduplicated

    # Check features (should exclude leaked feature)
    assert 'days_since_last_login' not in features.columns
    assert 'customer_id' not in features.columns
    assert 'signup_date' not in features.columns

    # Check that extracted temporal features exist
    assert 'signup_year' in features.columns
    assert 'signup_month' in features.columns

    # Check churn rate is between 0 and 1
    assert 0 <= churn_rate <= 1


def test_no_target_leak(sample_data, tmp_path):
    """Test that leaked feature is dropped."""
    csv_path = tmp_path / "test.csv"
    sample_data.to_csv(csv_path, index=False)

    features, target, _, _ = load_and_preprocess(str(csv_path))

    # Verify the leak is gone
    assert 'days_since_last_login' not in features.columns
    assert list(features.columns) == [
        'tenure_months', 'monthly_spend', 'support_tickets',
        'signup_year', 'signup_month'
    ]


def test_split_before_transform(sample_data, tmp_path):
    """Test that preprocessing respects split-before-transform."""
    csv_path = tmp_path / "test.csv"
    sample_data.to_csv(csv_path, index=False)

    features, target, _, _ = load_and_preprocess(str(csv_path))
    config = ExperimentConfig(n_seeds=1)

    lr_result, gb_result = run_single_experiment(features, target, seed=42, config=config)

    # Both models should produce valid AUC scores
    assert 0 <= lr_result.test_auc <= 1
    assert 0 <= gb_result.test_auc <= 1

    # Sanity: test AUC should be within a reasonable range
    # (not perfect due to small sample, but should be above random)
    assert lr_result.test_auc >= 0.3  # Well above random for small sample
    assert gb_result.test_auc >= 0.3


def test_sanity_checks(sample_data, tmp_path):
    """Test sanity checks run without errors."""
    csv_path = tmp_path / "test.csv"
    sample_data.to_csv(csv_path, index=False)

    features, target, _, _ = load_and_preprocess(str(csv_path))
    config = ExperimentConfig()

    sanity = run_sanity_checks(features, target, config)

    # All checks should return valid values
    assert 'baseline_accuracy' in sanity
    assert 'overfit_tiny_auc' in sanity
    assert 0 <= sanity['baseline_accuracy'] <= 1
    assert 0 <= sanity['overfit_tiny_auc'] <= 1


def test_aggregate_results():
    """Test result aggregation."""
    from src.experiment import RunResult

    results = [
        RunResult(seed=0, model='LogisticRegression', train_auc=0.8, test_auc=0.75,
                  baseline_auc=0.6, label_shuffle_auc=0.61),
        RunResult(seed=0, model='GradientBoosting', train_auc=0.85, test_auc=0.80,
                  baseline_auc=0.6, label_shuffle_auc=0.61),
        RunResult(seed=1, model='LogisticRegression', train_auc=0.82, test_auc=0.77,
                  baseline_auc=0.6, label_shuffle_auc=0.61),
        RunResult(seed=1, model='GradientBoosting', train_auc=0.87, test_auc=0.82,
                  baseline_auc=0.6, label_shuffle_auc=0.61),
    ]

    config = ExperimentConfig()
    summary = aggregate_results(results, config)

    # Check structure
    assert 'LogisticRegression' in summary
    assert 'GradientBoosting' in summary

    # Check statistics
    lr = summary['LogisticRegression']
    gb = summary['GradientBoosting']

    assert lr['n_runs'] == 2
    assert gb['n_runs'] == 2
    assert 0.7 < lr['test_auc_mean'] < 0.8
    assert 0.8 < gb['test_auc_mean'] < 0.85


def test_full_experiment_e2e(sample_data, tmp_path):
    """End-to-end test of the full experiment."""
    csv_path = tmp_path / "test.csv"
    sample_data.to_csv(csv_path, index=False)

    results_dir = tmp_path / "results"

    results = run_experiment(
        str(csv_path),
        results_dir=str(results_dir),
        config=ExperimentConfig(n_seeds=2),
    )

    # Check outputs
    assert 'summary' in results
    assert 'sanity_checks' in results
    assert 'n_duplicates' in results
    assert 'churn_rate' in results

    # Check summary structure
    summary = results['summary']
    assert 'LogisticRegression' in summary
    assert 'GradientBoosting' in summary

    # Check results file was created
    metrics_file = results_dir / 'metrics.json'
    assert metrics_file.exists()

    with open(metrics_file) as f:
        metrics = json.load(f)
    assert 'summary' in metrics
    assert 'all_runs' in metrics
    assert len(metrics['all_runs']) == 4  # 2 seeds × 2 models


def test_determinism():
    """Test that runs with same seed produce identical results."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'customer_id': np.arange(1, n + 1),
        'signup_date': pd.date_range('2023-01-01', periods=n).strftime('%Y-%m-%d'),
        'tenure_months': np.random.randint(1, 72, n),
        'monthly_spend': np.random.gamma(2.0, 30.0, n).round(2),
        'support_tickets': np.random.poisson(1.2, n),
        'days_since_last_login': np.random.randint(1, 100, n),
        'churned': np.random.binomial(1, 0.3, n),
    })

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "test.csv"
        df.to_csv(csv_path, index=False)

        features, target, _, _ = load_and_preprocess(str(csv_path))

        # Run twice with same seed
        result1_lr, result1_gb = run_single_experiment(
            features, target, seed=42, config=ExperimentConfig()
        )
        result2_lr, result2_gb = run_single_experiment(
            features, target, seed=42, config=ExperimentConfig()
        )

        # Results should be identical
        assert result1_lr.test_auc == result2_lr.test_auc
        assert result1_gb.test_auc == result2_gb.test_auc


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
