"""Tests for the main experiment."""

import pytest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

from src.experiment import ChurnExperiment


@pytest.fixture
def temp_churn_csv(tmp_path):
    """Create a temporary churn dataset for testing."""
    data = {
        'customer_id': range(1, 201),
        'signup_date': pd.date_range('2023-01-01', periods=200),
        'tenure_months': np.random.randint(1, 100, 200),
        'monthly_spend': np.random.uniform(10, 200, 200),
        'support_tickets': np.random.randint(0, 10, 200),
        'account_status': np.random.choice(['active', 'closed'], 200),
        'churned': np.random.randint(0, 2, 200),
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_churn.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_experiment_initialization(temp_churn_csv):
    """Test experiment initialization."""
    seeds = [42, 123]
    exp = ChurnExperiment(temp_churn_csv, seeds=seeds)

    assert exp.data_path == temp_churn_csv
    assert exp.seeds == seeds
    assert len(exp.results['runs']) == 0


def test_experiment_run_single_seed(temp_churn_csv):
    """Test running a single seed."""
    exp = ChurnExperiment(temp_churn_csv, seeds=[42])
    result = exp.run_seed(42)

    # Check result structure
    assert 'seed' in result
    assert 'data_split' in result
    assert 'baseline' in result
    assert 'models' in result
    assert 'logistic_regression' in result['models']
    assert 'gradient_boosting' in result['models']

    # Check metrics are present
    for model_name in ['logistic_regression', 'gradient_boosting']:
        metrics = result['models'][model_name]
        assert 'roc_auc' in metrics
        assert 'f1' in metrics
        assert 'accuracy' in metrics


def test_experiment_full_run(temp_churn_csv):
    """Test full experiment run across multiple seeds."""
    seeds = [42, 123, 456]
    exp = ChurnExperiment(temp_churn_csv, seeds=seeds)
    results = exp.run()

    # Check top-level structure
    assert 'config' in results
    assert 'runs' in results
    assert 'summary' in results

    # Check number of runs
    assert len(results['runs']) == len(seeds)

    # Check summary structure
    summary = results['summary']
    assert 'logistic_regression' in summary
    assert 'gradient_boosting' in summary
    assert 'comparison' in summary

    # Check aggregated metrics
    for model_name in ['logistic_regression', 'gradient_boosting']:
        for metric_name in ['roc_auc', 'f1', 'accuracy']:
            agg = summary[model_name][metric_name]
            assert 'mean' in agg
            assert 'std' in agg
            assert 'values' in agg
            assert len(agg['values']) == len(seeds)


def test_experiment_comparison_metrics(temp_churn_csv):
    """Test that comparison metrics are computed."""
    exp = ChurnExperiment(temp_churn_csv, seeds=[42, 123])
    results = exp.run()

    comparison = results['summary']['comparison']
    assert 'primary_metric' in comparison
    assert comparison['primary_metric'] == 'roc_auc'
    assert 'gb_auc_mean' in comparison
    assert 'lr_auc_mean' in comparison
    assert 'difference' in comparison
    assert 'gb_wins' in comparison

    # Difference should be well-defined
    assert comparison['difference'] == (
        comparison['gb_auc_mean'] - comparison['lr_auc_mean']
    )


def test_experiment_models_produce_valid_metrics(temp_churn_csv):
    """Test that both models produce valid metrics."""
    exp = ChurnExperiment(temp_churn_csv, seeds=[42])
    results = exp.run()

    # Get model accuracies
    for run in results['runs']:
        lr_acc = run['models']['logistic_regression']['accuracy']
        gb_acc = run['models']['gradient_boosting']['accuracy']

        # Metrics should be in valid range
        assert 0 <= lr_acc <= 1, f"LR accuracy out of range: {lr_acc}"
        assert 0 <= gb_acc <= 1, f"GB accuracy out of range: {gb_acc}"


def test_experiment_deterministic(temp_churn_csv):
    """Test that same seed produces same results."""
    exp1 = ChurnExperiment(temp_churn_csv, seeds=[42])
    results1 = exp1.run()
    metrics1_lr = results1['runs'][0]['models']['logistic_regression']['roc_auc']
    metrics1_gb = results1['runs'][0]['models']['gradient_boosting']['roc_auc']

    exp2 = ChurnExperiment(temp_churn_csv, seeds=[42])
    results2 = exp2.run()
    metrics2_lr = results2['runs'][0]['models']['logistic_regression']['roc_auc']
    metrics2_gb = results2['runs'][0]['models']['gradient_boosting']['roc_auc']

    assert metrics1_lr == metrics2_lr
    assert metrics1_gb == metrics2_gb


def test_experiment_json_export(temp_churn_csv, tmp_path):
    """Test JSON export."""
    exp = ChurnExperiment(temp_churn_csv, seeds=[42])
    results = exp.run()

    output_path = tmp_path / "results.json"
    exp.to_json(str(output_path))

    assert output_path.exists()

    # Verify JSON is valid and contains expected structure
    import json
    with open(output_path) as f:
        exported = json.load(f)
    assert 'config' in exported
    assert 'runs' in exported
    assert 'summary' in exported
