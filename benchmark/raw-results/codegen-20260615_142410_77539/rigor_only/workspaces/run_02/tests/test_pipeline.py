"""Integration tests for the full experiment pipeline."""
import pytest
import os
import tempfile
import json
from pathlib import Path

from src.experiment import ExperimentRunner


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_experiment_runner_initialization():
    """Test ExperimentRunner can be initialized."""
    runner = ExperimentRunner(csv_path="churn.csv")
    assert runner.csv_path == "churn.csv"
    assert runner.results == []


def test_experiment_end_to_end():
    """Test full experiment pipeline (with minimal seeds for speed)."""
    # Ensure dataset exists
    if not os.path.exists("churn.csv"):
        pytest.skip("churn.csv not found; run generate_dataset first")

    runner = ExperimentRunner(csv_path="churn.csv")

    # Run experiment with just 2 seeds for testing (faster)
    result = runner.run_experiment(n_seeds=2)

    # Check structure
    assert 'config' in result
    assert 'results' in result
    assert 'all_seeds' in result

    # Check config
    config = result['config']
    assert 'n_duplicates_removed' in config
    assert 'class_balance' in config
    assert 'n_seeds' in config
    assert config['n_seeds'] == 2

    # Check results
    results = result['results']
    assert 'logistic_regression' in results
    assert 'gradient_boosting' in results

    lr_result = results['logistic_regression']
    gb_result = results['gradient_boosting']

    # Check all metrics present
    for metric in ['roc_auc', 'precision', 'recall', 'f1', 'neg_log_loss']:
        assert f'{metric}_mean' in lr_result
        assert f'{metric}_std' in lr_result
        assert f'{metric}_mean' in gb_result
        assert f'{metric}_std' in gb_result

    # Check metric values are reasonable
    assert 0 <= lr_result['roc_auc_mean'] <= 1
    assert 0 <= gb_result['roc_auc_mean'] <= 1

    # Check all_seeds per-seed results
    assert len(result['all_seeds']['logistic_regression']) == 2
    assert len(result['all_seeds']['gradient_boosting']) == 2


def test_save_results(temp_dir):
    """Test results are saved correctly."""
    if not os.path.exists("churn.csv"):
        pytest.skip("churn.csv not found; run generate_dataset first")

    runner = ExperimentRunner(csv_path="churn.csv")
    result = runner.run_experiment(n_seeds=2)

    # Save to temp directory
    runner.save_results(temp_dir, result)

    # Check files exist
    metrics_file = os.path.join(temp_dir, 'metrics.json')
    assert os.path.exists(metrics_file)

    # Check metrics.json is valid JSON
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    assert 'config' in metrics
    assert 'results' in metrics

    # Check REPORT.md exists
    assert os.path.exists('REPORT.md')
    with open('REPORT.md', 'r') as f:
        report = f.read()
    assert 'Gradient Boosting' in report
    assert 'Logistic Regression' in report


def test_sanity_checks_pass():
    """Test that sanity checks pass without assertion errors."""
    if not os.path.exists("churn.csv"):
        pytest.skip("churn.csv not found; run generate_dataset first")

    runner = ExperimentRunner(csv_path="churn.csv")

    # This should not raise any AssertionError
    result = runner.run_experiment(n_seeds=1)
    assert result is not None
