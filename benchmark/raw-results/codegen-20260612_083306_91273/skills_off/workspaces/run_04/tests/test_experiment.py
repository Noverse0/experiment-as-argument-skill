"""Tests for churn experiment pipeline."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.dataset import load_and_prepare, get_feature_names
from src.experiment import ChurnExperiment


@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        'tenure_months': np.random.randint(1, 72, n),
        'monthly_spend': np.random.gamma(2.0, 30.0, n).round(2),
        'support_tickets': np.random.poisson(1.2, n),
    })
    y = pd.Series(np.random.randint(0, 2, n), name='churned')
    return X, y


def test_load_and_prepare(tmp_path):
    """Test that data loading excludes leakage columns."""
    # Create test CSV with all columns
    df = pd.DataFrame({
        'customer_id': [1, 2, 3],
        'signup_date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'tenure_months': [10, 20, 30],
        'monthly_spend': [100.0, 200.0, 300.0],
        'support_tickets': [1, 2, 3],
        'account_status': ['active', 'closed', 'active'],
        'churned': [0, 1, 0],
    })
    csv_path = tmp_path / 'test.csv'
    df.to_csv(csv_path, index=False)

    X, y = load_and_prepare(str(csv_path))

    # Check shape and columns
    assert X.shape == (3, 3), "Should have 3 rows and 3 features"
    assert list(X.columns) == ['tenure_months', 'monthly_spend', 'support_tickets']
    assert len(y) == 3
    assert y.tolist() == [0, 1, 0]

    # Check excluded columns are gone
    assert 'customer_id' not in X.columns
    assert 'signup_date' not in X.columns
    assert 'account_status' not in X.columns


def test_deduplication(tmp_path):
    """Test that exact duplicates are removed."""
    df = pd.DataFrame({
        'customer_id': [1, 2, 3, 1],  # Row 0 and 3 are duplicates
        'signup_date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01'],
        'tenure_months': [10, 20, 30, 10],
        'monthly_spend': [100.0, 200.0, 300.0, 100.0],
        'support_tickets': [1, 2, 3, 1],
        'account_status': ['active', 'closed', 'active', 'active'],
        'churned': [0, 1, 0, 0],
    })
    csv_path = tmp_path / 'test_dupes.csv'
    df.to_csv(csv_path, index=False)

    X, y = load_and_prepare(str(csv_path))

    # After dedup, should have 3 rows
    assert len(X) == 3, "Duplicates should be removed"


def test_feature_names():
    """Test that feature names are consistent."""
    names = get_feature_names()
    assert names == ['tenure_months', 'monthly_spend', 'support_tickets']
    assert len(names) == 3


def test_experiment_sanity_checks(sample_data):
    """Test that sanity checks run without errors."""
    X, y = sample_data
    exp = ChurnExperiment(X, y, seeds=[42])
    checks = exp.run_sanity_checks()

    assert 'baseline_auc' in checks
    assert 'normal_auc' in checks
    assert 'shuffled_auc' in checks
    assert 'label_shuffle_valid' in checks
    assert 0 <= checks['baseline_auc'] <= 1
    assert 0 <= checks['normal_auc'] <= 1
    assert 0 <= checks['shuffled_auc'] <= 1


def test_experiment_comparison(sample_data):
    """Test that comparison runs and produces results."""
    X, y = sample_data
    exp = ChurnExperiment(X, y, seeds=[42, 123])
    results = exp.run_comparison()

    assert 'LogisticRegression' in results
    assert 'GradientBoosting' in results
    assert len(results['LogisticRegression']) == 2  # Two seeds
    assert len(results['GradientBoosting']) == 2

    # Check structure of first result
    first_lr = results['LogisticRegression'][0]
    assert 'auc' in first_lr
    assert 'accuracy' in first_lr
    assert 'precision' in first_lr
    assert 'recall' in first_lr
    assert 'f1' in first_lr


def test_summarize_results(sample_data):
    """Test that summary statistics are computed correctly."""
    X, y = sample_data
    exp = ChurnExperiment(X, y, seeds=[42, 123])
    results = exp.run_comparison()
    summary = exp.summarize_results(results)

    assert 'LogisticRegression' in summary
    assert 'GradientBoosting' in summary

    lr_summary = summary['LogisticRegression']
    assert 'auc' in lr_summary
    assert 'mean' in lr_summary['auc']
    assert 'std' in lr_summary['auc']
    assert 'values' in lr_summary['auc']
    assert len(lr_summary['auc']['values']) == 2  # Two seeds


def test_write_results(tmp_path, sample_data):
    """Test that results are written to JSON."""
    X, y = sample_data
    exp = ChurnExperiment(X, y, seeds=[42])
    checks = exp.run_sanity_checks()
    results = exp.run_comparison()
    summary = exp.summarize_results(results)

    output_dir = tmp_path / "results"
    exp.write_results(str(output_dir), checks, summary)

    result_file = output_dir / "metrics.json"
    assert result_file.exists()

    import json
    with open(result_file) as f:
        data = json.load(f)
    assert 'claim' in data
    assert 'design' in data
    assert 'results' in data
    assert 'sanity_checks' in data
