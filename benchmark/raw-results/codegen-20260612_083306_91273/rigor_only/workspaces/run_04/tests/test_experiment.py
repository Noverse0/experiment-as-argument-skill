"""Tests for the churn prediction experiment."""
import pytest
from pathlib import Path

from src.experiment import ChurnExperiment


@pytest.fixture
def csv_path():
    """Path to the generated churn dataset."""
    return "churn.csv"


def test_experiment_runs(csv_path):
    """Test that the experiment runs without error."""
    if not Path(csv_path).exists():
        pytest.skip(f"{csv_path} not found; run make_dataset.py first")

    exp = ChurnExperiment(csv_path, seed=42)
    results = exp.run()

    assert 'duplicates_removed' in results
    assert 'n_samples' in results
    assert 'churn_rate' in results
    assert 'n_features' in results
    assert 'model_comparison' in results


def test_sanity_checks_pass(csv_path):
    """Test that sanity checks are computed."""
    if not Path(csv_path).exists():
        pytest.skip(f"{csv_path} not found; run make_dataset.py first")

    exp = ChurnExperiment(csv_path, seed=42)
    results = exp.run()

    # Sanity checks should not raise exceptions
    assert exp.sanity_checks['baseline_f1'] >= 0
    assert exp.sanity_checks['overfit_test_loss'] >= 0
    assert exp.sanity_checks['label_shuffle_f1'] >= 0


def test_model_comparison_has_variance(csv_path):
    """Test that model comparison captures variance across runs."""
    if not Path(csv_path).exists():
        pytest.skip(f"{csv_path} not found; run make_dataset.py first")

    exp = ChurnExperiment(csv_path, seed=42)
    results = exp.run()

    comp = results['model_comparison']
    assert 'LogisticRegression' in comp
    assert 'GradientBoosting' in comp

    for model_name in ['LogisticRegression', 'GradientBoosting']:
        for metric in ['precision', 'recall', 'f1', 'roc_auc']:
            m = comp[model_name][metric]
            assert 'mean' in m
            assert 'std' in m
            assert 'runs' in m
            assert len(m['runs']) == 3  # 3 runs
            assert m['mean'] > 0
            assert m['std'] >= 0


def test_churn_rate_reasonable(csv_path):
    """Test that churn rate is in a reasonable range."""
    if not Path(csv_path).exists():
        pytest.skip(f"{csv_path} not found; run make_dataset.py first")

    exp = ChurnExperiment(csv_path, seed=42)
    results = exp.run()

    # Churn rate should be between 0 and 1
    assert 0 < results['churn_rate'] < 1


def test_deduplication_works(csv_path):
    """Test that deduplication removes duplicates."""
    if not Path(csv_path).exists():
        pytest.skip(f"{csv_path} not found; run make_dataset.py first")

    exp = ChurnExperiment(csv_path, seed=42)
    results = exp.run()

    # Dataset has 200 duplicates
    assert results['duplicates_removed'] == 200
    # After dedup: 4000 original + 200 dupes - 200 dropped = 4000
    assert results['n_samples'] == 4000
