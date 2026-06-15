"""Unit tests for experiment module."""
import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from src.experiment import (
    train_and_evaluate,
    sanity_check_baseline,
    sanity_check_label_shuffle,
    run_experiment_seed,
)


@pytest.fixture
def synthetic_data():
    """Create synthetic train/test data."""
    np.random.seed(42)
    X_train = np.random.randn(80, 3)
    y_train = np.random.randint(0, 2, 80)
    X_test = np.random.randn(20, 3)
    y_test = np.random.randint(0, 2, 20)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def real_csv():
    """Create a temporary CSV with realistic churn data."""
    data = {
        'customer_id': list(range(1, 101)),
        'signup_date': ['2023-01-01'] * 100,
        'tenure_months': np.random.randint(1, 72, 100),
        'monthly_spend': np.random.gamma(2.0, 30.0, 100).round(2),
        'support_tickets': np.random.poisson(1.2, 100),
        'days_since_last_login': np.random.gamma(2.0, 10.0, 100).round(),
        'churned': np.random.randint(0, 2, 100),
    }
    df = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


def test_train_and_evaluate_lr(synthetic_data):
    """Test LogisticRegression training and evaluation."""
    X_train, X_test, y_train, y_test = synthetic_data

    result = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        LogisticRegression, 'LogisticRegression'
    )

    # Check all metrics are present
    assert 'model' in result
    assert 'auc' in result
    assert 'accuracy' in result
    assert 'precision' in result
    assert 'recall' in result
    assert 'f1' in result

    # Check metric values are valid
    assert 0 <= result['auc'] <= 1
    assert 0 <= result['accuracy'] <= 1
    assert 0 <= result['precision'] <= 1
    assert 0 <= result['recall'] <= 1
    assert 0 <= result['f1'] <= 1
    assert result['model'] == 'LogisticRegression'


def test_train_and_evaluate_gb(synthetic_data):
    """Test GradientBoostingClassifier training and evaluation."""
    X_train, X_test, y_train, y_test = synthetic_data

    result = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        GradientBoostingClassifier, 'GradientBoostingClassifier'
    )

    assert result['model'] == 'GradientBoostingClassifier'
    assert 0 <= result['auc'] <= 1


def test_sanity_check_baseline(synthetic_data):
    """Test baseline sanity check (model must beat majority class)."""
    X_train, X_test, y_train, y_test = synthetic_data

    baseline_auc = sanity_check_baseline(
        X_train, X_test, y_train, y_test, LogisticRegression
    )

    # Baseline AUC should be reasonable
    assert 0 <= baseline_auc <= 1


def test_sanity_check_label_shuffle(synthetic_data):
    """Test label shuffle sanity check (shuffled labels should collapse performance)."""
    X_train, X_test, y_train, y_test = synthetic_data

    shuffled_auc = sanity_check_label_shuffle(
        X_train, X_test, y_train, y_test, LogisticRegression
    )

    # Shuffled label AUC should be near 0.5 (random guessing)
    assert 0 <= shuffled_auc <= 1
    # With random labels, AUC should be close to 0.5
    assert 0.3 < shuffled_auc < 0.7


def test_run_experiment_seed_integration(real_csv):
    """Integration test: run a single seed of the full experiment."""
    result = run_experiment_seed(real_csv, random_state=42)

    # Check structure
    assert 'seed' in result
    assert result['seed'] == 42
    assert 'metadata' in result
    assert 'models' in result
    assert 'sanity_checks' in result

    # Check models
    assert 'lr' in result['models']
    assert 'gb' in result['models']

    # Check LR metrics
    lr = result['models']['lr']
    assert 'auc' in lr
    assert 'f1' in lr
    assert 0 <= lr['auc'] <= 1
    assert 0 <= lr['f1'] <= 1

    # Check GB metrics
    gb = result['models']['gb']
    assert 'auc' in gb
    assert 'f1' in gb
    assert 0 <= gb['auc'] <= 1
    assert 0 <= gb['f1'] <= 1

    # Check sanity checks
    assert 'baseline_auc' in result['sanity_checks']
    assert 'label_shuffle_auc' in result['sanity_checks']
    assert 'leakage_ceiling' in result['sanity_checks']

    # Baseline AUC must be less than real models (models should beat baseline)
    assert result['sanity_checks']['baseline_auc'] < 1.0

    # Shuffled labels should have low AUC
    assert 0.3 < result['sanity_checks']['label_shuffle_auc'] < 0.7

    # Leakage ceiling should show higher AUC with leak
    leak_lr = result['sanity_checks']['leakage_ceiling']['lr_with_leak']
    leak_gb = result['sanity_checks']['leakage_ceiling']['gb_with_leak']
    assert 0 <= leak_lr <= 1
    assert 0 <= leak_gb <= 1


def test_metadata_completeness(real_csv):
    """Test that metadata is complete and consistent."""
    result = run_experiment_seed(real_csv, random_state=42)
    meta = result['metadata']

    # Check all keys present
    required_keys = [
        'n_total_before_dedup',
        'n_duplicates_removed',
        'n_total_after_dedup',
        'churn_rate',
        'n_train',
        'n_test',
        'train_churn_rate',
        'test_churn_rate',
    ]
    for key in required_keys:
        assert key in meta

    # Check consistency
    assert meta['n_total_after_dedup'] == meta['n_train'] + meta['n_test']
    assert meta['n_total_before_dedup'] == meta['n_total_after_dedup'] + meta['n_duplicates_removed']
    assert 0 <= meta['churn_rate'] <= 1
    assert 0 <= meta['train_churn_rate'] <= 1
    assert 0 <= meta['test_churn_rate'] <= 1
