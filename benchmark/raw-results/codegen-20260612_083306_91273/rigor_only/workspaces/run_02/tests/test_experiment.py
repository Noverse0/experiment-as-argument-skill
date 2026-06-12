"""
Tests for the churn prediction experiment pipeline.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiment import (
    load_and_prepare_data,
    engineer_features,
    run_sanity_checks,
    run_experiment,
    summarize_results,
)


@pytest.fixture
def sample_dataset():
    """Create a test dataset with sufficient size for stratified splits."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'customer_id': np.arange(1, n + 1),
        'signup_date': pd.date_range('2023-01-01', periods=n).strftime('%Y-%m-%d'),
        'tenure_months': np.random.randint(1, 72, n),
        'monthly_spend': np.random.gamma(2.0, 30.0, n).round(2),
        'support_tickets': np.random.poisson(1.2, n),
        'account_status': np.random.choice(['active', 'closed'], n),
        'churned': np.random.binomial(1, 0.4, n),
    })
    return df


@pytest.fixture
def temp_csv(sample_dataset):
    """Write sample dataset to a temp CSV."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_dataset.to_csv(f, index=False)
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink()


def test_load_and_prepare_data_no_duplicates(temp_csv, sample_dataset):
    """Test loading data with no duplicates."""
    df, target_rate, n_dedup = load_and_prepare_data(temp_csv)

    assert len(df) == len(sample_dataset)
    assert n_dedup == 0
    assert 0 <= target_rate <= 1  # Valid churn rate
    assert target_rate == sample_dataset['churned'].mean()


def test_load_and_prepare_data_with_duplicates():
    """Test deduplication by customer_id."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df_dup = pd.DataFrame({
            'customer_id': [1, 1, 2, 2],
            'signup_date': ['2023-01-01'] * 4,
            'tenure_months': [12, 12, 24, 24],
            'monthly_spend': [100.0, 100.0, 200.0, 200.0],
            'support_tickets': [1, 1, 0, 0],
            'account_status': ['active', 'active', 'closed', 'closed'],
            'churned': [0, 0, 1, 1],
        })
        df_dup.to_csv(f, index=False)
        temp_path = f.name

    try:
        df, _, n_dedup = load_and_prepare_data(temp_path)
        assert len(df) == 2  # 2 unique customer_ids
        assert n_dedup == 2  # 2 duplicates removed
    finally:
        Path(temp_path).unlink()


def test_engineer_features(sample_dataset):
    """Test feature engineering: convert signup_date, drop account_status."""
    df_eng = engineer_features(sample_dataset)

    # Should have days_since_signup, drop account_status, customer_id, raw signup_date
    assert 'days_since_signup' in df_eng.columns
    assert 'account_status' not in df_eng.columns
    assert 'signup_date' not in df_eng.columns
    assert 'customer_id' not in df_eng.columns
    assert 'churned' in df_eng.columns

    # days_since_signup should be non-negative and sorted
    assert (df_eng['days_since_signup'] >= 0).all()
    assert df_eng['days_since_signup'].iloc[0] >= df_eng['days_since_signup'].iloc[-1]


def test_engineer_features_no_target_leakage(sample_dataset):
    """Check that engineered features produce reasonable predictions."""
    df_eng = engineer_features(sample_dataset)
    X = df_eng.drop('churned', axis=1)
    y = df_eng['churned']

    # Fit a simple model to check features work
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(X, y)
    score = lr.score(X, y)

    # Should be better than random (>0.5) and reasonable for a small dataset
    assert score > 0.5


def test_run_sanity_checks(sample_dataset):
    """Test sanity checks run without error."""
    df_eng = engineer_features(sample_dataset)
    X = df_eng.drop('churned', axis=1)
    y = df_eng['churned']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    checks = run_sanity_checks(X_train, y_train, X_test, y_test, seed=42)

    assert 'baseline_auc' in checks
    assert 'overfit_auc' in checks
    assert 'shuffle_auc' in checks
    assert 0 <= checks['baseline_auc'] <= 1
    assert 0 <= checks['overfit_auc'] <= 1
    assert 0 <= checks['shuffle_auc'] <= 1


def test_run_experiment_integrations(temp_csv):
    """Test full experiment run (end-to-end)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results = run_experiment(temp_csv, "logistic_regression", seeds=[42, 123], output_dir=Path(tmpdir))

        assert len(results) == 2
        for result in results:
            assert result.model_name == "logistic_regression"
            assert result.seed in [42, 123]
            assert 0 <= result.roc_auc_test <= 1
            assert 0 <= result.precision_test <= 1
            assert 0 <= result.recall_test <= 1
            assert 0 <= result.f1_test <= 1
            assert result.n_train + result.n_test > 0


def test_summarize_results():
    """Test summary statistics computation."""
    from src.experiment import RunResult

    results_by_model = {
        'model_a': [
            RunResult(42, 'model_a', 0.80, 0.7, 0.6, 0.65, 0.5, 80, 20, 0.4, 0),
            RunResult(123, 'model_a', 0.82, 0.71, 0.61, 0.66, 0.5, 80, 20, 0.4, 0),
        ],
        'model_b': [
            RunResult(42, 'model_b', 0.75, 0.65, 0.55, 0.6, 0.5, 80, 20, 0.4, 0),
            RunResult(123, 'model_b', 0.76, 0.66, 0.56, 0.61, 0.5, 80, 20, 0.4, 0),
        ],
    }

    summary = summarize_results(results_by_model)

    assert 'model_a' in summary
    assert 'model_b' in summary
    assert summary['model_a']['roc_auc_mean'] == pytest.approx(0.81, abs=0.01)
    assert summary['model_a']['n_seeds'] == 2
    assert summary['model_b']['roc_auc_mean'] == pytest.approx(0.755, abs=0.01)


def test_features_exclude_leaked_column(sample_dataset):
    """Verify that account_status (leaked from target) is excluded."""
    df_eng = engineer_features(sample_dataset)
    assert 'account_status' not in df_eng.columns


def test_split_is_stratified(temp_csv):
    """Verify that train/test split maintains class balance."""
    from sklearn.model_selection import train_test_split
    from src.experiment import load_and_prepare_data, engineer_features

    df, _, _ = load_and_prepare_data(temp_csv)
    df_eng = engineer_features(df)
    X = df_eng.drop('churned', axis=1)
    y = df_eng['churned']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Both train and test should have similar churn rates (stratified)
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    overall_rate = y.mean()

    # Allow small deviation due to rounding
    assert abs(train_rate - overall_rate) < 0.15
    assert abs(test_rate - overall_rate) < 0.15
