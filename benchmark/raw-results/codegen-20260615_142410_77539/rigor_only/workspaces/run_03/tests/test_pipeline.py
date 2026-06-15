"""Tests for churn experiment pipeline.

Covers data loading, preprocessing, split logic, and sanity checks.
"""
import pytest
import pandas as pd
import numpy as np
from src.preprocessing import (
    load_data,
    check_duplicates,
    time_based_split,
    engineer_features,
    prepare_split,
    get_baseline_prediction,
)
from src.experiment import ExperimentRunner


@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing."""
    np.random.seed(42)
    n = 200
    data = {
        'customer_id': np.arange(1, n + 1),
        'signup_date': pd.date_range('2023-01-01', periods=n, freq='D').strftime('%Y-%m-%d'),
        'tenure_months': np.random.randint(1, 72, n),
        'monthly_spend': np.random.gamma(2.0, 30.0, n).round(2),
        'support_tickets': np.random.poisson(1.2, n),
        'days_since_last_login': np.random.randint(0, 100, n),
        'churned': np.random.randint(0, 2, n),
    }
    return pd.DataFrame(data)


def test_load_data(tmp_path, sample_data):
    """Test data loading from CSV."""
    csv_file = tmp_path / "test.csv"
    sample_data.to_csv(csv_file, index=False)
    df = load_data(str(csv_file))
    assert len(df) == len(sample_data)
    assert list(df.columns) == list(sample_data.columns)


def test_check_duplicates_no_dups(sample_data):
    """Test duplicate detection when there are none."""
    n_dup = check_duplicates(sample_data)
    assert n_dup == 0


def test_check_duplicates_with_dups(sample_data):
    """Test duplicate detection with duplicates present."""
    sample_with_dups = pd.concat([sample_data, sample_data.iloc[:10]], ignore_index=True)
    n_dup = check_duplicates(sample_with_dups)
    assert n_dup == 10


def test_time_based_split(sample_data):
    """Test that split respects temporal order."""
    train, test = time_based_split(sample_data, train_ratio=0.8)

    # Check sizes
    assert len(train) + len(test) == len(sample_data)
    assert abs(len(train) / len(sample_data) - 0.8) < 0.01

    # Check temporal order: train should have earlier dates than test
    max_train_date = train['signup_date'].max()
    min_test_date = test['signup_date'].min()
    assert max_train_date <= min_test_date


def test_engineer_features_drops_target_leak(sample_data):
    """Test that days_since_last_login (target leak) is dropped."""
    X, scaler = engineer_features(sample_data)

    # Check that days_since_last_login is not in features
    assert 'days_since_last_login' not in X.columns

    # Check expected features are present
    expected_features = ['tenure_months', 'monthly_spend', 'support_tickets', 'days_since_signup']
    assert list(X.columns) == expected_features


def test_engineer_features_scaling(sample_data):
    """Test that features are scaled (mean ~0, std ~1)."""
    X, scaler = engineer_features(sample_data)

    # Check approximate scaling (relaxed tolerance due to finite sample variance)
    assert np.allclose(X.mean(), 0, atol=0.01)
    assert np.allclose(X.std(), 1, atol=0.01)


def test_engineer_features_scaler_fit_once(sample_data):
    """Test that scaler is fit on first dataset, applied to second."""
    train = sample_data.iloc[:100]
    test = sample_data.iloc[100:]

    X_train, scaler = engineer_features(train)
    X_test, scaler_same = engineer_features(test, fit_scaler=scaler)

    # Scaler should be the same object
    assert scaler is scaler_same

    # Test set should have different mean/std (fit on train, not test)
    assert not np.allclose(X_test.mean(), 0, atol=1e-5)


def test_prepare_split(sample_data):
    """Test full prepare_split pipeline."""
    train, test = time_based_split(sample_data)
    X_train, y_train, X_test, y_test, scaler = prepare_split(train, test)

    # Check shapes
    assert len(X_train) == len(train)
    assert len(X_test) == len(test)
    assert len(y_train) == len(train)
    assert len(y_test) == len(test)

    # Check y values are binary
    assert set(y_train) <= {0, 1}
    assert set(y_test) <= {0, 1}

    # Check features are scaled
    assert np.allclose(X_train.mean(), 0, atol=1e-5)


def test_baseline_prediction(sample_data):
    """Test baseline prediction generation."""
    y = sample_data['churned'].values
    baseline = get_baseline_prediction(y)

    # Baseline should be all same value (majority class)
    assert len(np.unique(baseline)) == 1

    # Should be 1.0 if churn rate > 0.5, else 0.0
    churn_rate = y.mean()
    expected = 1.0 if churn_rate > 0.5 else 0.0
    assert baseline[0] == expected


def test_experiment_runner_basic(sample_data):
    """Test ExperimentRunner initialization and basic methods."""
    train, test = time_based_split(sample_data)
    X_train, y_train, X_test, y_test, _ = prepare_split(train, test)

    runner = ExperimentRunner(X_train, y_train, X_test, y_test, seeds=[42])
    assert runner.X_train is X_train
    assert runner.y_train is y_train
    assert len(runner.seeds) == 1


def test_experiment_models_train_and_predict(sample_data):
    """Test that both models can train and make predictions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier

    train, test = time_based_split(sample_data)
    X_train, y_train, X_test, y_test, _ = prepare_split(train, test)

    runner = ExperimentRunner(X_train, y_train, X_test, y_test, seeds=[42])

    # LogisticRegression
    lr_result = runner.run_model(LogisticRegression, {'max_iter': 1000}, 42)
    assert 'roc_auc' in lr_result
    assert 'pr_auc' in lr_result
    assert 0 <= lr_result['roc_auc'] <= 1

    # GradientBoosting
    gb_result = runner.run_model(GradientBoostingClassifier, {'n_estimators': 10}, 42)
    assert 'roc_auc' in gb_result
    assert 'pr_auc' in gb_result
    assert 0 <= gb_result['roc_auc'] <= 1


def test_experiment_baseline_sanity_check(sample_data):
    """Test baseline sanity check."""
    train, test = time_based_split(sample_data)
    X_train, y_train, X_test, y_test, _ = prepare_split(train, test)

    runner = ExperimentRunner(X_train, y_train, X_test, y_test)
    baseline = get_baseline_prediction(y_test)
    baseline_auc = runner.sanity_check_baseline(baseline)

    # Baseline should produce AUC between 0 and 1
    assert 0 <= baseline_auc <= 1


def test_experiment_label_shuffle_detects_leakage(sample_data):
    """Test that label shuffle test works."""
    train, test = time_based_split(sample_data)
    X_train, y_train, X_test, y_test, _ = prepare_split(train, test)

    runner = ExperimentRunner(X_train, y_train, X_test, y_test, seeds=[42])

    from sklearn.linear_model import LogisticRegression
    auc_shuffled = runner.sanity_check_label_shuffle(LogisticRegression, {'max_iter': 1000}, 42)

    # Shuffled labels should give AUC close to 0.5 (random)
    assert 0.4 <= auc_shuffled <= 0.6, f"Expected ~0.5, got {auc_shuffled}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
