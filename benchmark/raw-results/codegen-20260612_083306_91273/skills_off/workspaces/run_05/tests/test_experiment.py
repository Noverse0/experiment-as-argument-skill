"""Tests for the churn prediction experiment."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.experiment import (
    check_duplicates,
    compute_metrics,
    load_and_preprocess_data,
    preprocess_features,
    run_single_seed,
    split_data,
)


@pytest.fixture
def sample_csv(tmp_path):
    """Create a small sample CSV for testing."""
    df = pd.DataFrame({
        'customer_id': range(1, 101),
        'signup_date': ['2023-01-01'] * 100,
        'tenure_months': np.random.randint(1, 72, 100),
        'monthly_spend': np.random.rand(100) * 100,
        'support_tickets': np.random.randint(0, 5, 100),
        'account_status': ['active'] * 50 + ['closed'] * 50,
        'churned': [0] * 50 + [1] * 50,
    })
    csv_path = tmp_path / 'test.csv'
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_load_and_preprocess(sample_csv):
    """Test data loading and column drops."""
    df = load_and_preprocess_data(sample_csv)

    # Check that leaked columns are dropped
    assert 'account_status' not in df.columns
    assert 'customer_id' not in df.columns
    assert 'signup_date' not in df.columns

    # Check that temporal feature is created
    assert 'days_since_signup' in df.columns

    # Check target is present
    assert 'churned' in df.columns

    # Check features
    expected_features = {'tenure_months', 'monthly_spend', 'support_tickets', 'days_since_signup', 'churned'}
    assert set(df.columns) == expected_features


def test_check_duplicates():
    """Test duplicate detection."""
    df = pd.DataFrame({
        'a': [1, 2, 3, 3],
        'b': [4, 5, 6, 6],
    })
    assert check_duplicates(df) == 1

    df_no_dup = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    assert check_duplicates(df_no_dup) == 0


def test_split_data_stratification(sample_csv):
    """Test that split maintains class balance."""
    df = load_and_preprocess_data(sample_csv)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, seed=42)

    # Check sizes
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    # Check stratification (class balance should be similar in both)
    train_ratio = y_train.sum() / len(y_train)
    test_ratio = y_test.sum() / len(y_test)
    total_ratio = df['churned'].sum() / len(df)

    # Within 5% of total ratio (loose tolerance for small sample)
    assert abs(train_ratio - total_ratio) < 0.1
    assert abs(test_ratio - total_ratio) < 0.1


def test_preprocess_features_fit_only_on_train(sample_csv):
    """Test that scaler is fit on train only."""
    df = load_and_preprocess_data(sample_csv)
    X_train, X_test, _, _ = split_data(df, seed=42)

    X_train_scaled, X_test_scaled = preprocess_features(X_train, X_test)

    # Check shapes
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape

    # Check that scaling happened (values should be different from original)
    # At least some values should have changed
    assert not np.allclose(X_train_scaled, X_train.values)

    # Check that scaling is different if we fit on test
    from sklearn.preprocessing import StandardScaler
    scaler_test = StandardScaler()
    X_test_alt = scaler_test.fit_transform(X_test)
    # Should be different (fit on different data)
    assert not np.allclose(X_test_scaled, X_test_alt)


def test_compute_metrics():
    """Test metric computation."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8])

    metrics = compute_metrics(y_true, y_pred, y_pred_proba)

    # Check all expected metrics are present
    expected_keys = {'accuracy', 'precision', 'recall', 'f1', 'auc_roc'}
    assert set(metrics.keys()) == expected_keys

    # Check ranges
    for key, value in metrics.items():
        assert 0 <= value <= 1, f"{key}={value} out of range"


def test_run_single_seed(sample_csv):
    """Test training a single model on one seed."""
    df = load_and_preprocess_data(sample_csv)
    X_train, X_test, y_train, y_test = split_data(df, seed=42)
    X_train, X_test = preprocess_features(X_train, X_test)

    result = run_single_seed(X_train, X_test, y_train, y_test, LogisticRegression, seed=42)

    # Check result structure
    assert result.seed == 42
    assert result.model_name == 'LogisticRegression'
    assert len(result.train_metrics) > 0
    assert len(result.test_metrics) > 0
    assert set(result.train_metrics.keys()) == {'accuracy', 'precision', 'recall', 'f1', 'auc_roc'}

    # Baseline: test accuracy should be > 0
    assert result.test_metrics['accuracy'] > 0


def test_gradient_boosting_trainable(sample_csv):
    """Test that GradientBoostingClassifier can be trained."""
    df = load_and_preprocess_data(sample_csv)
    X_train, X_test, y_train, y_test = split_data(df, seed=42)
    X_train, X_test = preprocess_features(X_train, X_test)

    result = run_single_seed(X_train, X_test, y_train, y_test, GradientBoostingClassifier, seed=42)

    assert result.model_name == 'GradientBoostingClassifier'
    assert result.test_metrics['accuracy'] > 0


def test_deterministic_with_same_seed(sample_csv):
    """Test that the same seed produces the same results."""
    df = load_and_preprocess_data(sample_csv)

    # Run 1
    X_train1, X_test1, y_train1, y_test1 = split_data(df.copy(), seed=99)
    X_train1, X_test1 = preprocess_features(X_train1, X_test1)
    result1 = run_single_seed(X_train1, X_test1, y_train1, y_test1, LogisticRegression, seed=99)

    # Run 2 (same seed)
    X_train2, X_test2, y_train2, y_test2 = split_data(df.copy(), seed=99)
    X_train2, X_test2 = preprocess_features(X_train2, X_test2)
    result2 = run_single_seed(X_train2, X_test2, y_train2, y_test2, LogisticRegression, seed=99)

    # Results should be identical
    assert result1.test_metrics['auc_roc'] == result2.test_metrics['auc_roc']
    assert result1.test_metrics['accuracy'] == result2.test_metrics['accuracy']
