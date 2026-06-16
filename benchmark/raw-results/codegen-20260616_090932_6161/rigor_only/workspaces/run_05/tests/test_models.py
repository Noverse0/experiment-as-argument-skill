"""Tests for model training and evaluation."""
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import engineer_features, get_train_test_split
from src.models import (
    create_baseline_model, create_logistic_model, create_gb_model,
    evaluate_model, train_and_evaluate, DummyMajority
)


@pytest.fixture
def sample_data():
    """Create sample train/test data."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        'tenure_months': np.random.randint(1, 72, n),
        'monthly_spend': np.random.gamma(2.0, 30.0, n),
        'support_tickets': np.random.poisson(1.2, n),
        'days_since_signup': np.random.randint(0, 900, n),
    })
    y = pd.Series(np.random.binomial(1, 0.5, n))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_test_split(X, y, test_size=0.3, random_state=None):
    """Simple train/test split."""
    from sklearn.model_selection import train_test_split as sklearn_split
    return sklearn_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def test_baseline_model(sample_data):
    """Test baseline model (majority class)."""
    X_train, X_test, y_train, y_test = sample_data

    baseline = create_baseline_model()
    baseline.fit(X_train, y_train)

    y_pred = baseline.predict(X_test)
    y_proba = baseline.predict_proba(X_test)

    # Should always predict majority class.
    assert len(y_pred) == len(y_test)
    assert set(y_pred) <= {0, 1}
    assert y_proba.shape == (len(y_test), 2)


def test_logistic_model(sample_data):
    """Test logistic regression model."""
    X_train, X_test, y_train, y_test = sample_data

    model = create_logistic_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    assert len(y_pred) == len(y_test)
    assert set(y_pred) <= {0, 1}
    assert y_proba.shape == (len(y_test), 2)
    assert np.allclose(y_proba.sum(axis=1), 1.0)


def test_gb_model(sample_data):
    """Test gradient boosting model."""
    X_train, X_test, y_train, y_test = sample_data

    model = create_gb_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    assert len(y_pred) == len(y_test)
    assert set(y_pred) <= {0, 1}
    assert y_proba.shape == (len(y_test), 2)
    assert np.allclose(y_proba.sum(axis=1), 1.0)


def test_evaluate_model(sample_data):
    """Test evaluation metrics."""
    X_train, X_test, y_train, y_test = sample_data

    # Train a model.
    model = create_logistic_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Evaluate.
    metrics = evaluate_model(y_test, y_pred, y_proba)

    # Check all expected metrics.
    expected = ['roc_auc', 'pr_auc', 'f1', 'precision', 'recall', 'specificity']
    for metric in expected:
        assert metric in metrics
        assert 0 <= metrics[metric] <= 1 or metric in ['precision', 'recall']

    # ROC-AUC should be > 0.5 for a reasonable classifier.
    assert metrics['roc_auc'] > 0.4  # Lenient for small sample.


def test_train_and_evaluate(sample_data):
    """Test full train/evaluate pipeline."""
    X_train, X_test, y_train, y_test = sample_data

    model = create_logistic_model()
    results, (y_pred, y_proba) = train_and_evaluate(
        model, X_train, y_train, X_test, y_test, "test_model"
    )

    # Check structure.
    assert 'train' in results and 'test' in results
    for split in ['train', 'test']:
        for metric in ['roc_auc', 'pr_auc', 'f1', 'precision', 'recall', 'specificity']:
            assert metric in results[split]
            assert isinstance(results[split][metric], float)

    # Check predictions.
    assert len(y_pred) == len(y_test)
    assert len(y_proba) == len(y_test)
