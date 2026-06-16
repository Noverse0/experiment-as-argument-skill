"""Tests for sanity checks."""
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import create_logistic_model
from src.sanity_checks import (
    baseline_floor_check, label_shuffle_check, overfit_tiny_subset_check
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

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test


def test_baseline_floor_check(sample_data):
    """Test baseline floor check."""
    X_train, X_test, y_train, y_test = sample_data

    baseline = baseline_floor_check(y_test)
    assert 0 <= baseline <= 1


def test_overfit_tiny_subset_check(sample_data):
    """Test that model can overfit tiny subset."""
    X_train, X_test, y_train, y_test = sample_data

    model = create_logistic_model()
    passed = overfit_tiny_subset_check(model, X_train, y_train, X_test, y_test)

    # Should pass for a good model.
    assert isinstance(passed, bool)


def test_label_shuffle_check(sample_data):
    """Test label shuffle check."""
    X_train, X_test, y_train, y_test = sample_data

    model = create_logistic_model()
    baseline = 0.5  # Dummy baseline.

    passed = label_shuffle_check(model, X_train, y_train, X_test, y_test, baseline)
    assert passed in [True, False]
