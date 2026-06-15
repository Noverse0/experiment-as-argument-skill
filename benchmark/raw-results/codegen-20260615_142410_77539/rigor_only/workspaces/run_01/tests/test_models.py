"""Tests for model training and evaluation."""
import numpy as np
import pytest
from src.dataset import load_and_deduplicate, time_based_split, prepare_features
from src.models import build_logistic_regression, build_gradient_boosting, train_and_evaluate


@pytest.fixture
def train_test_data():
    """Fixture providing train/test split."""
    df = load_and_deduplicate("churn.csv")
    train, test = time_based_split(df, test_fraction=0.2)
    X_train, y_train = prepare_features(train)
    X_test, y_test = prepare_features(test)
    return X_train, y_train, X_test, y_test


def test_logistic_regression_builds(train_test_data):
    """Test LogisticRegression pipeline instantiates."""
    lr = build_logistic_regression()
    assert lr is not None
    assert hasattr(lr, 'fit')


def test_gradient_boosting_builds():
    """Test GradientBoosting instantiates."""
    gb = build_gradient_boosting(random_state=42)
    assert gb is not None
    assert hasattr(gb, 'fit')


def test_logistic_regression_trains_and_predicts(train_test_data):
    """Test LogisticRegression can be trained and make predictions."""
    X_train, y_train, X_test, y_test = train_test_data
    lr = build_logistic_regression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)

    assert y_pred.shape == (len(X_test),)
    assert y_proba.shape == (len(X_test), 2)
    assert set(np.unique(y_pred)) <= {0, 1}


def test_gradient_boosting_trains_and_predicts(train_test_data):
    """Test GradientBoosting can be trained and make predictions."""
    X_train, y_train, X_test, y_test = train_test_data
    gb = build_gradient_boosting(random_state=42)
    gb.fit(X_train, y_train)

    y_pred = gb.predict(X_test)
    y_proba = gb.predict_proba(X_test)

    assert y_pred.shape == (len(X_test),)
    assert y_proba.shape == (len(X_test), 2)
    assert set(np.unique(y_pred)) <= {0, 1}


def test_train_and_evaluate_returns_metrics(train_test_data):
    """Test evaluation returns all expected metrics."""
    X_train, y_train, X_test, y_test = train_test_data
    lr = build_logistic_regression()

    metrics = train_and_evaluate(lr, X_train, y_train, X_test, y_test)

    expected_keys = {'auc', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1'}
    assert set(metrics.keys()) == expected_keys

    # All should be floats
    for v in metrics.values():
        assert isinstance(v, (float, np.floating)), f"Expected float, got {type(v)}"


def test_both_models_beat_baseline(train_test_data):
    """Test both models have reasonable AUC (> 0.5 random guess)."""
    X_train, y_train, X_test, y_test = train_test_data

    lr = build_logistic_regression()
    lr_metrics = train_and_evaluate(lr, X_train, y_train, X_test, y_test)

    gb = build_gradient_boosting(random_state=42)
    gb_metrics = train_and_evaluate(gb, X_train, y_train, X_test, y_test)

    # AUC should be above random (0.5)
    assert lr_metrics['auc'] > 0.5, "LR should beat random"
    assert gb_metrics['auc'] > 0.5, "GB should beat random"


def test_auc_in_valid_range(train_test_data):
    """Test AUC scores are in [0, 1]."""
    X_train, y_train, X_test, y_test = train_test_data
    lr = build_logistic_regression()
    metrics = train_and_evaluate(lr, X_train, y_train, X_test, y_test)

    auc = metrics['auc']
    assert 0 <= auc <= 1, f"AUC should be in [0, 1], got {auc}"


def test_deterministic_seed(train_test_data):
    """Test that same seed produces same results."""
    X_train, y_train, X_test, y_test = train_test_data

    gb1 = build_gradient_boosting(random_state=42)
    m1 = train_and_evaluate(gb1, X_train, y_train, X_test, y_test)

    gb2 = build_gradient_boosting(random_state=42)
    m2 = train_and_evaluate(gb2, X_train, y_train, X_test, y_test)

    # Should be identical
    for key in m1:
        v1, v2 = m1[key], m2[key]
        if not np.isnan(v1) and not np.isnan(v2):
            assert np.isclose(v1, v2), f"{key}: {v1} != {v2}"
