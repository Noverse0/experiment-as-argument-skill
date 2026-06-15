"""Tests for experiment logic."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from src.experiment import compute_baseline, evaluate_model, summarize_results, compute_effect_size, run_experiment


@pytest.fixture
def sample_classification_data():
    """Create sample classification data."""
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    X_test = np.random.randn(30, 5)
    y_train = np.random.binomial(1, 0.5, 100)
    y_test = np.random.binomial(1, 0.5, 30)
    return X_train, X_test, y_train, y_test


def test_compute_baseline(sample_classification_data):
    """Test baseline computation."""
    X_train, X_test, y_train, y_test = sample_classification_data

    baseline = compute_baseline(y_test)

    assert "roc_auc" in baseline
    assert "f1" in baseline
    assert "precision" in baseline
    assert "recall" in baseline
    assert all(0 <= v <= 1 for v in baseline.values()), "Metrics should be in [0, 1]"


def test_evaluate_model(sample_classification_data):
    """Test single model evaluation."""
    X_train, X_test, y_train, y_test = sample_classification_data

    model = LogisticRegression(random_state=42, max_iter=1000)
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

    assert "roc_auc" in metrics
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    # Model should beat random guessing on average
    assert metrics["roc_auc"] > 0.4, f"ROC-AUC should be > 0.4, got {metrics['roc_auc']}"


def test_run_experiment(sample_classification_data):
    """Test full experiment run."""
    X_train, X_test, y_train, y_test = sample_classification_data

    # Combine train and test for the experiment
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    results = run_experiment(X, y, n_seeds=2, feature_set="clean")

    assert results["feature_set"] == "clean"
    assert results["baseline"] is not None
    assert len(results["logistic_regression"]) == 2
    assert len(results["gradient_boosting"]) == 2


def test_summarize_results(sample_classification_data):
    """Test results summarization."""
    X_train, X_test, y_train, y_test = sample_classification_data

    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    results = run_experiment(X, y, n_seeds=3, feature_set="clean")
    summary = summarize_results(results)

    assert "baseline" in summary
    assert "logistic_regression" in summary
    assert "gradient_boosting" in summary

    # Check structure of summary
    for model in ["logistic_regression", "gradient_boosting"]:
        for metric in ["roc_auc", "f1", "precision", "recall"]:
            assert metric in summary[model]
            assert "mean" in summary[model][metric]
            assert "std" in summary[model][metric]
            assert "n" in summary[model][metric]
            assert summary[model][metric]["n"] == 3


def test_effect_size_computation(sample_classification_data):
    """Test effect size computation."""
    X_train, X_test, y_train, y_test = sample_classification_data

    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    results = run_experiment(X, y, n_seeds=3, feature_set="clean")
    summary = summarize_results(results)
    effects = compute_effect_size(summary)

    assert "roc_auc" in effects
    assert "f1" in effects
    for metric in ["roc_auc", "f1", "precision", "recall"]:
        assert "difference" in effects[metric]
        assert "effect_size_cohens_d" in effects[metric]
        assert "gb_mean" in effects[metric]
        assert "lr_mean" in effects[metric]


def test_model_consistency_same_seed(sample_classification_data):
    """Test that same model and seed produces consistent results."""
    X_train, X_test, y_train, y_test = sample_classification_data

    model1 = LogisticRegression(random_state=42, max_iter=1000)
    metrics1 = evaluate_model(model1, X_train, X_test, y_train, y_test)

    model2 = LogisticRegression(random_state=42, max_iter=1000)
    metrics2 = evaluate_model(model2, X_train, X_test, y_train, y_test)

    # Same seed should give same results (within floating point precision)
    for key in metrics1:
        assert np.isclose(metrics1[key], metrics2[key]), f"Metrics differ for {key}"
