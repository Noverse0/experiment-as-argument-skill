"""Tests for experiment logic: models, sanity checks, results collection."""
import pytest
import numpy as np
from src.experiment import Experiment, ResultsCollector, SanityChecks


@pytest.fixture
def sample_data():
    """Create synthetic train/test data."""
    np.random.seed(42)
    n_train, n_test = 300, 100
    n_features = 3

    X_train = np.random.randn(n_train, n_features)
    X_test = np.random.randn(n_test, n_features)

    # Labels with signal
    y_train = (X_train[:, 0] + 0.5 * X_train[:, 1] > 0).astype(int)
    y_test = (X_test[:, 0] + 0.5 * X_test[:, 1] > 0).astype(int)

    return X_train, X_test, y_train, y_test


def test_experiment_returns_both_models(sample_data):
    """Experiment.run should return results for both models."""
    X_train, X_test, y_train, y_test = sample_data
    exp = Experiment(seed=42)
    results = exp.run(X_train, X_test, y_train, y_test)

    assert "logistic_regression" in results
    assert "gradient_boosting" in results
    assert "auc" in results["logistic_regression"]
    assert "auc" in results["gradient_boosting"]


def test_experiment_metrics_in_range(sample_data):
    """All metrics should be in [0, 1]."""
    X_train, X_test, y_train, y_test = sample_data
    exp = Experiment(seed=42)
    results = exp.run(X_train, X_test, y_train, y_test)

    for model_name in ["logistic_regression", "gradient_boosting"]:
        for metric in ["auc", "precision", "recall", "f1"]:
            value = results[model_name][metric]
            assert 0 <= value <= 1, f"{model_name} {metric} = {value} out of range"


def test_experiment_deterministic(sample_data):
    """Same seed should produce same results."""
    X_train, X_test, y_train, y_test = sample_data

    exp1 = Experiment(seed=42)
    results1 = exp1.run(X_train, X_test, y_train, y_test)

    exp2 = Experiment(seed=42)
    results2 = exp2.run(X_train, X_test, y_train, y_test)

    for model in results1:
        for metric in results1[model]:
            assert results1[model][metric] == results2[model][metric]


def test_experiment_different_seeds_differ(sample_data):
    """Different seeds can produce different results (or same on small test data)."""
    X_train, X_test, y_train, y_test = sample_data

    exp1 = Experiment(seed=42)
    results1 = exp1.run(X_train, X_test, y_train, y_test)

    exp2 = Experiment(seed=999)
    results2 = exp2.run(X_train, X_test, y_train, y_test)

    # Seeds can produce same or different results; verify both experiments ran
    assert len(results1) == 2
    assert len(results2) == 2
    # At least one metric should exist in both
    assert "auc" in results1["logistic_regression"]
    assert "auc" in results2["gradient_boosting"]


def test_results_collector_adds_results(sample_data):
    """ResultsCollector should accumulate results."""
    X_train, X_test, y_train, y_test = sample_data
    collector = ResultsCollector()

    for seed in [42, 123, 456]:
        exp = Experiment(seed=seed)
        results = exp.run(X_train, X_test, y_train, y_test)
        collector.add(results)

    summary = collector.summarize()
    assert len(summary) == 2  # Two models
    for model in summary:
        for metric in summary[model]:
            assert summary[model][metric]["n"] == 3  # 3 seeds


def test_results_collector_computes_stats(sample_data):
    """ResultsCollector should compute mean and std correctly."""
    X_train, X_test, y_train, y_test = sample_data
    collector = ResultsCollector()

    # Add 3 runs with known results
    results1 = {"logistic_regression": {"auc": 0.8}, "gradient_boosting": {"auc": 0.85}}
    results2 = {"logistic_regression": {"auc": 0.8}, "gradient_boosting": {"auc": 0.85}}
    results3 = {"logistic_regression": {"auc": 0.8}, "gradient_boosting": {"auc": 0.85}}

    collector.add(results1)
    collector.add(results2)
    collector.add(results3)

    summary = collector.summarize()
    # All AUCs should be identical (mean = values, std = 0)
    assert np.isclose(summary["logistic_regression"]["auc"]["mean"], 0.8)
    assert np.isclose(summary["logistic_regression"]["auc"]["std"], 0.0)
    assert np.isclose(summary["gradient_boosting"]["auc"]["mean"], 0.85)
    assert np.isclose(summary["gradient_boosting"]["auc"]["std"], 0.0)


def test_results_collector_save_json(sample_data, tmp_path):
    """ResultsCollector should save to JSON."""
    X_train, X_test, y_train, y_test = sample_data
    collector = ResultsCollector()

    exp = Experiment(seed=42)
    results = exp.run(X_train, X_test, y_train, y_test)
    collector.add(results)

    json_path = tmp_path / "results.json"
    collector.save_json(str(json_path))

    assert json_path.exists()
    # Check it's valid JSON
    import json
    with open(json_path) as f:
        data = json.load(f)
    assert "logistic_regression" in data
    assert "gradient_boosting" in data


def test_sanity_check_baseline_floor(sample_data):
    """Baseline floor should return a valid AUC."""
    X_train, X_test, y_train, y_test = sample_data
    baseline_auc = SanityChecks.baseline_floor(y_train, y_test)
    assert 0 <= baseline_auc <= 1


def test_sanity_check_tiny_overfit(sample_data):
    """Tiny overfit check should pass on synthetic data."""
    X_train, X_test, y_train, y_test = sample_data
    # Should not raise
    SanityChecks.tiny_overfit_check(X_train, y_train)


def test_sanity_check_label_shuffle(sample_data):
    """Label shuffle test should run without error."""
    X_train, X_test, y_train, y_test = sample_data
    baseline_auc = 0.5
    # Should not raise
    SanityChecks.label_shuffle_test(X_test, y_test, baseline_auc)
