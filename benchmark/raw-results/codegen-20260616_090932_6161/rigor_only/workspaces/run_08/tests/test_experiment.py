"""Tests for experiment module."""
import pytest
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

from src.experiment import ChurnExperiment


@pytest.fixture
def experiment():
    """Create experiment instance."""
    data_file = Path(__file__).parent.parent / "churn.csv"
    return ChurnExperiment(str(data_file), n_seeds=2, test_month=10)


def test_experiment_initialization(experiment):
    """Test experiment init."""
    assert "churn.csv" in experiment.data_path
    assert experiment.n_seeds == 2
    assert experiment.test_month == 10
    assert "LogisticRegression" in experiment.results
    assert "GradientBoostingClassifier" in experiment.results


def test_sanity_check_label_shuffle(experiment):
    """Test that label shuffle gives AUC near 0.5."""
    shuffled_auc = experiment.sanity_check_label_shuffle()
    assert 0.4 < shuffled_auc < 0.6


def test_sanity_check_overfit_tiny(experiment):
    """Test that model overfits tiny subset."""
    train_loss = experiment.sanity_check_overfit_tiny()
    assert train_loss < 0.5


def test_run_seed(experiment):
    """Test running experiment for one seed."""
    experiment.run_seed(0)

    assert len(experiment.results["LogisticRegression"]) == 1
    assert len(experiment.results["GradientBoostingClassifier"]) == 1

    lr_metrics = experiment.results["LogisticRegression"][0]
    assert 0 <= lr_metrics.test_auc <= 1
    assert 0 <= lr_metrics.test_accuracy <= 1
    assert lr_metrics.seed == 0


def test_run_all_seeds(experiment):
    """Test running all seeds."""
    experiment.run_all_seeds()

    assert len(experiment.results["LogisticRegression"]) == 2
    assert len(experiment.results["GradientBoostingClassifier"]) == 2


def test_get_summary(experiment):
    """Test summary generation."""
    experiment.run_all_seeds()
    summary = experiment.get_summary()

    assert "LogisticRegression" in summary
    assert "GradientBoostingClassifier" in summary

    for model_name in ["LogisticRegression", "GradientBoostingClassifier"]:
        stats = summary[model_name]
        assert "test_auc_mean" in stats
        assert "test_auc_std" in stats
        assert "test_auc_values" in stats
        assert len(stats["test_auc_values"]) == 2
        assert stats["n_seeds"] == 2
        assert 0 <= stats["test_auc_mean"] <= 1
