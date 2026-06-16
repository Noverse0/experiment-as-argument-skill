import pytest
import pandas as pd
import tempfile
from pathlib import Path
from src.experiment import ChurnExperiment


@pytest.fixture
def temp_csv(tmp_path):
    """Create a temporary churn CSV for testing."""
    import numpy as np
    np.random.seed(42)
    n = 100
    # Create varied features
    tenure = np.random.randint(1, 60, n)
    spend = np.random.uniform(10, 200, n)
    tickets = np.random.randint(0, 5, n)
    # Create churned with some correlation to features
    churned = (spend < 50).astype(int) | (tenure < 12).astype(int)

    data = {
        "customer_id": range(1, n + 1),
        "signup_date": ["2023-01-01"] * n,
        "tenure_months": tenure.tolist(),
        "monthly_spend": spend.tolist(),
        "support_tickets": tickets.tolist(),
        "days_since_last_login": np.random.randint(1, 100, n).tolist(),
        "churned": churned.tolist(),
    }
    df = pd.DataFrame(data)
    csv_file = tmp_path / "test_churn.csv"
    df.to_csv(csv_file, index=False)
    return str(csv_file)


def test_experiment_initialization(temp_csv):
    """Test experiment initializes correctly."""
    exp = ChurnExperiment(temp_csv, seeds=2, test_size=0.3)
    assert exp.csv_path == temp_csv
    assert exp.seeds == 2
    assert exp.test_size == 0.3


def test_experiment_loads_data(temp_csv):
    """Test data loading in experiment."""
    exp = ChurnExperiment(temp_csv, seeds=2)
    exp.load_and_validate()
    assert exp.data is not None
    assert exp.data.shape[0] == 100
    assert exp.baseline_rate > 0


def test_single_seed_run(temp_csv):
    """Test that a single seed produces valid metrics."""
    exp = ChurnExperiment(temp_csv, seeds=2)
    exp.load_and_validate()

    metrics = exp.run_single_seed("LogisticRegression", seed=42)
    assert "roc_auc" in metrics
    assert "recall" in metrics
    assert "precision" in metrics
    assert "f1" in metrics
    assert "balanced_accuracy" in metrics
    assert "accuracy" in metrics

    # All metrics should be between 0 and 1
    for metric, value in metrics.items():
        assert 0 <= value <= 1, f"{metric} out of bounds: {value}"


def test_both_models_run(temp_csv):
    """Test that both models can be instantiated and run."""
    exp = ChurnExperiment(temp_csv, seeds=1)
    exp.load_and_validate()

    for model_name in ["LogisticRegression", "GradientBoostingClassifier"]:
        metrics = exp.run_single_seed(model_name, seed=42)
        assert metrics is not None
        assert len(metrics) == 6


def test_baseline_check_passes(temp_csv):
    """Test that baseline sanity check passes."""
    exp = ChurnExperiment(temp_csv, seeds=1)
    exp.load_and_validate()
    exp.sanity_check_baseline()  # Should not raise


def test_label_shuffle_check_passes(temp_csv):
    """Test that label-shuffle sanity check passes."""
    exp = ChurnExperiment(temp_csv, seeds=1)
    exp.load_and_validate()
    exp.sanity_check_label_shuffle()  # Should not raise


def test_full_experiment_runs(temp_csv):
    """Test that the full experiment runs without errors."""
    exp = ChurnExperiment(temp_csv, seeds=2, test_size=0.3)
    exp.run_experiment()
    results = exp.get_results()

    assert "LogisticRegression" in results
    assert "GradientBoostingClassifier" in results

    for model_name in results.keys():
        assert "roc_auc" in results[model_name]
        assert "mean" in results[model_name]["roc_auc"]
        assert "std" in results[model_name]["roc_auc"]


def test_results_aggregation(temp_csv):
    """Test that metrics are properly aggregated across seeds."""
    exp = ChurnExperiment(temp_csv, seeds=3)
    exp.load_and_validate()

    metrics_list = []
    for seed in range(3):
        metrics = exp.run_single_seed("LogisticRegression", seed)
        metrics_list.append(metrics)

    agg = exp._aggregate_metrics(metrics_list)
    assert "roc_auc" in agg
    assert "mean" in agg["roc_auc"]
    assert "std" in agg["roc_auc"]
    assert agg["roc_auc"]["std"] >= 0


def test_invalid_model_raises(temp_csv):
    """Test that invalid model name raises error."""
    exp = ChurnExperiment(temp_csv, seeds=1)
    exp.load_and_validate()
    with pytest.raises(ValueError, match="Unknown model"):
        exp.run_single_seed("NonexistentModel", seed=42)


def test_experiment_reproducibility(temp_csv):
    """Test that same seed produces same results."""
    exp1 = ChurnExperiment(temp_csv, seeds=1)
    exp1.load_and_validate()
    metrics1 = exp1.run_single_seed("LogisticRegression", seed=42)

    exp2 = ChurnExperiment(temp_csv, seeds=1)
    exp2.load_and_validate()
    metrics2 = exp2.run_single_seed("LogisticRegression", seed=42)

    # Should produce identical results
    for metric in metrics1.keys():
        assert abs(metrics1[metric] - metrics2[metric]) < 1e-6
