"""Tests for the churn experiment pipeline."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.preprocessing import load_and_clean, prepare_features, FitOnTrainScaler
from src.experiment import ChurnExperiment


@pytest.fixture
def sample_data():
    """Create a minimal sample dataset for testing."""
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "signup_date": [
                "2023-01-01",
                "2023-02-01",
                "2023-03-01",
                "2023-04-01",
                "2023-05-01",
            ],
            "tenure_months": [12, 24, 6, 18, 30],
            "monthly_spend": [50.0, 100.0, 25.0, 75.0, 150.0],
            "support_tickets": [1, 2, 0, 3, 1],
            "days_since_last_login": [5, 10, 3, 20, 2],
            "churned": [0, 1, 0, 1, 0],
        }
    )


class TestPreprocessing:
    """Tests for preprocessing utilities."""

    def test_load_and_clean_removes_duplicates(self, tmp_path):
        """Test that load_and_clean removes exact duplicate rows."""
        # Create a CSV with duplicates
        df = pd.DataFrame(
            {
                "customer_id": [1, 2, 1, 2],
                "tenure_months": [12, 24, 12, 24],
                "monthly_spend": [50.0, 100.0, 50.0, 100.0],
                "support_tickets": [1, 2, 1, 2],
                "days_since_last_login": [5, 10, 5, 10],
                "churned": [0, 1, 0, 1],
            }
        )
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        # Load and clean
        cleaned = load_and_clean(str(csv_path))

        # Should have removed 2 duplicates
        assert len(cleaned) == 2

    def test_prepare_features_returns_correct_columns(self, sample_data):
        """Test that prepare_features returns correct features and target."""
        X, y = prepare_features(sample_data)

        # Check features
        assert list(X.columns) == [
            "tenure_months",
            "monthly_spend",
            "support_tickets",
        ]
        assert X.shape[0] == 5
        assert X.shape[1] == 3

        # Check target
        assert list(y) == [0, 1, 0, 1, 0]

    def test_prepare_features_drops_sensitive_columns(self, sample_data):
        """Test that sensitive columns are dropped."""
        X, y = prepare_features(sample_data)

        # These should NOT be in the features
        assert "customer_id" not in X.columns
        assert "days_since_last_login" not in X.columns
        assert "signup_date" not in X.columns

    def test_fit_on_train_scaler_fit_then_transform(self, sample_data):
        """Test scaler: fit on train, transform on test."""
        X_train = sample_data[["tenure_months", "monthly_spend", "support_tickets"]].iloc[:3]
        X_test = sample_data[["tenure_months", "monthly_spend", "support_tickets"]].iloc[3:]

        scaler = FitOnTrainScaler()
        scaler.fit(X_train)

        # Transform should not raise error
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Check shapes
        assert X_train_scaled.shape == (3, 3)
        assert X_test_scaled.shape == (2, 3)

        # Scaled values should be different from original
        assert not np.allclose(X_train_scaled, X_train.values)

    def test_fit_on_train_scaler_requires_fit(self):
        """Test that scaler raises error if used before fit."""
        scaler = FitOnTrainScaler()
        X = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
                "c": [7, 8, 9],
            }
        )

        with pytest.raises(ValueError, match="must be fitted"):
            scaler.transform(X)

    def test_fit_transform_works(self, sample_data):
        """Test fit_transform convenience method."""
        X = sample_data[["tenure_months", "monthly_spend", "support_tickets"]]
        scaler = FitOnTrainScaler()
        X_scaled = scaler.fit_transform(X)

        assert X_scaled.shape == X.shape
        assert scaler.fitted


class TestExperiment:
    """Tests for experiment logic."""

    def test_experiment_initialization(self, tmp_path):
        """Test experiment initialization."""
        exp = ChurnExperiment(csv_path="dummy.csv", output_dir=str(tmp_path))
        assert exp.csv_path == "dummy.csv"
        assert exp.output_dir == tmp_path
        assert len(exp.results) == 0

    def test_sanity_check_baseline_returns_rate(self, sample_data):
        """Test baseline check returns majority class rate."""
        y = sample_data["churned"]
        exp = ChurnExperiment()
        rate = exp.sanity_check_baseline(y)

        assert isinstance(rate, float)
        assert 0 <= rate <= 1
        assert rate == 0.6  # 3 out of 5 are class 0

    def test_run_single_seed_returns_dict_with_metrics(
        self, sample_data, tmp_path
    ):
        """Test that run_single_seed returns all required metrics."""
        X, y = prepare_features(sample_data)
        exp = ChurnExperiment(output_dir=str(tmp_path))

        result = exp.run_single_seed(X, y, seed=42)

        # Check structure
        assert "seed" in result
        assert result["seed"] == 42

        # Check all metrics present
        for model_prefix in ["lr", "gb"]:
            assert f"{model_prefix}_f1" in result
            assert f"{model_prefix}_accuracy" in result
            assert f"{model_prefix}_precision" in result
            assert f"{model_prefix}_recall" in result
            assert f"{model_prefix}_auc" in result

        # Check ranges
        for key in result:
            if key != "seed":
                assert 0 <= result[key] <= 1, f"{key} out of range: {result[key]}"

    def test_run_stores_results(self, tmp_path):
        """Test that run() stores results from multiple seeds."""
        # Create a minimal test CSV
        df = pd.DataFrame(
            {
                "customer_id": range(1, 101),
                "signup_date": ["2023-01-01"] * 100,
                "tenure_months": np.random.randint(1, 72, 100),
                "monthly_spend": np.random.uniform(10, 200, 100),
                "support_tickets": np.random.randint(0, 5, 100),
                "days_since_last_login": np.random.randint(1, 100, 100),
                "churned": np.random.randint(0, 2, 100),
            }
        )
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        exp = ChurnExperiment(
            csv_path=str(csv_path), output_dir=str(tmp_path)
        )
        exp.run(seeds=[42])

        # Should have 1 result
        assert len(exp.results) == 1
        assert exp.results[0]["seed"] == 42

    def test_save_results_creates_files(self, tmp_path):
        """Test that save_results creates output files."""
        # Minimal results
        exp = ChurnExperiment(output_dir=str(tmp_path))
        exp.results = [
            {
                "seed": 42,
                "lr_f1": 0.7,
                "lr_accuracy": 0.75,
                "lr_precision": 0.72,
                "lr_recall": 0.68,
                "lr_auc": 0.78,
                "gb_f1": 0.75,
                "gb_accuracy": 0.8,
                "gb_precision": 0.77,
                "gb_recall": 0.73,
                "gb_auc": 0.82,
            }
        ]

        exp.save_results()

        # Check files exist
        assert (tmp_path / "metrics.json").exists()
        assert (tmp_path / "summary.json").exists()
        assert Path("REPORT.md").exists()

    def test_report_generated_with_conclusion(self, tmp_path):
        """Test that REPORT.md is generated with a conclusion."""
        exp = ChurnExperiment(output_dir=str(tmp_path))
        exp.results = [
            {
                "seed": 42,
                "lr_f1": 0.7,
                "lr_accuracy": 0.75,
                "lr_precision": 0.72,
                "lr_recall": 0.68,
                "lr_auc": 0.78,
                "gb_f1": 0.75,
                "gb_accuracy": 0.8,
                "gb_precision": 0.77,
                "gb_recall": 0.73,
                "gb_auc": 0.82,
            }
        ]

        exp.save_results()

        # Read the report
        with open("REPORT.md") as f:
            report = f.read()

        # Check key sections
        assert "## Claim" in report
        assert "## Methodology" in report
        assert "## Results" in report
        assert "## Conclusion" in report
        assert "## Limitations" in report
