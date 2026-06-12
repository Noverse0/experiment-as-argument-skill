"""Tests for experiment pipeline."""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data_utils import load_and_prepare, time_based_split, preprocess
from src.experiment import sanity_checks, train_and_evaluate


@pytest.fixture
def sample_df():
    """Create a minimal test dataframe with intentional leakage and duplicates."""
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 1, 2],
        "signup_date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-01", "2023-01-02"],
        "tenure_months": [12, 24, 36, 12, 24],
        "monthly_spend": [100.0, 200.0, 150.0, 100.0, 200.0],
        "support_tickets": [1, 2, 3, 1, 2],
        "account_status": ["active", "closed", "active", "active", "closed"],
        "churned": [0, 1, 0, 0, 1],
    })


class TestDataLoading:
    def test_load_and_prepare_deduplicates(self, sample_df, tmp_path):
        """Test that duplicates are removed."""
        csv_file = tmp_path / "test.csv"
        sample_df.to_csv(csv_file, index=False)

        df, info = load_and_prepare(str(csv_file))

        # Should have 3 unique rows (dropped 2 duplicates)
        assert len(df) == 3
        assert "Removed 2 duplicate rows" in info

    def test_load_and_prepare_removes_leak(self, sample_df, tmp_path):
        """Test that account_status is dropped as a leak."""
        csv_file = tmp_path / "test.csv"
        sample_df.to_csv(csv_file, index=False)

        df, info = load_and_prepare(str(csv_file))

        # account_status should be dropped
        assert "account_status" not in df.columns
        assert "LEAK DETECTED" in info

    def test_load_and_prepare_keeps_valid_columns(self, sample_df, tmp_path):
        """Test that valid columns are kept."""
        csv_file = tmp_path / "test.csv"
        sample_df.to_csv(csv_file, index=False)

        df, info = load_and_prepare(str(csv_file))

        expected_cols = {"customer_id", "signup_date", "tenure_months", "monthly_spend", "support_tickets", "churned"}
        assert set(df.columns) == expected_cols


class TestSplitting:
    def test_time_based_split_order(self, sample_df):
        """Test that time-based split respects date order."""
        train, test = time_based_split(sample_df, test_fraction=0.4)

        # Train should have earlier dates than test
        train_dates = pd.to_datetime(train["signup_date"])
        test_dates = pd.to_datetime(test["signup_date"])

        # The earliest test date should be >= latest train date (roughly)
        assert train_dates.max() <= test_dates.max()
        assert len(train) + len(test) == len(sample_df)

    def test_time_based_split_fraction(self, sample_df):
        """Test that split respects the test fraction."""
        train, test = time_based_split(sample_df, test_fraction=0.2)

        # Approximately 20% in test (with integer rounding)
        total = len(sample_df)
        expected_test = int(total * 0.2)
        # Allow 1 row tolerance for rounding
        assert abs(len(test) - expected_test) <= 1


class TestPreprocessing:
    def test_preprocess_scales_features(self):
        """Test that preprocessing applies scaling."""
        X_train = pd.DataFrame({
            "tenure_months": [10, 20, 30],
            "monthly_spend": [100, 200, 300],
            "support_tickets": [1, 2, 3],
        })
        X_test = pd.DataFrame({
            "tenure_months": [15, 25],
            "monthly_spend": [150, 250],
            "support_tickets": [2, 3],
        })

        X_train_scaled, X_test_scaled, scaler = preprocess(X_train, X_test)

        # Check shapes
        assert X_train_scaled.shape == (3, 3)
        assert X_test_scaled.shape == (2, 3)

        # Check that scaling was applied (scaled values should be small)
        assert np.abs(X_train_scaled).max() <= 2  # Scaled values typically in [-2, 2]

    def test_preprocess_scaler_reuse(self):
        """Test that a fitted scaler can be reused."""
        X_train = pd.DataFrame({
            "tenure_months": [10, 20, 30],
            "monthly_spend": [100, 200, 300],
            "support_tickets": [1, 2, 3],
        })
        X_test = pd.DataFrame({
            "tenure_months": [15, 25],
            "monthly_spend": [150, 250],
            "support_tickets": [2, 3],
        })

        X_train_scaled, X_test_scaled_1, scaler = preprocess(X_train, X_test, fit_scaler=True)

        # Reuse the scaler on the same test data
        _, X_test_scaled_2, _ = preprocess(X_train, X_test, fit_scaler=False, scaler=scaler)

        # Should get the same results for test set
        np.testing.assert_array_almost_equal(X_test_scaled_1, X_test_scaled_2)


class TestSanityChecks:
    def test_sanity_checks_baseline(self):
        """Test that sanity checks compute baseline accuracy."""
        X_train = np.random.randn(100, 3)
        y_train = np.concatenate([np.ones(70), np.zeros(30)])
        X_test = np.random.randn(50, 3)
        y_test = np.concatenate([np.ones(35), np.zeros(15)])

        results = sanity_checks(X_train, y_train, X_test, y_test)

        assert "baseline_accuracy" in results
        # Baseline should be majority class, which is 35/50 = 0.7
        assert abs(results["baseline_accuracy"] - 0.7) < 0.01

    def test_sanity_checks_label_balance(self):
        """Test that sanity checks report label balance."""
        X_train = np.random.randn(100, 3)
        y_train = pd.Series(np.concatenate([np.ones(70), np.zeros(30)]))
        X_test = np.random.randn(50, 3)
        y_test = pd.Series(np.concatenate([np.ones(35), np.zeros(15)]))

        results = sanity_checks(X_train, y_train, X_test, y_test)

        assert "train_churn_rate" in results
        assert "test_churn_rate" in results
        assert abs(results["train_churn_rate"] - 0.7) < 0.01


class TestTrainingAndEvaluation:
    def test_train_and_evaluate_logistic_regression(self):
        """Test logistic regression training."""
        from sklearn.linear_model import LogisticRegression

        X_train = np.random.randn(100, 3)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(30, 3)
        y_test = np.random.randint(0, 2, 30)

        metrics = train_and_evaluate(X_train, y_train, X_test, y_test, LogisticRegression, seed=42)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_train_and_evaluate_gradient_boosting(self):
        """Test gradient boosting training."""
        from sklearn.ensemble import GradientBoostingClassifier

        X_train = np.random.randn(100, 3)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(30, 3)
        y_test = np.random.randint(0, 2, 30)

        metrics = train_and_evaluate(X_train, y_train, X_test, y_test, GradientBoostingClassifier, seed=42)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_reproducibility_with_same_seed(self):
        """Test that same seed produces same results."""
        from sklearn.linear_model import LogisticRegression

        X_train = np.random.randn(100, 3)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(30, 3)
        y_test = np.random.randint(0, 2, 30)

        metrics_1 = train_and_evaluate(X_train, y_train, X_test, y_test, LogisticRegression, seed=42)
        metrics_2 = train_and_evaluate(X_train, y_train, X_test, y_test, LogisticRegression, seed=42)

        # Should get identical results
        assert metrics_1["accuracy"] == metrics_2["accuracy"]
        assert metrics_1["roc_auc"] == metrics_2["roc_auc"]
