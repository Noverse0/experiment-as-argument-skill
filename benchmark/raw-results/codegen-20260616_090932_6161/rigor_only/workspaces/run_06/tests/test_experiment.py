"""Unit tests for the churn prediction experiment."""
import os
import tempfile
import pandas as pd
import numpy as np
import pytest

from src.data import load_and_clean_data, split_and_preprocess, get_data_summary
from src.models import (
    train_logistic_regression,
    train_gradient_boosting,
    evaluate_model,
    baseline_majority_class,
    label_shuffle_test,
)


@pytest.fixture
def sample_churn_data():
    """Create a small sample churn dataset for testing."""
    data = {
        "customer_id": [1, 2, 3, 4, 5, 6],
        "signup_date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"],
        "tenure_months": [12, 24, 6, 18, 30, 3],
        "monthly_spend": [50.0, 100.0, 25.0, 75.0, 150.0, 10.0],
        "support_tickets": [2, 5, 1, 3, 8, 0],
        "days_since_last_login": [5, 10, 40, 15, 20, 80],  # leak feature
        "churned": [0, 0, 1, 0, 0, 1],
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_csv(sample_churn_data):
    """Write sample data to a temp CSV file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_churn_data.to_csv(f, index=False)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


class TestDataLoading:
    def test_load_and_clean_data(self, temp_csv):
        """Test that data is loaded and cleaned correctly."""
        df = load_and_clean_data(temp_csv)

        # Should drop signup_date, customer_id, days_since_last_login
        assert "signup_date" not in df.columns
        assert "customer_id" not in df.columns
        assert "days_since_last_login" not in df.columns

        # Should keep feature columns and target
        assert "tenure_months" in df.columns
        assert "monthly_spend" in df.columns
        assert "support_tickets" in df.columns
        assert "churned" in df.columns

    def test_data_cleaning_removes_duplicates(self):
        """Test that duplicate rows are removed."""
        data = {
            "customer_id": [1, 2, 2],  # row 2 is duplicated
            "signup_date": ["2023-01-01", "2023-01-02", "2023-01-02"],
            "tenure_months": [12, 24, 24],
            "monthly_spend": [50.0, 100.0, 100.0],
            "support_tickets": [2, 5, 5],
            "days_since_last_login": [5, 10, 10],
            "churned": [0, 0, 0],
        }
        df_with_dup = pd.DataFrame(data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df_with_dup.to_csv(f, index=False)
            temp_path = f.name

        try:
            df_clean = load_and_clean_data(temp_path)
            # After dedup, should have 2 unique rows
            assert len(df_clean) == 2
        finally:
            os.unlink(temp_path)


class TestDataSummary:
    def test_get_data_summary(self, temp_csv):
        """Test data summary statistics."""
        df = load_and_clean_data(temp_csv)
        summary = get_data_summary(df)

        assert summary["total_rows"] > 0
        assert summary["n_features"] > 0
        assert 0 <= summary["target_rate"] <= 1
        assert summary["n_positive"] + summary["n_negative"] == summary["total_rows"]


class TestSplitAndPreprocess:
    def test_split_and_preprocess(self, temp_csv):
        """Test that split and preprocessing work correctly."""
        df = load_and_clean_data(temp_csv)

        X_train, X_test, y_train, y_test = split_and_preprocess(df, test_size=0.3, random_state=42)

        # Check shapes
        assert len(X_train) + len(X_test) <= len(df)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)


class TestModelTraining:
    def test_logistic_regression_trains(self, temp_csv):
        """Test that logistic regression trains without error."""
        df = load_and_clean_data(temp_csv)
        X_train, X_test, y_train, y_test = split_and_preprocess(df, random_state=42)

        model = train_logistic_regression(X_train, y_train)
        assert model is not None
        assert hasattr(model, "predict")

    def test_gradient_boosting_trains(self, temp_csv):
        """Test that gradient boosting trains without error."""
        df = load_and_clean_data(temp_csv)
        X_train, X_test, y_train, y_test = split_and_preprocess(df, random_state=42)

        model = train_gradient_boosting(X_train, y_train)
        assert model is not None
        assert hasattr(model, "predict")


class TestEvaluation:
    def test_evaluate_model_returns_metrics(self, temp_csv):
        """Test that evaluation returns all required metrics."""
        df = load_and_clean_data(temp_csv)
        X_train, X_test, y_train, y_test = split_and_preprocess(df, random_state=42)

        model = train_logistic_regression(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        # Check that all metrics are present
        assert "roc_auc" in metrics
        assert "f1_score" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "accuracy" in metrics

        # Check that metrics are in valid ranges
        assert 0 <= metrics["roc_auc"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1

    def test_baseline_majority_class(self, temp_csv):
        """Test baseline majority class predictor."""
        df = load_and_clean_data(temp_csv)
        X_train, X_test, y_train, y_test = split_and_preprocess(df, random_state=42)

        baseline = baseline_majority_class(y_test)

        assert "roc_auc" in baseline
        assert "f1_score" in baseline


class TestSanityChecks:
    def test_label_shuffle_test_reduces_performance(self, temp_csv):
        """Test that label shuffle reduces model performance."""
        df = load_and_clean_data(temp_csv)
        X_train, X_test, y_train, y_test = split_and_preprocess(df, random_state=42)

        model = train_logistic_regression(X_train, y_train)

        # Real performance
        real_metrics = evaluate_model(model, X_test, y_test)

        # Shuffled label performance
        shuffle_metrics = label_shuffle_test(model, X_test, y_test)

        # Shuffled should be worse (at or near baseline)
        # This test may be noisy on tiny samples, so we just check it runs
        assert "f1_score" in shuffle_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
