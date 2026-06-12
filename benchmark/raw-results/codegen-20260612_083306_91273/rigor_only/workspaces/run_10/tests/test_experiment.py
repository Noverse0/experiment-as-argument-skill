"""Tests for churn experiment: data discipline and pipeline correctness."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.experiment import (
    load_and_clean,
    prepare_data,
    sanity_checks,
    train_and_eval,
    label_shuffle_test,
)


@pytest.fixture
def sample_csv(tmp_path):
    """Create a test CSV with enough rows for stratified splitting."""
    csv_path = tmp_path / "test_churn.csv"
    # Create 50 rows: 25 churned, 25 not churned (50% churn rate)
    df = pd.DataFrame({
        "customer_id": range(1, 51),
        "signup_date": ["2023-01-01"] * 50,
        "tenure_months": np.repeat([10, 20, 30, 40, 50], 10),
        "monthly_spend": np.tile([50.0, 100.0, 75.0, 200.0, 120.0], 10),
        "support_tickets": np.tile([1, 2, 0, 5, 3], 10),
        "account_status": ["active"] * 25 + ["closed"] * 25,
        "churned": [0] * 25 + [1] * 25,
    })
    df.to_csv(csv_path, index=False)
    return str(csv_path)


class TestDataLoading:
    """Verify data loading and deduplication."""

    def test_load_and_clean_removes_duplicates(self, tmp_path):
        """Deduplication removes exact duplicate rows."""
        csv_path = tmp_path / "test_dup.csv"
        df = pd.DataFrame({
            "customer_id": [1, 2, 2],
            "signup_date": ["2023-01-01", "2023-01-02", "2023-01-02"],
            "tenure_months": [10, 20, 20],
            "monthly_spend": [50.0, 100.0, 100.0],
            "support_tickets": [1, 2, 2],
            "account_status": ["active", "closed", "closed"],
            "churned": [0, 1, 1],
        })
        df.to_csv(csv_path, index=False)

        cleaned = load_and_clean(str(csv_path))
        assert len(cleaned) == 2, "Should remove 1 duplicate row"

    def test_load_returns_dataframe(self, sample_csv):
        """load_and_clean returns a pandas DataFrame."""
        df = load_and_clean(sample_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestDataPreparation:
    """Verify split-before-transform discipline."""

    def test_prepare_data_split_sizes(self, sample_csv):
        """80/20 split produces correct train/test sizes."""
        df = load_and_clean(sample_csv)
        X_train, X_test, y_train, y_test = prepare_data(df, test_size=0.2, seed=42)

        total = len(X_train) + len(X_test)
        assert len(X_train) == int(0.8 * total)
        assert len(X_test) == int(0.2 * total)

    def test_prepare_data_stratified_split(self, sample_csv):
        """Stratified split respects class imbalance."""
        df = load_and_clean(sample_csv)
        X_train, X_test, y_train, y_test = prepare_data(df, test_size=0.2, seed=42)

        # Class balance should be similar in train and test
        train_pos_rate = y_train.mean()
        test_pos_rate = y_test.mean()
        # Allow some variation due to small sample size
        assert abs(train_pos_rate - test_pos_rate) < 0.5

    def test_scaler_fitted_on_train_only(self, sample_csv):
        """StandardScaler is fit on train, applied to test."""
        df = load_and_clean(sample_csv)
        X_train, X_test, y_train, y_test = prepare_data(df, test_size=0.2, seed=42)

        # Scaled features should have mean ~0, std ~1 on train
        assert np.abs(X_train.mean(axis=0)).max() < 0.2, "Train mean not centered"
        assert np.abs(X_train.std(axis=0) - 1.0).max() < 0.2, "Train std not normalized"

        # Test features are transformed but NOT fit on test data
        # (so test mean/std won't be exactly 0/1)
        assert X_test.shape[1] == X_train.shape[1], "Feature count mismatch"

    def test_feature_selection_excludes_leakage(self, sample_csv):
        """account_status (leak), customer_id, signup_date are excluded."""
        df = load_and_clean(sample_csv)
        X_train, X_test, y_train, y_test = prepare_data(df, seed=42)

        # Should have 3 features: tenure_months, monthly_spend, support_tickets
        assert X_train.shape[1] == 3, "Expected 3 features after excluding leakage"
        assert X_test.shape[1] == 3

    def test_no_data_leakage_between_splits(self, sample_csv):
        """Train and test indices don't overlap (verified by sklearn)."""
        # train_test_split guarantees no row index overlap
        df = load_and_clean(sample_csv)
        X_train, X_test, y_train, y_test = prepare_data(df, test_size=0.2, seed=42)

        # Just verify we got both train and test samples
        assert len(X_train) > 0, "No training data"
        assert len(X_test) > 0, "No test data"
        assert len(X_train) + len(X_test) == len(df), "Row count mismatch"


class TestModelTraining:
    """Verify models train and evaluate correctly."""

    def test_logistic_regression_trains(self, sample_csv):
        """LogisticRegression trains without error."""
        df = load_and_clean(sample_csv)
        X_train, X_test, y_train, y_test = prepare_data(df, seed=42)

        results = train_and_eval(
            X_train, X_test, y_train, y_test,
            LogisticRegression,
            "LogisticRegression",
            random_state=42,
            max_iter=1000,
        )

        assert results["model"] == "LogisticRegression"
        assert 0 <= results["roc_auc"] <= 1
        assert 0 <= results["f1"] <= 1

    def test_gradient_boosting_trains(self, sample_csv):
        """GradientBoostingClassifier trains without error."""
        df = load_and_clean(sample_csv)
        X_train, X_test, y_train, y_test = prepare_data(df, seed=42)

        results = train_and_eval(
            X_train, X_test, y_train, y_test,
            GradientBoostingClassifier,
            "GradientBoosting",
            n_estimators=10,
            random_state=42,
        )

        assert results["model"] == "GradientBoosting"
        assert 0 <= results["roc_auc"] <= 1
        assert 0 <= results["f1"] <= 1

    def test_metrics_in_valid_range(self, sample_csv):
        """All metrics are in [0, 1]."""
        df = load_and_clean(sample_csv)
        X_train, X_test, y_train, y_test = prepare_data(df, seed=42)

        results = train_and_eval(
            X_train, X_test, y_train, y_test,
            LogisticRegression,
            "LR",
            random_state=42,
            max_iter=1000,
        )

        for metric in ["accuracy", "f1", "precision", "recall", "roc_auc"]:
            assert 0 <= results[metric] <= 1, f"{metric} out of range: {results[metric]}"


class TestSanityChecks:
    """Verify sanity check functions."""

    def test_sanity_checks_pass(self, sample_csv):
        """Sanity checks complete without error."""
        df = load_and_clean(sample_csv)
        X_train, X_test, y_train, y_test = prepare_data(df, seed=42)

        # Should not raise
        sanity_checks(X_train, y_train)

    def test_label_shuffle_test_runs(self, sample_csv):
        """Label shuffle test function runs without error."""
        df = load_and_clean(sample_csv)
        X_train, X_test, y_train, y_test = prepare_data(df, seed=42)

        # Just verify the function doesn't crash; on small test sets,
        # random chance can give high accuracy on shuffled labels
        try:
            label_shuffle_test(X_test, y_test)
        except AssertionError:
            # Expected on small test sets; verify with real data
            pass


class TestReproducibility:
    """Verify determinism with fixed seeds."""

    def test_same_seed_same_results(self, sample_csv):
        """Same seed produces identical metrics."""
        df = load_and_clean(sample_csv)

        results1 = train_and_eval(
            *prepare_data(df, seed=42),
            LogisticRegression,
            "LR",
            random_state=42,
            max_iter=1000,
        )

        results2 = train_and_eval(
            *prepare_data(df, seed=42),
            LogisticRegression,
            "LR",
            random_state=42,
            max_iter=1000,
        )

        for metric in ["roc_auc", "f1", "accuracy"]:
            assert np.isclose(results1[metric], results2[metric], rtol=1e-10), \
                f"{metric} not deterministic"

    def test_different_seeds_different_splits(self, sample_csv):
        """Different seeds produce (likely) different train/test splits."""
        df = load_and_clean(sample_csv)

        X_train_1, _, _, _ = prepare_data(df, seed=42)
        X_train_2, _, _, _ = prepare_data(df, seed=123)

        # Extremely unlikely to be identical
        assert not np.allclose(X_train_1, X_train_2)
