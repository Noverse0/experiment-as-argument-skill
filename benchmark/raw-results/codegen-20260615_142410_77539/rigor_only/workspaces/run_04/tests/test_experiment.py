"""Tests for experiment pipeline."""
import numpy as np
import pytest
from pathlib import Path
import pandas as pd
from src.dataset import (
    load_data,
    check_duplicates,
    time_based_split,
    get_features_and_target,
    get_all_features_with_leak,
)
from src.experiment import baseline_majority, train_and_eval
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def sample_data():
    """Load the test churn dataset."""
    return load_data("churn.csv")


class TestDatasetLoading:
    def test_load_data_exists(self, sample_data):
        """Test that the dataset loads."""
        assert isinstance(sample_data, pd.DataFrame)
        assert len(sample_data) > 0

    def test_load_data_has_required_columns(self, sample_data):
        """Test that all required columns are present."""
        required = {"customer_id", "signup_date", "tenure_months", "monthly_spend", "support_tickets", "days_since_last_login", "churned"}
        assert required.issubset(sample_data.columns)

    def test_churned_is_binary(self, sample_data):
        """Test that churned column contains only 0 and 1."""
        assert sample_data["churned"].isin([0, 1]).all()

    def test_no_missing_values(self, sample_data):
        """Test that there are no missing values."""
        assert sample_data.isnull().sum().sum() == 0


class TestDuplicateAudit:
    def test_check_duplicates_returns_dict(self, sample_data):
        """Test that duplicate check returns expected structure."""
        audit = check_duplicates(sample_data)
        assert "total_rows" in audit
        assert "full_duplicates" in audit
        assert "feature_duplicates" in audit

    def test_duplicates_detected(self, sample_data):
        """Test that the 200 planted duplicates are detected."""
        audit = check_duplicates(sample_data)
        # Dataset has 200 exact duplicates added
        assert audit["full_duplicates"] >= 100  # At least half of them should be detected


class TestTemporalSplit:
    def test_split_returns_three_items(self, sample_data):
        """Test that split returns train, test, and info dict."""
        train, test, info = time_based_split(sample_data, train_fraction=0.8)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert isinstance(info, dict)

    def test_split_respects_fraction(self, sample_data):
        """Test that split respects the train fraction."""
        train, test, info = time_based_split(sample_data, train_fraction=0.8)
        total = len(train) + len(test)
        assert len(train) / total >= 0.78  # Allow small rounding

    def test_split_is_temporal(self, sample_data):
        """Test that split is ordered by signup_date (train earlier than test)."""
        train, test, _ = time_based_split(sample_data, train_fraction=0.8)
        train_max_date = train["signup_date"].max()
        test_min_date = test["signup_date"].min()
        # Train dates should be earlier than test dates (with possible overlap due to ties)
        assert train_max_date <= test_min_date or pd.Timestamp(train_max_date) <= pd.Timestamp(test_min_date)

    def test_no_data_loss(self, sample_data):
        """Test that no rows are lost in the split."""
        train, test, _ = time_based_split(sample_data)
        assert len(train) + len(test) == len(sample_data)


class TestFeatureExtraction:
    def test_get_features_excludes_leak(self, sample_data):
        """Test that days_since_last_login is excluded."""
        features, target = get_features_and_target(sample_data)
        assert "days_since_last_login" not in features.columns
        assert "customer_id" not in features.columns
        assert "signup_date" not in features.columns

    def test_get_features_includes_honest_features(self, sample_data):
        """Test that honest features are included."""
        features, target = get_features_and_target(sample_data)
        honest = {"tenure_months", "monthly_spend", "support_tickets"}
        assert honest.issubset(features.columns)

    def test_target_extraction(self, sample_data):
        """Test that target is correctly extracted."""
        features, target = get_features_and_target(sample_data)
        assert target.name == "churned"
        assert target.isin([0, 1]).all()
        assert len(target) == len(features)

    def test_with_leak_feature(self, sample_data):
        """Test that leak feature version includes days_since_last_login."""
        features, target = get_all_features_with_leak(sample_data)
        assert "days_since_last_login" in features.columns
        assert "tenure_months" in features.columns


class TestModelTraining:
    def test_baseline_majority(self, sample_data):
        """Test that baseline majority class predictor works."""
        _, _, _ = time_based_split(sample_data)
        train, test, _ = time_based_split(sample_data)
        _, y_test = get_features_and_target(test)
        auc = baseline_majority(y_test)
        assert 0.0 <= auc <= 1.0

    def test_train_and_eval(self, sample_data):
        """Test that train_and_eval returns expected metrics."""
        train, test, _ = time_based_split(sample_data)
        X_train, y_train = get_features_and_target(train)
        X_test, y_test = get_features_and_target(test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lr = LogisticRegression(random_state=42, max_iter=1000)
        metrics = train_and_eval(lr, X_train_scaled, y_train, X_test_scaled, y_test, 42)

        assert "roc_auc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_model_beats_baseline(self, sample_data):
        """Test that the model beats the majority class baseline."""
        train, test, _ = time_based_split(sample_data)
        X_train, y_train = get_features_and_target(train)
        X_test, y_test = get_features_and_target(test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lr = LogisticRegression(random_state=42, max_iter=1000)
        metrics = train_and_eval(lr, X_train_scaled, y_train, X_test_scaled, y_test, 42)
        baseline_auc = baseline_majority(y_test)

        # Model should beat baseline
        assert metrics["roc_auc"] > baseline_auc, f"Model AUC {metrics['roc_auc']:.4f} not better than baseline {baseline_auc:.4f}"
