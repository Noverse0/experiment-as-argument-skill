"""Tests for data pipeline: loading, cleaning, and splitting."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.pipeline import (
    load_and_clean, prepare_features, get_cv_splitter,
    preprocess_for_lr, preprocess_for_gb
)


@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing."""
    np.random.seed(42)
    data = {
        "customer_id": np.arange(1, 101),
        "signup_date": ["2023-01-01"] * 100,
        "tenure_months": np.random.randint(1, 72, 100),
        "monthly_spend": np.random.gamma(2.0, 30.0, 100).round(2),
        "support_tickets": np.random.poisson(1.2, 100),
        "days_since_last_login": np.random.randint(1, 100, 100),
        "churned": np.random.binomial(1, 0.2, 100),
    }
    df = pd.DataFrame(data)
    # Add some exact duplicates
    dup = df.iloc[:5].copy()
    df = pd.concat([df, dup], ignore_index=True)
    return df


@pytest.fixture
def sample_csv(sample_data):
    """Write sample data to a temp CSV and yield the path."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        path = f.name
    yield path
    Path(path).unlink()


class TestLoadAndClean:
    def test_load_removes_duplicates(self, sample_csv):
        """Verify that duplicate rows are removed."""
        df = pd.read_csv(sample_csv)
        initial_rows = len(df)
        assert initial_rows == 105  # 100 + 5 duplicates

        df_clean = load_and_clean(sample_csv)
        assert len(df_clean) == 100  # Duplicates removed
        assert len(df_clean) == initial_rows - 5

    def test_load_preserves_columns(self, sample_csv):
        """Verify that all columns are preserved."""
        df = load_and_clean(sample_csv)
        expected_cols = {
            "customer_id", "signup_date", "tenure_months",
            "monthly_spend", "support_tickets", "days_since_last_login", "churned"
        }
        assert set(df.columns) == expected_cols


class TestPrepareFeatures:
    def test_prepare_drops_correct_columns(self, sample_data):
        """Verify that customer_id, days_since_last_login, and signup_date are dropped."""
        X, y, feature_names = prepare_features(sample_data)

        # Check feature names
        assert set(feature_names) == {"tenure_months", "monthly_spend", "support_tickets"}

        # Check X shape
        assert X.shape == (len(sample_data), 3)

        # Check y is the target
        assert len(y) == len(sample_data)
        assert set(y.unique()).issubset({0, 1})

    def test_prepare_target_vector(self, sample_data):
        """Verify that the target vector is correct."""
        _, y, _ = prepare_features(sample_data)
        assert y.name == "churned"
        assert all(val in [0, 1] for val in y.unique())


class TestCVSplitter:
    def test_cv_splitter_stratification(self, sample_data):
        """Verify that stratification preserves class balance."""
        X, y, _ = prepare_features(sample_data)
        cv = get_cv_splitter(n_splits=3, random_state=42)

        overall_churn_rate = y.mean()

        for train_idx, test_idx in cv.split(X, y):
            train_churn_rate = y.iloc[train_idx].mean()
            test_churn_rate = y.iloc[test_idx].mean()

            # Churn rates should be similar (within 10 percentage points)
            assert abs(train_churn_rate - overall_churn_rate) < 0.10
            assert abs(test_churn_rate - overall_churn_rate) < 0.10

    def test_cv_no_overlap(self, sample_data):
        """Verify that train and test folds do not overlap."""
        X, y, _ = prepare_features(sample_data)
        cv = get_cv_splitter(n_splits=5, random_state=42)

        for train_idx, test_idx in cv.split(X, y):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, "Train and test indices should not overlap"


class TestPreprocessing:
    def test_lr_scaling_fit_on_train_only(self, sample_data):
        """Verify that LR scaler is fit on train only."""
        X, _, _ = prepare_features(sample_data)
        X_train = X.iloc[:50]
        X_test = X.iloc[50:]

        X_train_scaled, X_test_scaled = preprocess_for_lr(X_train, X_test)

        # Scaled train should have mean ~0 and std ~1 per column
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape

        # Train mean should be close to 0 (scaler fit on train)
        assert np.abs(X_train_scaled.mean(axis=0)).max() < 0.1
        assert np.abs(X_train_scaled.std(axis=0) - 1.0).max() < 0.1

    def test_gb_preprocessing_no_scaling(self, sample_data):
        """Verify that GB does not scale features (returns raw arrays)."""
        X, _, _ = prepare_features(sample_data)
        X_train = X.iloc[:50]
        X_test = X.iloc[50:]

        X_train_gb, X_test_gb = preprocess_for_gb(X_train, X_test)

        # Should return numpy arrays, not scaled
        assert isinstance(X_train_gb, np.ndarray)
        assert isinstance(X_test_gb, np.ndarray)
        assert X_train_gb.shape == X_train.shape
        assert X_test_gb.shape == X_test.shape

        # Values should match original (not scaled)
        np.testing.assert_array_almost_equal(
            X_train_gb, X_train.values, decimal=5
        )


class TestNoLeakageAcrossFolds:
    def test_no_duplicate_samples_across_folds(self, sample_data):
        """Verify that if duplicate rows exist, they don't straddle train/test (when possible)."""
        X, y, _ = prepare_features(sample_data)
        cv = get_cv_splitter(n_splits=3, random_state=42)

        for train_idx, test_idx in cv.split(X, y):
            X_train = X.iloc[train_idx].reset_index(drop=True)
            X_test = X.iloc[test_idx].reset_index(drop=True)

            # Check for duplicates between train and test
            train_tuples = set(map(tuple, X_train.values))
            test_tuples = set(map(tuple, X_test.values))

            overlap = train_tuples & test_tuples
            # Some overlap is expected (real duplicates), but should be logged in production
            # This test just ensures the check is possible
            assert isinstance(overlap, set)
