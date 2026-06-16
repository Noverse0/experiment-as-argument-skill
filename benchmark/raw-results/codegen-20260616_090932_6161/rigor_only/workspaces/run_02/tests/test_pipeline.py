"""Tests for the churn prediction pipeline."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import subprocess
import shutil

from src.pipeline import ChurnExperiment


@pytest.fixture
def sample_dataset(tmp_path):
    """Generate a small test dataset in a temporary directory."""
    # Copy make_dataset.py to temp dir
    make_dataset_src = Path("make_dataset.py")
    make_dataset_dst = tmp_path / "make_dataset.py"
    shutil.copy(make_dataset_src, make_dataset_dst)

    output_path = tmp_path / "test_churn.csv"

    subprocess.run([
        "python3", "make_dataset.py",
        "--seed", "99",
        "--out", "test_churn.csv"
    ], check=True, cwd=str(tmp_path))

    yield str(output_path)


def test_data_loading(sample_dataset):
    """Test dataset loads correctly."""
    exp = ChurnExperiment(seed=42)
    df = exp.load_data(sample_dataset)

    assert df is not None
    assert len(df) > 0
    assert "churned" in df.columns
    assert "days_since_last_login" in df.columns


def test_preprocessing_leakage_exclusion(sample_dataset):
    """Test that days_since_last_login (the leak) is excluded."""
    exp = ChurnExperiment(seed=42)
    df = exp.load_data(sample_dataset)
    df = exp.preprocess(df)

    # days_since_last_login should be excluded
    assert "days_since_last_login" not in df.columns
    # But other features should remain
    assert "tenure_months" in df.columns
    assert "monthly_spend" in df.columns
    assert "support_tickets" in df.columns
    assert "churned" in df.columns


def test_preprocessing_deduplication(sample_dataset):
    """Test that duplicates are removed."""
    exp = ChurnExperiment(seed=42)
    df = exp.load_data(sample_dataset)

    # Count exact duplicates in raw data (there are 200)
    n_before = len(df)
    df = exp.preprocess(df)
    n_after = len(df)

    # Should have removed some duplicates
    assert n_after < n_before
    assert n_after > 0


def test_feature_engineering(sample_dataset):
    """Test that signup_date is properly extracted."""
    exp = ChurnExperiment(seed=42)
    df = exp.load_data(sample_dataset)
    df = exp.preprocess(df)

    # signup_date should be dropped, but year/month extracted
    assert "signup_date" not in df.columns
    assert "signup_year" in df.columns
    assert "signup_month" in df.columns
    assert df["signup_year"].min() >= 2023
    assert df["signup_month"].min() >= 1
    assert df["signup_month"].max() <= 12


def test_split_reproducibility(sample_dataset):
    """Test that splits are deterministic with same seed."""
    exp1 = ChurnExperiment(seed=42)
    df = exp1.load_data(sample_dataset)
    df = exp1.preprocess(df)

    (X_train_a, y_train_a), (X_test_a, y_test_a) = exp1.split_data(df)

    exp2 = ChurnExperiment(seed=42)
    (X_train_b, y_train_b), (X_test_b, y_test_b) = exp2.split_data(df)

    # Should be identical with same seed
    assert len(X_train_a) == len(X_train_b)
    assert len(X_test_a) == len(X_test_b)
    assert y_train_a.equals(y_train_b)
    assert y_test_a.equals(y_test_b)


def test_split_stratification(sample_dataset):
    """Test that stratification preserves class balance."""
    exp = ChurnExperiment(seed=42)
    df = exp.load_data(sample_dataset)
    df = exp.preprocess(df)

    (X_train, y_train), (X_test, y_test) = exp.split_data(df)

    train_churn_rate = y_train.mean()
    test_churn_rate = y_test.mean()
    overall_churn_rate = (y_train.sum() + y_test.sum()) / (len(y_train) + len(y_test))

    # Class balance should be preserved roughly
    assert abs(train_churn_rate - overall_churn_rate) < 0.05
    assert abs(test_churn_rate - overall_churn_rate) < 0.05


def test_models_train_and_evaluate(sample_dataset):
    """Test that both models train without error."""
    exp = ChurnExperiment(seed=42)
    df = exp.load_data(sample_dataset)
    df = exp.preprocess(df)
    (X_train, y_train), (X_test, y_test) = exp.split_data(df)

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    # Test LogisticRegression
    lr_clf = LogisticRegression(random_state=42, max_iter=1000)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr_metrics = exp.train_and_evaluate(lr_clf, X_train_scaled, y_train, X_test_scaled, y_test, "LR")

    assert "roc_auc" in lr_metrics
    assert 0 <= lr_metrics["roc_auc"] <= 1
    assert "precision" in lr_metrics
    assert "recall" in lr_metrics
    assert "f1" in lr_metrics

    # Test GradientBoosting
    gb_clf = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=5)
    gb_metrics = exp.train_and_evaluate(gb_clf, X_train, y_train, X_test, y_test, "GB")

    assert "roc_auc" in gb_metrics
    assert 0 <= gb_metrics["roc_auc"] <= 1


def test_baseline_sanity_check(sample_dataset):
    """Test that models beat the baseline."""
    exp = ChurnExperiment(seed=42)
    df = exp.load_data(sample_dataset)
    df = exp.preprocess(df)
    (X_train, y_train), (X_test, y_test) = exp.split_data(df)

    baseline_metrics = exp.train_baseline(y_train, y_test)

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    # Train models
    lr_clf = LogisticRegression(random_state=42, max_iter=1000)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr_metrics = exp.train_and_evaluate(lr_clf, X_train_scaled, y_train, X_test_scaled, y_test, "LR")

    gb_clf = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=5)
    gb_metrics = exp.train_and_evaluate(gb_clf, X_train, y_train, X_test, y_test, "GB")

    # Both should beat baseline
    assert lr_metrics["roc_auc"] > baseline_metrics["roc_auc"], \
        f"LR ({lr_metrics['roc_auc']:.4f}) did not beat baseline ({baseline_metrics['roc_auc']:.4f})"
    assert gb_metrics["roc_auc"] > baseline_metrics["roc_auc"], \
        f"GB ({gb_metrics['roc_auc']:.4f}) did not beat baseline ({baseline_metrics['roc_auc']:.4f})"


def test_full_experiment_run(sample_dataset):
    """Integration test: run the full experiment."""
    exp = ChurnExperiment(seed=42)
    result = exp.run(sample_dataset)

    assert "seed" in result
    assert "n_samples" in result
    assert "n_train" in result
    assert "n_test" in result
    assert "churn_rate" in result
    assert "baseline" in result
    assert "logistic_regression" in result
    assert "gradient_boosting" in result

    # Verify structure
    assert result["logistic_regression"]["roc_auc"] > result["baseline"]["roc_auc"]
    assert result["gradient_boosting"]["roc_auc"] > result["baseline"]["roc_auc"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
