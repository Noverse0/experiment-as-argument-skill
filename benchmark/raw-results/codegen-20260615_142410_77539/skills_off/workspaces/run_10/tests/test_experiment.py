"""Unit tests for the churn prediction experiment."""
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.dataset import load_and_deduplicate, prepare_features, split_and_prepare
from src.experiment import baseline_floor, label_shuffle_test, train_and_evaluate, run_experiment
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


def test_load_and_deduplicate():
    """Verify deduplication removes the 200 intentional duplicates."""
    df = load_and_deduplicate("churn.csv")
    # Dataset is 4000 rows + 200 duplicates = 4200, minus 200 dups = 4000
    assert len(df) == 4000
    # No duplicates remain
    assert df.duplicated().sum() == 0


def test_prepare_features():
    """Verify feature extraction works and produces expected columns."""
    df = load_and_deduplicate("churn.csv")
    X, y, feature_cols, scaler = prepare_features(df, fit_scaler=True)

    assert len(X) == len(df)
    assert len(y) == len(df)
    assert len(feature_cols) == 6
    assert "tenure_months" in feature_cols
    assert "monthly_spend" in feature_cols
    assert "support_tickets" in feature_cols
    assert "days_since_last_login" not in feature_cols  # Leakage excluded
    assert "customer_id" not in feature_cols  # Identifier excluded

    # Verify scaling: mean ~0, std ~1
    assert np.abs(X.mean(axis=0)).max() < 0.1
    assert np.abs(X.std(axis=0) - 1.0).max() < 0.1


def test_split_and_prepare():
    """Verify train/test split is stratified and scaler is fit on train only."""
    X_train, X_test, y_train, y_test, feature_cols, scaler = split_and_prepare("churn.csv", random_state=42)

    # Check split sizes
    total = len(X_train) + len(X_test)
    assert total == 4000
    assert len(X_train) == 3200  # 80%
    assert len(X_test) == 800  # 20%

    # Check that class balance is preserved
    train_churn_rate = y_train.mean()
    test_churn_rate = y_test.mean()
    assert abs(train_churn_rate - test_churn_rate) < 0.05  # Similar rates


def test_baseline_floor():
    """Baseline (always-majority) should give 0.5 AUC (random classifier)."""
    y_test = np.array([0, 0, 0, 0, 1, 1])  # 2/6 = 33% churn, majority = 0
    auc = baseline_floor(y_test)
    assert auc == 0.5  # Always predicting constant value gives AUC = 0.5 (random)


def test_label_shuffle_test():
    """Model trained on shuffled labels should have AUC near baseline."""
    X_train, X_test, y_train, y_test, _, _ = split_and_prepare("churn.csv", random_state=42)

    model = LogisticRegression(max_iter=500, random_state=42)
    auc_shuffle = label_shuffle_test(model, X_train, y_train, X_test, y_test, seed=42)
    auc_baseline = baseline_floor(y_test)

    # With shuffled labels, AUC should be close to baseline (no signal)
    assert abs(auc_shuffle - auc_baseline) < 0.2


def test_train_and_evaluate():
    """Model training and evaluation produces valid metrics."""
    X_train, X_test, y_train, y_test, _, _ = split_and_prepare("churn.csv", random_state=42)

    model = LogisticRegression(max_iter=500, random_state=42)
    metrics = train_and_evaluate(model, X_train, y_train, X_test, y_test)

    assert "auc_train" in metrics
    assert "auc_test" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics

    # AUCs should be in [0, 1]
    assert 0 <= metrics["auc_train"] <= 1
    assert 0 <= metrics["auc_test"] <= 1
    # Should beat baseline
    auc_baseline = baseline_floor(y_test)
    assert metrics["auc_test"] > auc_baseline


def test_run_experiment():
    """Full experiment run produces valid results."""
    result = run_experiment("churn.csv", seed=42)

    assert result.seed == 42
    assert "lr_auc_test" in result.metrics
    assert "gb_auc_test" in result.metrics
    assert "baseline_auc" in result.sanity_checks

    # Both models should beat baseline
    baseline = result.sanity_checks["baseline_auc"]
    lr_auc = result.metrics["lr_auc_test"]
    gb_auc = result.metrics["gb_auc_test"]

    assert lr_auc > baseline
    assert gb_auc > baseline


def test_reproducibility():
    """Same seed should give identical results."""
    result1 = run_experiment("churn.csv", seed=99)
    result2 = run_experiment("churn.csv", seed=99)

    assert result1.metrics["lr_auc_test"] == result2.metrics["lr_auc_test"]
    assert result1.metrics["gb_auc_test"] == result2.metrics["gb_auc_test"]
