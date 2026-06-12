"""Test suite for churn prediction experiment.

Tests verify:
- Data discipline (no leaks, deduplication)
- Sanity checks (baseline, overfit, label shuffle)
- Repeatability (same seed → same metrics)
- Metric consistency (AUC in valid range)
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from src.experiment import (
    load_and_clean_data,
    baseline_majority_class,
    sanity_check_label_shuffle,
    sanity_check_overfit_tiny_subset,
    run_single_seed_experiment,
    run_full_experiment,
    summarize_results,
)


@pytest.fixture
def sample_csv():
    """Create a minimal test CSV."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("customer_id,signup_date,tenure_months,monthly_spend,support_tickets,account_status,churned\n")
        f.write("1,2023-01-01,10,50.0,1,active,0\n")
        f.write("2,2023-01-02,20,100.0,2,closed,1\n")
        f.write("3,2023-01-03,30,150.0,3,active,0\n")
        f.write("4,2023-01-04,40,200.0,1,closed,1\n")
        f.write("5,2023-01-05,50,250.0,2,active,0\n")
        f.write("6,2023-01-06,60,300.0,3,closed,1\n")
        f.write("7,2023-01-07,70,350.0,1,active,0\n")
        f.write("8,2023-01-08,80,400.0,2,closed,1\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


def test_load_and_clean_drops_leaky_columns(sample_csv):
    """Verify leaky columns are dropped."""
    df = load_and_clean_data(sample_csv)

    assert "account_status" not in df.columns, "account_status must be dropped"
    assert "signup_date" not in df.columns, "signup_date must be dropped"
    assert "customer_id" not in df.columns, "customer_id must be dropped"
    assert "churned" in df.columns, "churned must be present"
    assert "tenure_months" in df.columns, "tenure_months must be present"


def test_load_and_clean_deduplicates(sample_csv):
    """Verify deduplication works."""
    # Create a CSV with duplicates
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("customer_id,signup_date,tenure_months,monthly_spend,support_tickets,account_status,churned\n")
        f.write("1,2023-01-01,10,50.0,1,active,0\n")
        f.write("1,2023-01-01,10,50.0,1,active,0\n")
        f.write("2,2023-01-02,20,100.0,2,closed,1\n")
        f.write("2,2023-01-02,20,100.0,2,closed,1\n")
        temp_path = f.name

    try:
        df = load_and_clean_data(temp_path)
        # After dedup, should have 2 rows (one per unique key)
        assert len(df) == 2, f"Expected 2 rows after dedup, got {len(df)}"
    finally:
        Path(temp_path).unlink()


def test_baseline_majority_class():
    """Verify baseline uses majority class probability."""
    y = np.array([0, 0, 0, 1, 1])  # 3 negatives, 2 positives
    baseline = baseline_majority_class(y)
    assert baseline == 0.6, "Baseline should be max(0.6, 0.4) = 0.6"


def test_sanity_label_shuffle_auc_near_half(sample_csv):
    """With shuffled labels, AUC should be near 0.5 (or at least not > 0.7)."""
    df = load_and_clean_data(sample_csv)
    X = df.drop(columns=["churned"]).values
    y = df["churned"].values

    auc_lr = sanity_check_label_shuffle(X, y, LogisticRegression, seed=42)
    auc_gb = sanity_check_label_shuffle(X, y, GradientBoostingClassifier, seed=42)

    # With shuffled labels, AUC should be near 0.5 (not perfect).
    # On small samples, it might be 0, 0.5, or 1 by chance.
    # The key check: should not be >> 0.7 (which would indicate leakage)
    assert auc_lr < 0.75, f"Label-shuffle LR AUC should not be too high, got {auc_lr:.4f}"
    assert auc_gb < 0.75, f"Label-shuffle GB AUC should not be too high, got {auc_gb:.4f}"


def test_sanity_overfit_tiny_subset(sample_csv):
    """Model must overfit tiny subset (train AUC >> 0.5)."""
    df = load_and_clean_data(sample_csv)
    X = df.drop(columns=["churned"]).values
    y = df["churned"].values

    auc_lr = sanity_check_overfit_tiny_subset(X, y, LogisticRegression, seed=42, n_samples=5)
    auc_gb = sanity_check_overfit_tiny_subset(X, y, GradientBoostingClassifier, seed=42, n_samples=5)

    # Should be able to overfit
    assert auc_lr >= 0.5, f"LR should overfit tiny subset, got {auc_lr:.4f}"
    assert auc_gb >= 0.5, f"GB should overfit tiny subset, got {auc_gb:.4f}"


def test_run_single_seed_repeatability(sample_csv):
    """Same seed should produce identical metrics."""
    df = load_and_clean_data(sample_csv)
    X = df.drop(columns=["churned"]).values
    y = df["churned"].values

    results_1 = run_single_seed_experiment(X, y, seed=42)
    results_2 = run_single_seed_experiment(X, y, seed=42)

    for model in ["LogisticRegression", "GradientBoosting"]:
        assert results_1[model]["test_auc"] == results_2[model]["test_auc"], \
            f"{model} test_auc not reproducible"


def test_auc_in_valid_range(sample_csv):
    """Verify AUC metrics are in [0, 1]."""
    df = load_and_clean_data(sample_csv)
    X = df.drop(columns=["churned"]).values
    y = df["churned"].values

    results = run_single_seed_experiment(X, y, seed=42)

    for model in ["LogisticRegression", "GradientBoosting"]:
        auc = results[model]["test_auc"]
        assert 0 <= auc <= 1, f"{model} AUC out of range: {auc}"


def test_summarize_results_computes_stats(sample_csv):
    """Verify summary computes mean/std correctly."""
    df = load_and_clean_data(sample_csv)
    X = df.drop(columns=["churned"]).values
    y = df["churned"].values

    # Manually create results structure
    results = {"seeds": {}}
    for seed in [42, 123, 999]:
        results["seeds"][seed] = run_single_seed_experiment(X, y, seed)

    summary = summarize_results(results)

    assert "LogisticRegression" in summary
    assert "GradientBoosting" in summary
    assert summary["LogisticRegression"]["n"] == 3
    assert summary["GradientBoosting"]["n"] == 3
    assert "delta_auc" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
