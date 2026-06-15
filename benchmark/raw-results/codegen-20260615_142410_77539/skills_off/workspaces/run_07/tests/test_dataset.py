"""Tests for dataset loading and preprocessing."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import load_and_prepare


def test_dataset_loading(tmp_path):
    """Test that dataset loads and deduplicates correctly."""
    # Create a minimal test CSV
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 1, 2],
        "signup_date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-01", "2023-01-02"],
        "tenure_months": [12, 24, 36, 12, 24],
        "monthly_spend": [100.0, 200.0, 300.0, 100.0, 200.0],
        "support_tickets": [1, 2, 3, 1, 2],
        "days_since_last_login": [5, 10, 15, 5, 10],
        "churned": [0, 1, 0, 0, 1],
    })
    df.to_csv(csv_file, index=False)

    X, y, signup_dates, metadata = load_and_prepare(str(csv_file))

    # Check deduplication
    assert metadata["n_rows_original"] == 5
    assert metadata["n_duplicates_removed"] == 2
    assert metadata["n_rows_clean"] == 3

    # Check features selected
    assert list(X.columns) == ["tenure_months", "monthly_spend", "support_tickets"]
    assert len(X) == 3
    assert len(y) == 3

    # Check target
    assert metadata["target"] == "churned"


def test_feature_selection(tmp_path):
    """Test that days_since_last_login and customer_id are excluded."""
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({
        "customer_id": [1, 2],
        "signup_date": ["2023-01-01", "2023-01-02"],
        "tenure_months": [12, 24],
        "monthly_spend": [100.0, 200.0],
        "support_tickets": [1, 2],
        "days_since_last_login": [5, 10],
        "churned": [0, 1],
    })
    df.to_csv(csv_file, index=False)

    X, y, _, metadata = load_and_prepare(str(csv_file))

    # Verify excluded features
    assert "days_since_last_login" in metadata["excluded_features"]
    assert "customer_id" in metadata["excluded_features"]
    assert "signup_date" in metadata["excluded_features"]
    assert "days_since_last_login" not in X.columns
    assert "customer_id" not in X.columns


def test_class_distribution(tmp_path):
    """Test that class distribution is captured."""
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "signup_date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
        "tenure_months": [12, 24, 36, 48],
        "monthly_spend": [100.0, 200.0, 300.0, 400.0],
        "support_tickets": [1, 2, 3, 4],
        "days_since_last_login": [5, 10, 15, 20],
        "churned": [0, 0, 1, 1],
    })
    df.to_csv(csv_file, index=False)

    _, y, _, metadata = load_and_prepare(str(csv_file))

    assert metadata["class_distribution"][0] == 2
    assert metadata["class_distribution"][1] == 2
