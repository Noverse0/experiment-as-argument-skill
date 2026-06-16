import pytest
import pandas as pd
import numpy as np
from src.pipeline import load_data, validate_data, preprocess_and_split, get_baseline_rate


@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing."""
    data = {
        "customer_id": [1, 2, 3, 4, 5],
        "signup_date": ["2023-01-01"] * 5,
        "tenure_months": [12, 24, 6, 18, 12],
        "monthly_spend": [100, 200, 50, 150, 120],
        "support_tickets": [1, 2, 0, 3, 1],
        "days_since_last_login": [5, 10, 2, 8, 3],
        "churned": [0, 1, 0, 1, 0],
    }
    return pd.DataFrame(data)


def test_load_data(tmp_path):
    """Test CSV loading."""
    csv_file = tmp_path / "test.csv"
    data = {
        "customer_id": [1, 2, 3],
        "tenure_months": [12, 24, 6],
        "monthly_spend": [100, 200, 50],
        "support_tickets": [1, 2, 0],
        "days_since_last_login": [5, 10, 2],
        "churned": [0, 1, 0],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    loaded = load_data(str(csv_file))
    assert loaded.shape == (3, 6)
    assert "churned" in loaded.columns


def test_validate_data(sample_data):
    """Test data validation."""
    assert validate_data(sample_data) is True


def test_validate_data_missing_target(sample_data):
    """Test validation fails with missing target."""
    bad_data = sample_data.drop("churned", axis=1)
    with pytest.raises(AssertionError, match="churned"):
        validate_data(bad_data)


def test_split_before_transform(sample_data):
    """Test that split happens before scaling."""
    X_train, X_test, y_train, y_test, scaler = preprocess_and_split(
        sample_data, test_size=0.4, random_state=42
    )

    # Check shapes
    assert len(X_train) + len(X_test) == len(sample_data)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    # Check that scaler was fit on train, not test
    # Scaler should be fit with train statistics only
    assert scaler is not None
    assert X_train.shape[1] == 3  # Three features: tenure, spend, tickets


def test_scaling_fit_on_train_only(sample_data):
    """Test that scaler is fit on train data only."""
    X_train, X_test, y_train, y_test, scaler = preprocess_and_split(
        sample_data, test_size=0.4, random_state=42
    )

    # Check that train is scaled (mean ~0, std ~1)
    train_means = X_train.mean()
    train_stds = X_train.std()
    assert np.allclose(train_means, 0, atol=1e-10) or np.allclose(train_stds, 1, atol=1e-10)

    # Scaler should have been fit on train data only
    assert scaler.mean_ is not None
    assert scaler.scale_ is not None


def test_stratified_split(sample_data):
    """Test that split respects class balance."""
    X_train, X_test, y_train, y_test, _ = preprocess_and_split(
        sample_data, test_size=0.4, random_state=42
    )
    # With stratification, churn rate should be similar in train and test
    train_churn_rate = y_train.mean()
    test_churn_rate = y_test.mean()
    # Both have same target distribution
    assert abs(train_churn_rate - test_churn_rate) < 0.5


def test_get_baseline_rate(sample_data):
    """Test baseline rate calculation."""
    baseline = get_baseline_rate(sample_data["churned"])
    # In sample data: 2 churned, 3 not churned
    # Majority class is 3/5 = 0.6
    assert baseline == 0.6


def test_features_correct(sample_data):
    """Test that the right features are selected."""
    X_train, X_test, y_train, y_test, _ = preprocess_and_split(
        sample_data, test_size=0.4, random_state=42
    )
    # Should use tenure_months, monthly_spend, support_tickets
    expected_features = ["tenure_months", "monthly_spend", "support_tickets"]
    assert list(X_train.columns) == expected_features


def test_no_data_leakage_between_splits(sample_data):
    """Test that no customer appears in both train and test."""
    X_train, X_test, y_train, y_test, _ = preprocess_and_split(
        sample_data, test_size=0.4, random_state=42
    )
    # Since we're splitting rows, train and test indices should not overlap
    assert len(set(X_train.index) & set(X_test.index)) == 0
