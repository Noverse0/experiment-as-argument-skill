"""Tests for data loading and cleaning."""
import pandas as pd
import pytest

from src.data import FEATURES, LEAK_FEATURES, TARGET, get_X_y, load_and_clean


def _make_csv(tmp_path, rows):
    df = pd.DataFrame(rows)
    path = tmp_path / "churn.csv"
    df.to_csv(path, index=False)
    return str(path)


MINIMAL_ROW = {
    "customer_id": 1,
    "signup_date": "2023-01-01",
    "tenure_months": 12,
    "monthly_spend": 50.0,
    "support_tickets": 0,
    "days_since_last_login": 5,
    "churned": 0,
}


@pytest.fixture
def clean_csv(tmp_path):
    rows = [
        {**MINIMAL_ROW, "customer_id": i, "signup_date": f"2023-0{i}-01", "churned": i % 2}
        for i in range(1, 6)
    ]
    return _make_csv(tmp_path, rows)


def test_deduplication_removes_exact_copies(tmp_path):
    row = MINIMAL_ROW.copy()
    rows = [row, row, {**row, "customer_id": 2, "signup_date": "2023-02-01"}]
    path = _make_csv(tmp_path, rows)

    df = load_and_clean(path)
    assert len(df) == 2, f"Expected 2 unique rows, got {len(df)}"


def test_sorted_ascending_by_signup_date(clean_csv):
    df = load_and_clean(clean_csv)
    diffs = df["signup_date"].diff().dropna()
    assert (diffs >= pd.Timedelta(0)).all(), "Not sorted ascending by signup_date"


def test_leak_features_excluded_from_X(clean_csv):
    df = load_and_clean(clean_csv)
    X, _ = get_X_y(df)
    for col in LEAK_FEATURES:
        assert col not in X.columns, f"Leak feature '{col}' present in X"


def test_all_model_features_present(clean_csv):
    df = load_and_clean(clean_csv)
    X, _ = get_X_y(df)
    for feat in FEATURES:
        assert feat in X.columns, f"Feature '{feat}' missing from X"


def test_target_is_binary(clean_csv):
    df = load_and_clean(clean_csv)
    _, y = get_X_y(df)
    assert set(y.unique()).issubset({0, 1}), f"Target values: {y.unique()}"


def test_no_missing_values_in_features(clean_csv):
    df = load_and_clean(clean_csv)
    X, _ = get_X_y(df)
    assert X.isnull().sum().sum() == 0, "Missing values found in features"
