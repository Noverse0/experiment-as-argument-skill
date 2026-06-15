"""Tests for data loading, cleaning, and model construction."""
import numpy as np
import pandas as pd
import pytest
from src.pipeline import load_and_clean, make_models, FEATURE_COLS, TARGET_COL, DROP_COLS


def _write_sample_csv(tmp_path, n=80, n_dupes=5):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "signup_date": ["2023-01-01"] * n,
        "tenure_months": rng.integers(1, 72, n),
        "monthly_spend": rng.gamma(2, 30, n).round(2),
        "support_tickets": rng.integers(0, 5, n),
        "days_since_last_login": rng.integers(1, 100, n),
        "churned": rng.integers(0, 2, n),
    })
    dup = df.sample(n_dupes, random_state=42)
    df = pd.concat([df, dup], ignore_index=True)
    path = tmp_path / "test_churn.csv"
    df.to_csv(path, index=False)
    return path, n  # n is expected rows after dedup


class TestLoadAndClean:
    def test_removes_exact_duplicates(self, tmp_path):
        path, expected = _write_sample_csv(tmp_path, n=80, n_dupes=5)
        X, y, dropped = load_and_clean(str(path))
        assert len(X) == expected
        assert dropped == 5

    def test_drops_all_excluded_columns(self, tmp_path):
        path, _ = _write_sample_csv(tmp_path)
        X, y, _ = load_and_clean(str(path))
        for col in DROP_COLS:
            assert col not in X.columns, f"{col} should be excluded (leak/id/temporal)"

    def test_target_leak_column_absent(self, tmp_path):
        path, _ = _write_sample_csv(tmp_path)
        X, y, _ = load_and_clean(str(path))
        assert "days_since_last_login" not in X.columns

    def test_returns_only_expected_features(self, tmp_path):
        path, _ = _write_sample_csv(tmp_path)
        X, y, _ = load_and_clean(str(path))
        assert list(X.columns) == FEATURE_COLS

    def test_target_is_binary(self, tmp_path):
        path, _ = _write_sample_csv(tmp_path)
        X, y, _ = load_and_clean(str(path))
        assert set(y.unique()).issubset({0, 1})

    def test_x_and_y_aligned(self, tmp_path):
        path, _ = _write_sample_csv(tmp_path)
        X, y, _ = load_and_clean(str(path))
        assert len(X) == len(y)

    def test_no_missing_values_in_features(self, tmp_path):
        path, _ = _write_sample_csv(tmp_path)
        X, y, _ = load_and_clean(str(path))
        assert X.isnull().sum().sum() == 0


class TestMakeModels:
    @pytest.fixture
    def small_data(self, tmp_path):
        path, _ = _write_sample_csv(tmp_path, n=60, n_dupes=0)
        return load_and_clean(str(path))[:2]  # X, y only

    def test_returns_two_models(self):
        models = make_models()
        assert set(models.keys()) == {"LogisticRegression", "GradientBoosting"}

    def test_models_fit_and_predict(self, small_data):
        X, y = small_data
        for name, model in make_models().items():
            model.fit(X, y)
            preds = model.predict(X)
            assert len(preds) == len(y), f"{name}: wrong prediction length"
            assert set(preds).issubset({0, 1}), f"{name}: predictions outside {{0,1}}"

    def test_models_predict_proba(self, small_data):
        X, y = small_data
        for name, model in make_models().items():
            model.fit(X, y)
            proba = model.predict_proba(X)
            assert proba.shape == (len(y), 2), f"{name}: wrong proba shape"
            assert np.allclose(proba.sum(axis=1), 1.0), f"{name}: probas don't sum to 1"

    def test_models_are_independent_instances(self):
        m1 = make_models()
        m2 = make_models()
        assert m1["LogisticRegression"] is not m2["LogisticRegression"]
