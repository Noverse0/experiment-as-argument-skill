import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import (
    ID_COLS,
    LEAKY_COLS,
    TARGET_COL,
    load_and_clean,
    make_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n_base=120, seed=0, with_dups=True):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n_base)
    spend = rng.gamma(2, 30, n_base).round(2)
    tickets = rng.poisson(1.2, n_base)
    logit = -1.2 - 0.03 * tenure + 0.01 * spend + 0.45 * tickets
    churn = (rng.random(n_base) < 1 / (1 + np.exp(-logit))).astype(int)
    dates = pd.date_range("2023-01-01", periods=n_base, freq="3D")
    df = pd.DataFrame({
        "customer_id": np.arange(1, n_base + 1),
        "signup_date": dates.strftime("%Y-%m-%d"),
        "tenure_months": tenure,
        "monthly_spend": spend,
        "support_tickets": tickets,
        "account_status": np.where(churn == 1, "closed", "active"),
        "churned": churn,
    })
    if with_dups:
        dups = df.sample(n=10, random_state=seed)
        df = pd.concat([df, dups], ignore_index=True)
    return df


@pytest.fixture
def sample_csv(tmp_path):
    df = _make_df()
    path = tmp_path / "churn.csv"
    df.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadAndClean:
    def test_duplicates_removed(self, sample_csv):
        df, stats = load_and_clean(sample_csv)
        assert stats["n_duplicates_removed"] == 10
        assert stats["n_clean"] == 120
        assert stats["n_raw"] == 130

    def test_sorted_by_date(self, sample_csv):
        df, _ = load_and_clean(sample_csv)
        assert df["signup_date"].is_monotonic_increasing

    def test_no_duplicates_after_clean(self, sample_csv):
        df, _ = load_and_clean(sample_csv)
        assert df.duplicated().sum() == 0


class TestMakeFeatures:
    def test_leaky_columns_absent(self, sample_csv):
        df, _ = load_and_clean(sample_csv)
        _, _, feature_names = make_features(df)
        for col in LEAKY_COLS:
            assert col not in feature_names, f"Leaky column '{col}' must not appear in features"

    def test_id_and_target_absent(self, sample_csv):
        df, _ = load_and_clean(sample_csv)
        _, _, feature_names = make_features(df)
        assert TARGET_COL not in feature_names
        for col in ID_COLS:
            assert col not in feature_names

    def test_shape_matches_rows(self, sample_csv):
        df, stats = load_and_clean(sample_csv)
        X, y, feature_names = make_features(df)
        assert X.shape[0] == stats["n_clean"]
        assert X.shape[1] == len(feature_names)
        assert len(y) == stats["n_clean"]

    def test_dtype_is_float(self, sample_csv):
        df, _ = load_and_clean(sample_csv)
        X, y, _ = make_features(df)
        assert X.dtype == float

    def test_no_nan(self, sample_csv):
        df, _ = load_and_clean(sample_csv)
        X, y, _ = make_features(df)
        assert not np.isnan(X).any(), "Feature matrix must not contain NaN"
        assert not np.isnan(y.astype(float)).any()

    def test_target_is_binary(self, sample_csv):
        df, _ = load_and_clean(sample_csv)
        _, y, _ = make_features(df)
        assert set(y).issubset({0, 1})

    def test_days_since_ref_is_nonnegative(self, sample_csv):
        df, _ = load_and_clean(sample_csv)
        X, _, feature_names = make_features(df)
        idx = feature_names.index("days_since_ref")
        assert (X[:, idx] >= 0).all(), "days_since_ref must be non-negative"


class TestSanityChecks:
    """Model-level sanity checks on synthetic data with known signal."""

    @pytest.fixture
    def synthetic_data(self):
        rng = np.random.default_rng(42)
        n = 300
        X = rng.standard_normal((n, 4))
        # True DGP: logistic function of first two features
        p = 1 / (1 + np.exp(-(X[:, 0] * 1.5 - X[:, 1] * 0.8)))
        y = (rng.random(n) < p).astype(int)
        return X, y

    def test_label_shuffle_degrades_auc(self, synthetic_data):
        """Label-shuffled model must have lower ROC-AUC than one trained on real labels."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = synthetic_data
        X_tr, X_te = X[:200], X[200:]
        y_tr, y_te = y[:200], y[200:]

        model = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=500, random_state=0))])
        model.fit(X_tr, y_tr)
        real_auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])

        rng = np.random.default_rng(7)
        y_shuffled = rng.permutation(y_tr)
        model_sh = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=500, random_state=0))])
        model_sh.fit(X_tr, y_shuffled)
        shuffled_auc = roc_auc_score(y_te, model_sh.predict_proba(X_te)[:, 1])

        assert real_auc > shuffled_auc, (
            f"Real AUC {real_auc:.3f} should exceed shuffled AUC {shuffled_auc:.3f}")

    def test_models_exceed_random_baseline(self, synthetic_data):
        """Both LR and GBM should achieve ROC-AUC > 0.5 on data with known signal."""
        from src.experiment import make_models

        X, y = synthetic_data
        X_tr, X_te = X[:200], X[200:]
        y_tr, y_te = y[:200], y[200:]

        models = make_models()
        for name in ("logistic_regression", "gradient_boosting"):
            m = clone(models[name])
            m.fit(X_tr, y_tr)
            auc = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
            assert auc > 0.5, f"{name} ROC-AUC {auc:.3f} must exceed 0.5 on data with real signal"


class TestCVPipeline:
    def test_cv_returns_correct_keys(self, sample_csv):
        from src.experiment import run_cv, summarize

        df, _ = load_and_clean(sample_csv)
        X, y, _ = make_features(df)
        fold_results = run_cv(X, y, n_splits=3)
        summary = summarize(fold_results)

        for name in ("logistic_regression", "gradient_boosting", "majority_baseline"):
            assert name in summary

        for metric in ("roc_auc", "f1", "accuracy"):
            assert metric in summary["logistic_regression"]
            assert metric in summary["gradient_boosting"]

    def test_cv_fold_count(self, sample_csv):
        from src.experiment import run_cv

        df, _ = load_and_clean(sample_csv)
        X, y, _ = make_features(df)
        fold_results = run_cv(X, y, n_splits=3)
        for name, folds in fold_results.items():
            assert len(folds) == 3, f"{name} should have 3 fold entries"

    def test_summary_mean_std_present(self, sample_csv):
        from src.experiment import run_cv, summarize

        df, _ = load_and_clean(sample_csv)
        X, y, _ = make_features(df)
        fold_results = run_cv(X, y, n_splits=3)
        summary = summarize(fold_results)

        lr = summary["logistic_regression"]["roc_auc"]
        assert "mean" in lr and "std" in lr and "values" in lr
        assert len(lr["values"]) == 3
