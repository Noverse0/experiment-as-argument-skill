"""Tests for the churn experiment pipeline."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

from src.data import FEATURE_COLS, TARGET_COL, load_and_prepare, get_X_y, class_balance
from src.pipeline import build_lr_pipeline, build_gbm_pipeline
from src.evaluate import cross_validate_temporal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_df(n=300, seed=0):
    X, y = make_classification(n_samples=n, n_features=3, n_informative=2,
                                n_redundant=1, n_repeated=0, random_state=seed)
    df = pd.DataFrame(X, columns=FEATURE_COLS)
    df[TARGET_COL] = y
    return df


def _churn_csv(tmp_path, with_duplicates=True):
    """Write a minimal churn-format CSV to a temp file."""
    rows = [
        (1, "2023-01-10", 12, 50.0, 1, 5, 0),
        (2, "2023-02-15", 24, 80.0, 3, 60, 1),
        (3, "2023-03-20", 6,  30.0, 0, 3, 0),
        (4, "2023-04-01", 36, 120.0, 2, 45, 1),
        (5, "2023-05-10", 18, 70.0, 1, 8, 0),
    ]
    if with_duplicates:
        rows += [(2, "2023-02-15", 24, 80.0, 3, 60, 1)]  # exact dup of row 2
    cols = ["customer_id", "signup_date", "tenure_months", "monthly_spend",
            "support_tickets", "days_since_last_login", "churned"]
    df = pd.DataFrame(rows, columns=cols)
    path = tmp_path / "test_churn.csv"
    df.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

class TestLoadAndPrepare:
    def test_removes_exact_duplicates(self, tmp_path):
        path = _churn_csv(tmp_path, with_duplicates=True)
        df = load_and_prepare(path)
        assert len(df) == 5, "Duplicate row should be removed"

    def test_no_duplicates_unchanged(self, tmp_path):
        path = _churn_csv(tmp_path, with_duplicates=False)
        df = load_and_prepare(path)
        assert len(df) == 5

    def test_sorted_by_signup_date(self, tmp_path):
        path = _churn_csv(tmp_path)
        df = load_and_prepare(path)
        dates = df["signup_date"].tolist()
        assert dates == sorted(dates)

    def test_returns_dataframe(self, tmp_path):
        path = _churn_csv(tmp_path)
        result = load_and_prepare(path)
        assert isinstance(result, pd.DataFrame)


class TestFeatureSelection:
    def test_no_leak_feature_in_cols(self):
        assert "days_since_last_login" not in FEATURE_COLS

    def test_no_identifier_in_cols(self):
        assert "customer_id" not in FEATURE_COLS

    def test_no_date_col_in_cols(self):
        assert "signup_date" not in FEATURE_COLS

    def test_legitimate_features_present(self):
        for col in ("tenure_months", "monthly_spend", "support_tickets"):
            assert col in FEATURE_COLS

    def test_get_X_y_shapes(self, tmp_path):
        path = _churn_csv(tmp_path)
        df = load_and_prepare(path)
        X, y = get_X_y(df)
        assert X.shape[1] == len(FEATURE_COLS)
        assert len(y) == len(df)
        assert y.name == TARGET_COL

    def test_class_balance_output(self):
        y = pd.Series([0, 0, 0, 1, 1])
        bal = class_balance(y)
        assert bal["n_positive"] == 2
        assert bal["n_negative"] == 3
        assert abs(bal["churn_rate"] - 0.4) < 1e-6


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

class TestLRPipeline:
    def test_fit_predict(self):
        df = _synthetic_df()
        X, y = df[FEATURE_COLS], df[TARGET_COL]
        pipe = build_lr_pipeline()
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape_and_sums(self):
        df = _synthetic_df()
        X, y = df[FEATURE_COLS], df[TARGET_COL]
        pipe = build_lr_pipeline()
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        assert proba.shape == (len(X), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_deterministic_with_same_seed(self):
        df = _synthetic_df()
        X, y = df[FEATURE_COLS], df[TARGET_COL]
        p1 = build_lr_pipeline(random_state=7)
        p2 = build_lr_pipeline(random_state=7)
        p1.fit(X, y)
        p2.fit(X, y)
        np.testing.assert_array_equal(
            p1.predict_proba(X), p2.predict_proba(X)
        )


class TestGBMPipeline:
    def test_fit_predict(self):
        df = _synthetic_df()
        X, y = df[FEATURE_COLS], df[TARGET_COL]
        pipe = build_gbm_pipeline()
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(X)

    def test_predict_proba_shape(self):
        df = _synthetic_df()
        X, y = df[FEATURE_COLS], df[TARGET_COL]
        pipe = build_gbm_pipeline()
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        assert proba.shape == (len(X), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class TestTemporalCV:
    def test_returns_correct_keys(self):
        df = _synthetic_df(n=400)
        X, y = df[FEATURE_COLS], df[TARGET_COL]
        scores = cross_validate_temporal(build_lr_pipeline(), X, y, n_splits=3)
        for metric in ("auc", "f1", "accuracy"):
            assert metric in scores
            assert "mean" in scores[metric]
            assert "std" in scores[metric]
            assert "values" in scores[metric]

    def test_n_folds_matches_splits(self):
        df = _synthetic_df(n=400)
        X, y = df[FEATURE_COLS], df[TARGET_COL]
        scores = cross_validate_temporal(build_lr_pipeline(), X, y, n_splits=4)
        assert len(scores["auc"]["values"]) == 4

    def test_auc_above_baseline(self):
        """Models on real signal must beat random (AUC > 0.5) on avg."""
        df = _synthetic_df(n=600, seed=42)
        X, y = df[FEATURE_COLS], df[TARGET_COL]
        scores = cross_validate_temporal(build_lr_pipeline(), X, y, n_splits=3)
        assert scores["auc"]["mean"] > 0.5

    def test_auc_in_valid_range(self):
        df = _synthetic_df(n=400)
        X, y = df[FEATURE_COLS], df[TARGET_COL]
        for scores in (
            cross_validate_temporal(build_lr_pipeline(), X, y, n_splits=3),
            cross_validate_temporal(build_gbm_pipeline(), X, y, n_splits=3),
        ):
            assert 0.0 <= scores["auc"]["mean"] <= 1.0


class TestLabelShuffleSanity:
    """With shuffled labels, AUC should fall near 0.5 (no real signal)."""

    def test_lr_degrades_on_shuffled_labels(self):
        rng = np.random.default_rng(0)
        df = _synthetic_df(n=500, seed=0)
        X = df[FEATURE_COLS]
        y_shuffled = pd.Series(rng.permutation(df[TARGET_COL].values))

        # Train on first half, evaluate on second half.
        mid = len(X) // 2
        pipe = build_lr_pipeline()
        pipe.fit(X.iloc[:mid], y_shuffled.iloc[:mid])
        proba = pipe.predict_proba(X.iloc[mid:])[:, 1]
        auc = roc_auc_score(y_shuffled.iloc[mid:], proba)
        assert auc < 0.65, (
            f"After label shuffle, AUC should be near 0.5; got {auc:.3f}. "
            "This may indicate feature-to-label leakage."
        )

    def test_gbm_degrades_on_shuffled_labels(self):
        rng = np.random.default_rng(1)
        df = _synthetic_df(n=500, seed=1)
        X = df[FEATURE_COLS]
        y_shuffled = pd.Series(rng.permutation(df[TARGET_COL].values))

        mid = len(X) // 2
        pipe = build_gbm_pipeline()
        pipe.fit(X.iloc[:mid], y_shuffled.iloc[:mid])
        proba = pipe.predict_proba(X.iloc[mid:])[:, 1]
        auc = roc_auc_score(y_shuffled.iloc[mid:], proba)
        assert auc < 0.65, (
            f"After label shuffle, AUC should be near 0.5; got {auc:.3f}."
        )
