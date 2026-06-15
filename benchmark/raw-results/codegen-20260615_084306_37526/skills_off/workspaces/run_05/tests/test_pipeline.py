import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import run_cv
from src.pipeline import FEATURE_COLS, LEAK_COLS, build_pipeline, load_and_clean


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(n: int, seed: int = 0) -> pd.DataFrame:
    """Matches the generative process in make_dataset.py so labels have real signal."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    spend = rng.gamma(2.0, 30.0, n).round(2)
    tickets = rng.poisson(1.2, n)
    logit = -1.2 - 0.03 * tenure + 0.01 * spend + 0.45 * tickets
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "customer_id": np.arange(n),
        "signup_date": "2023-01-01",
        "tenure_months": tenure,
        "monthly_spend": spend,
        "support_tickets": tickets,
        "days_since_last_login": rng.integers(1, 100, n),
        "churned": churn,
    })


@pytest.fixture
def csv_path(tmp_path):
    df = _make_df(600)
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def csv_with_dupes(tmp_path):
    df = _make_df(60)
    duped = pd.concat([df, df.iloc[:15]], ignore_index=True)
    path = tmp_path / "dupes.csv"
    duped.to_csv(path, index=False)
    return str(path), 60


# ---------------------------------------------------------------------------
# load_and_clean
# ---------------------------------------------------------------------------

def test_leak_columns_are_dropped(csv_path):
    X, _ = load_and_clean(csv_path)
    for col in LEAK_COLS:
        assert col not in X.columns, f"Leak column '{col}' must be dropped"


def test_identifier_columns_are_dropped(csv_path):
    X, _ = load_and_clean(csv_path)
    assert "customer_id" not in X.columns
    assert "signup_date" not in X.columns


def test_only_feature_cols_remain(csv_path):
    X, _ = load_and_clean(csv_path)
    assert list(X.columns) == FEATURE_COLS


def test_target_is_binary(csv_path):
    _, y = load_and_clean(csv_path)
    assert set(y.unique()).issubset({0, 1})


def test_deduplication_removes_exact_duplicates(csv_with_dupes):
    path, expected_len = csv_with_dupes
    X, y = load_and_clean(path)
    assert len(X) == expected_len


# ---------------------------------------------------------------------------
# build_pipeline
# ---------------------------------------------------------------------------

def test_build_logistic_regression():
    pipe = build_pipeline("logistic_regression")
    assert hasattr(pipe, "fit") and hasattr(pipe, "predict_proba")


def test_build_gradient_boosting():
    pipe = build_pipeline("gradient_boosting")
    assert hasattr(pipe, "fit") and hasattr(pipe, "predict_proba")


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_pipeline("xgboost")


def test_scaler_not_prefitted():
    pipe = build_pipeline("logistic_regression")
    scaler = pipe.named_steps["scaler"]
    assert not hasattr(scaler, "mean_"), "Scaler must not be fitted before the pipeline runs"


def test_gradient_boosting_has_no_scaler():
    pipe = build_pipeline("gradient_boosting")
    assert "scaler" not in pipe.named_steps


# ---------------------------------------------------------------------------
# Pipeline fit / predict
# ---------------------------------------------------------------------------

def test_pipeline_fits_and_produces_probabilities(csv_path):
    X, y = load_and_clean(csv_path)
    for name in ["logistic_regression", "gradient_boosting"]:
        pipe = build_pipeline(name)
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# run_cv
# ---------------------------------------------------------------------------

def test_cv_result_structure(csv_path):
    X, y = load_and_clean(csv_path)
    pipe = build_pipeline("logistic_regression")
    res = run_cv(pipe, X, y, seeds=[0], n_splits=3)
    for metric in ["roc_auc", "f1", "accuracy"]:
        assert metric in res
        assert {"mean", "std", "n", "values"} <= res[metric].keys()
        assert res[metric]["n"] == 3  # 1 seed × 3 folds


def test_cv_score_count(csv_path):
    X, y = load_and_clean(csv_path)
    pipe = build_pipeline("logistic_regression")
    res = run_cv(pipe, X, y, seeds=[0, 1], n_splits=4)
    assert res["roc_auc"]["n"] == 8  # 2 seeds × 4 folds


# ---------------------------------------------------------------------------
# Sanity: models beat the majority-class floor
# ---------------------------------------------------------------------------

def test_models_beat_majority_class_baseline(csv_path):
    X, y = load_and_clean(csv_path)
    for name in ["logistic_regression", "gradient_boosting"]:
        pipe = build_pipeline(name)
        res = run_cv(pipe, X, y, seeds=[42], n_splits=3)
        auc = res["roc_auc"]["mean"]
        assert auc > 0.5, f"{name} AUC={auc:.4f} did not exceed majority-class baseline 0.5"
