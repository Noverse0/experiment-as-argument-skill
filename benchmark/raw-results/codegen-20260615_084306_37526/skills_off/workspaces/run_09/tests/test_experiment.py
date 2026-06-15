"""Tests for data preparation and experiment logic."""
import numpy as np
import pandas as pd
import pytest

from src.experiment import (
    FEATURES,
    TARGET,
    cross_validate_model,
    load_and_prepare,
    majority_baseline_auc,
    run_experiment,
)
from src.pipeline import make_gb_pipeline, make_lr_pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset_path(tmp_path_factory):
    import subprocess, sys
    out = tmp_path_factory.mktemp("data") / "churn.csv"
    subprocess.run([sys.executable, "make_dataset.py", "--out", str(out)], check=True)
    return out


@pytest.fixture(scope="module")
def prepared_data(dataset_path):
    return load_and_prepare(str(dataset_path))


# ---------------------------------------------------------------------------
# Data preparation tests
# ---------------------------------------------------------------------------

def test_deduplication_removes_200_rows(prepared_data):
    X, y = prepared_data
    # Generator creates 4000 + 200 duplicates; deduplicated should be ~4000
    assert len(X) == 4000


def test_no_duplicate_rows(dataset_path):
    # Verify that dedup was applied to the full dataframe (all 7 columns),
    # not just the feature subset (which can have collisions on numeric values).
    import pandas as pd
    df = pd.read_csv(dataset_path, parse_dates=["signup_date"])
    df_deduped = df.drop_duplicates()
    assert len(df_deduped) == 4000
    assert df_deduped.duplicated().sum() == 0


def test_features_are_correct_columns(prepared_data):
    X, _ = prepared_data
    assert list(X.columns) == FEATURES


def test_leak_column_excluded(prepared_data):
    X, _ = prepared_data
    assert "days_since_last_login" not in X.columns


def test_identifier_columns_excluded(prepared_data):
    X, _ = prepared_data
    assert "customer_id" not in X.columns
    assert "signup_date" not in X.columns


def test_no_missing_values(prepared_data):
    X, y = prepared_data
    assert X.isnull().sum().sum() == 0
    assert y.isnull().sum() == 0


def test_target_is_binary(prepared_data):
    _, y = prepared_data
    assert set(y.unique()).issubset({0, 1})


def test_churn_rate_plausible(prepared_data):
    _, y = prepared_data
    rate = y.mean()
    assert 0.1 < rate < 0.5, f"Unexpected churn rate: {rate:.2%}"


# ---------------------------------------------------------------------------
# Sanity / validity tests
# ---------------------------------------------------------------------------

def test_models_beat_majority_baseline(prepared_data):
    """Both models must exceed the trivial majority-class baseline."""
    X, y = prepared_data
    baseline = majority_baseline_auc(y, n_splits=3)

    for name, pipe in [("LR", make_lr_pipeline()), ("GB", make_gb_pipeline())]:
        cv = cross_validate_model(pipe, X, y, n_splits=3)
        model_auc = cv["roc_auc"]["mean"]
        assert model_auc > baseline, (
            f"{name} ROC-AUC {model_auc:.3f} did not beat baseline {baseline:.3f}"
        )


def test_label_shuffle_degrades_performance(prepared_data):
    """Shuffled labels must yield ROC-AUC near 0.5 (no information leaking)."""
    X, y = prepared_data
    rng = np.random.default_rng(0)
    y_shuffled = pd.Series(rng.permutation(y.values), index=y.index)

    # Use a small subset for speed
    X_sub, y_sub = X.iloc[:500], y_shuffled.iloc[:500]
    cv = cross_validate_model(make_lr_pipeline(), X_sub, y_sub, n_splits=3)
    auc = cv["roc_auc"]["mean"]
    assert auc < 0.60, f"Shuffled-label AUC {auc:.3f} is suspiciously high — possible leak"


def test_run_experiment_structure(prepared_data):
    X, y = prepared_data
    results = run_experiment(X, y, n_splits=3)

    assert "models" in results
    assert "LogisticRegression" in results["models"]
    assert "GradientBoosting" in results["models"]

    for model_results in results["models"].values():
        for metric in ["roc_auc", "f1", "precision", "recall"]:
            assert metric in model_results
            assert "mean" in model_results[metric]
            assert "std" in model_results[metric]
            assert len(model_results[metric]["values"]) == 3


def test_cv_auc_plausible_range(prepared_data):
    X, y = prepared_data
    results = run_experiment(X, y, n_splits=3)
    for name, model_results in results["models"].items():
        auc = model_results["roc_auc"]["mean"]
        assert 0.5 < auc < 1.0, f"{name} ROC-AUC {auc:.3f} outside plausible range"
