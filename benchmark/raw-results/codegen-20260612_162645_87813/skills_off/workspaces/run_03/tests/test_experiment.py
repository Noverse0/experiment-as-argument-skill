"""Tests for the churn experiment pipeline.

These tests encode the rigor rules as executable checks: leak columns never
reach the model, dedup happens before any split, the time split is honored, the
sanity checks behave, and the run is reproducible.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn_experiment.data import (  # noqa: E402
    FEATURES,
    LEAK_COLUMNS,
    TARGET,
    assert_no_leak_columns,
    load_dataset,
)
from churn_experiment.evaluate import compare_models, evaluate_model  # noqa: E402
from churn_experiment.models import build_models  # noqa: E402
from churn_experiment.sanity import (  # noqa: E402
    label_shuffle_auc,
    overfit_tiny_subset,
)


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory) -> str:
    """Generate a churn CSV via the project's own generator (small + fast)."""
    import subprocess

    out = tmp_path_factory.mktemp("data") / "churn.csv"
    subprocess.check_call(
        ["python3", "make_dataset.py", "--out", str(out), "--seed", "7"], cwd=ROOT
    )
    return str(out)


@pytest.fixture(scope="module")
def dataset(csv_path):
    return load_dataset(csv_path)


# --- Data discipline -------------------------------------------------------

def test_duplicates_removed_before_split(csv_path, dataset):
    raw = pd.read_csv(csv_path)
    assert raw.duplicated().sum() > 0, "fixture should contain planted duplicates"
    # After load, no exact duplicate rows remain.
    assert dataset.frame.duplicated().sum() == 0
    assert dataset.n_duplicates_removed == raw.duplicated().sum()


def test_feature_matrix_excludes_leak_columns(dataset):
    cols = set(dataset.X.columns)
    for leak in LEAK_COLUMNS:
        assert leak not in cols, f"{leak} must not be a feature"
    assert cols == set(FEATURES)


def test_assert_no_leak_columns_raises():
    bad = pd.DataFrame({"account_status": [1], "tenure_months": [2]})
    with pytest.raises(AssertionError):
        assert_no_leak_columns(bad)


def test_account_status_would_be_a_perfect_leak(csv_path):
    # Document the trap: account_status perfectly determines the target.
    raw = pd.read_csv(csv_path)
    ct = pd.crosstab(raw["account_status"], raw[TARGET])
    # Each status maps to exactly one class -> perfect predictor if used.
    assert (ct > 0).sum(axis=1).max() == 1


def test_rows_ordered_by_time(dataset):
    dates = dataset.frame["signup_date"]
    assert dates.is_monotonic_increasing


# --- Sanity checks behave --------------------------------------------------

def test_label_shuffle_collapses_to_chance(dataset):
    models = build_models(seed=0)
    for name, m in models.items():
        auc = label_shuffle_auc(m, dataset, seed=0)
        assert abs(auc - 0.5) < 0.1, f"{name} predicts shuffled labels (AUC={auc})"


def test_overfit_tiny_subset(dataset):
    models = build_models(seed=0)
    for name, m in models.items():
        auc = overfit_tiny_subset(m, dataset)
        assert auc > 0.9, f"{name} cannot fit a tiny slice (AUC={auc})"


# --- Evaluation + comparison ----------------------------------------------

def test_models_beat_baseline_but_not_perfect(dataset):
    models = build_models(seed=0)
    for name, m in models.items():
        res = evaluate_model(m, dataset, n_splits=5)
        auc = res.mean("roc_auc")
        assert auc > 0.5, f"{name} does not beat chance (AUC={auc})"
        assert auc < 0.95, f"{name} suspiciously high — possible leak (AUC={auc})"
        assert res.n == 5


def test_reproducible_same_seed(dataset):
    m1 = build_models(seed=123)["gradient_boosting"]
    m2 = build_models(seed=123)["gradient_boosting"]
    r1 = evaluate_model(m1, dataset, n_splits=5)
    r2 = evaluate_model(m2, dataset, n_splits=5)
    assert r1.per_fold["roc_auc"] == r2.per_fold["roc_auc"]


def test_comparison_reports_variance(dataset):
    models = build_models(seed=0)
    lr = evaluate_model(models["logistic_regression"], dataset, n_splits=5)
    gb = evaluate_model(models["gradient_boosting"], dataset, n_splits=5)
    cmp = compare_models(lr, gb)
    # Conclusion must mention n and either a winner or "no detectable difference".
    assert "n=5" in cmp.conclusion
    assert cmp.sd_diff >= 0.0
