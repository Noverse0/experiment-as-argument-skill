"""Tests for the churn pipeline: leak defenses, determinism, and the sanity
checks that gate the comparison."""
from __future__ import annotations

import os
import subprocess
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from churn.data import (  # noqa: E402
    FEATURE_COLUMNS,
    LEAK_COLUMNS,
    load_clean,
)
from churn.experiment import (  # noqa: E402
    evaluate_model,
    run_comparison,
    run_sanity_checks,
)
from churn.models import make_gradient_boosting, make_logistic_regression


@pytest.fixture(scope="session")
def csv_path(tmp_path_factory) -> str:
    """Generate a fresh dataset once for the whole test session."""
    out = str(tmp_path_factory.mktemp("data") / "churn.csv")
    subprocess.run(
        ["python3", "make_dataset.py", "--out", out], cwd=ROOT, check=True
    )
    return out


@pytest.fixture(scope="session")
def data(csv_path):
    return load_clean(csv_path)


# --- leak defenses --------------------------------------------------------- #

def test_leak_and_id_columns_excluded(data):
    """The target-leaking and identifier columns never reach the features."""
    for col in LEAK_COLUMNS + ["customer_id", "signup_date"]:
        assert col not in data.X.columns
    assert list(data.X.columns) == FEATURE_COLUMNS


def test_duplicates_removed_before_use(csv_path, data):
    """The planted exact-duplicate rows are removed before any split, so none
    can straddle the train/test boundary.

    Note: after dropping `customer_id`, two *distinct* customers may coincide on
    the 3 coarse features + label. That is benign (real separate rows), not a
    straddling duplicate, so we assert on full original-row dedup, not on the
    feature subset."""
    raw = pd.read_csv(csv_path)
    assert data.n_duplicates_removed == raw.duplicated().sum()
    assert data.n_duplicates_removed > 0
    # No full original row survives more than once after cleaning.
    assert raw.drop_duplicates().shape[0] == len(data.X)


def test_time_ordering(data):
    """Rows are sorted by signup_date so the time split is forward-looking."""
    assert data.dates.is_monotonic_increasing


# --- determinism ----------------------------------------------------------- #

def test_runs_are_deterministic(data):
    """Same seed, same data -> identical metrics (no hidden nondeterminism)."""
    a = evaluate_model(make_gradient_boosting(), data.X, data.y)
    b = evaluate_model(make_gradient_boosting(), data.X, data.y)
    assert a == b


# --- sanity checks gate the comparison ------------------------------------- #

def test_sanity_checks_pass(data, csv_path):
    checks = run_sanity_checks(data, csv_path)
    assert checks["baseline_floor"]["passed"], checks["baseline_floor"]
    assert checks["label_shuffle"]["passed"], checks["label_shuffle"]
    assert checks["leakage_ceiling"]["passed"], checks["leakage_ceiling"]


def test_leakage_ceiling_is_near_perfect(data, csv_path):
    """account_status alone must be near-perfect, proving it is a leak."""
    checks = run_sanity_checks(data, csv_path)
    assert checks["leakage_ceiling"]["mean_roc_auc"] > 0.99


def test_models_beat_baseline(data):
    """Both models must clear the chance floor on the real (clean) task."""
    for factory in (make_logistic_regression, make_gradient_boosting):
        folds = evaluate_model(factory(), data.X, data.y)
        mean_auc = sum(f["roc_auc"] for f in folds) / len(folds)
        assert mean_auc > 0.55, mean_auc


# --- comparison output shape ----------------------------------------------- #

def test_comparison_structure_and_honest_verdict(data):
    res = run_comparison(data)
    comp = res["comparison"]
    assert comp["verdict"] in {
        "no_detectable_difference",
        "gradient_boosting_better",
        "logistic_regression_better",
    }
    assert len(comp["per_fold_diff"]) == res["n_splits"]
    for arm in res["arms"].values():
        assert "roc_auc" in arm and "mean" in arm["roc_auc"] and "sd" in arm["roc_auc"]
