"""Tests that guard the experiment's validity, not just that it runs.

A churn.csv is generated once per session in a temp dir so tests are isolated
from any committed data file.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data as data_mod  # noqa: E402
from src import sanity as sanity_mod  # noqa: E402
from src.data import FEATURES, prepare  # noqa: E402
from src.experiment import (  # noqa: E402
    SEEDS,
    conclude,
    cross_validate_arms,
    paired_comparison,
    run_full_experiment,
)


@pytest.fixture(scope="session")
def csv_path(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("data") / "churn.csv"
    subprocess.check_call(
        [sys.executable, str(ROOT / "make_dataset.py"), "--out", str(out)]
    )
    return str(out)


@pytest.fixture(scope="session")
def prepared(csv_path):
    return prepare(csv_path)


def test_dedup_removes_exact_duplicates(prepared):
    # The generator appends exactly 200 duplicate rows.
    assert prepared.n_duplicates_dropped == 200
    assert len(prepared.y) == prepared.n_raw - 200


def test_leaky_and_id_columns_are_dropped(prepared):
    cols = set(prepared.X.columns)
    assert "account_status" not in cols, "target leak must be dropped"
    assert "customer_id" not in cols, "identifier must be dropped"
    assert "signup_date" not in cols, "temporal column must not be a raw feature"
    assert list(prepared.X.columns) == FEATURES


def test_rows_are_time_ordered(csv_path):
    import pandas as pd

    raw = data_mod.load_raw(csv_path)
    df = raw.drop_duplicates().sort_values("signup_date", kind="mergesort")
    assert df["signup_date"].is_monotonic_increasing


def test_churn_rate_is_imbalanced(prepared):
    # ~27%: justifies AUC/PR-AUC over accuracy.
    assert 0.2 < prepared.churn_rate < 0.35


def test_baseline_floor_near_half(prepared):
    auc = sanity_mod.baseline_floor(prepared, SEEDS[0])
    assert abs(auc - 0.5) < 0.05


def test_label_shuffle_collapses_to_floor(prepared):
    # No information should survive shuffling the labels.
    auc = sanity_mod.label_shuffle_auc(prepared, SEEDS[0])
    assert abs(auc - 0.5) < 0.1


def test_overfit_tiny_subset(prepared):
    auc = sanity_mod.overfit_tiny_subset(prepared, SEEDS[0])
    assert auc > 0.95


def test_leakage_ceiling_is_near_perfect(csv_path):
    # Including account_status must expose a perfect leak.
    auc = sanity_mod.leakage_ceiling(csv_path, SEEDS[0])
    assert auc > 0.99


def test_models_beat_baseline_but_not_perfect(prepared):
    arms = cross_validate_arms(prepared)
    for name, arm in arms.items():
        m = sum(arm.cv_roc_auc) / len(arm.cv_roc_auc)
        assert m > 0.55, f"{name} must beat the no-information floor"
        assert m < 0.95, f"{name} AUC suspiciously high -> possible leak"


def test_determinism_same_seed(prepared):
    # Identical pipeline + same seeds must yield identical CV scores.
    a = cross_validate_arms(prepared)
    b = cross_validate_arms(prepared)
    assert a["gradient_boosting"].cv_roc_auc == b["gradient_boosting"].cv_roc_auc
    assert a["logistic_regression"].cv_roc_auc == b["logistic_regression"].cv_roc_auc


def test_paired_comparison_shape(prepared):
    arms = cross_validate_arms(prepared)
    comp = paired_comparison(arms)
    assert comp["n_pairs"] == len(SEEDS) * 5
    assert 0.0 <= comp["paired_p_value"] <= 1.0


def test_conclude_respects_significance():
    # No winner declared when the test is not significant.
    null = {"paired_p_value": 0.4, "delta_mean_gbm_minus_lr": 0.01}
    assert "No detectable difference" in conclude(null)
    sig = {"paired_p_value": 0.001, "delta_mean_gbm_minus_lr": 0.05}
    assert "outperforms" in conclude(sig)


def test_full_experiment_runs_and_is_consistent(prepared, csv_path):
    r = run_full_experiment(prepared, csv_path)
    assert set(r["arms"]) == {"logistic_regression", "gradient_boosting"}
    # Report conclusion must match the paired p-value (code/report consistency).
    p = r["comparison"]["paired_p_value"]
    if p >= 0.05:
        assert "No detectable difference" in r["conclusion"]
    else:
        assert "outperforms" in r["conclusion"]
