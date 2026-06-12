"""Tests for the churn experiment pipeline.

These assert the *rigor invariants* that make the comparison trustworthy, not just
that code runs: no leak column, dedup before split, time ordering, determinism, and
the sanity-check guarantees (baseline floor, leakage ceiling, label-shuffle collapse,
tiny-subset overfit).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import prepare, FEATURES, LEAK_COLS, ID_COLS  # noqa: E402
from src.evaluate import (  # noqa: E402
    baseline_floor,
    compare_arms,
    cross_validate_arm,
    label_shuffle_auc,
    leakage_ceiling_auc,
    overfit_tiny_subset_auc,
)
from src.pipeline import ARMS, make_pipeline  # noqa: E402

DATA = str(ROOT / "churn.csv")


@pytest.fixture(scope="module")
def data():
    assert Path(DATA).exists(), "run `python make_dataset.py --out churn.csv` first"
    return prepare(DATA)


# ----------------------------- data discipline ----------------------------- #

def test_leak_and_id_columns_are_excluded(data):
    for col in LEAK_COLS + ID_COLS + ["signup_date"]:
        assert col not in data.X.columns
    assert list(data.X.columns) == FEATURES


def test_duplicates_removed_before_split(data):
    # Generator appends exactly 200 exact duplicate rows.
    assert data.n_duplicates_dropped == 200
    assert data.X.duplicated().sum() < data.n_duplicates_dropped  # materially deduped


def test_target_is_binary_and_imbalanced(data):
    assert set(np.unique(data.y)) == {0, 1}
    assert 0.1 < data.churn_rate < 0.45  # minority positive class


# ----------------------------- determinism --------------------------------- #

def test_reruns_are_bit_identical(data):
    a = cross_validate_arm("logreg", data.X, data.y)
    b = cross_validate_arm("logreg", data.X, data.y)
    assert a["aggregate"]["roc_auc"]["values"] == b["aggregate"]["roc_auc"]["values"]


# ----------------------------- sanity checks ------------------------------- #

def test_baseline_floor_is_chance(data):
    assert abs(baseline_floor(data.X, data.y)["roc_auc_mean"] - 0.5) < 0.05


def test_leakage_ceiling_is_near_perfect():
    # Including account_status must expose the planted leak.
    assert leakage_ceiling_auc(DATA) > 0.99


def test_label_shuffle_collapses_to_chance(data):
    for arm in ARMS:
        auc = label_shuffle_auc(arm, data.X, data.y)
        assert abs(auc - 0.5) < 0.08, f"{arm} leaks: shuffled AUC={auc}"


def test_model_can_overfit_tiny_subset(data):
    for arm in ARMS:
        assert overfit_tiny_subset_auc(arm, data.X, data.y) > 0.9


# ----------------------------- real signal & comparison -------------------- #

def test_models_beat_baseline(data):
    for arm in ARMS:
        res = cross_validate_arm(arm, data.X, data.y)
        assert res["aggregate"]["roc_auc"]["mean"] > 0.55


def test_comparison_reports_honest_conclusion(data):
    a = cross_validate_arm("gboost", data.X, data.y)
    b = cross_validate_arm("logreg", data.X, data.y)
    comp = compare_arms(a, b)
    assert comp["n_folds"] == 5
    assert len(comp["ci95_diff"]) == 2
    # Conclusion must be one of the three allowed honest verdicts.
    assert (
        comp["conclusion"] == "no detectable difference"
        or "outperforms" in comp["conclusion"]
    )


def test_unknown_arm_raises():
    with pytest.raises(ValueError):
        make_pipeline("random_forest")


# ----------------------------- end-to-end ---------------------------------- #

def test_entrypoint_writes_artifacts(tmp_path):
    results_dir = tmp_path / "results"
    report = tmp_path / "REPORT.md"
    out = subprocess.run(
        [sys.executable, "run_experiment.py",
         "--data", DATA,
         "--results-dir", str(results_dir),
         "--report", str(report)],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert (results_dir / "metrics.json").exists()
    assert report.exists()
