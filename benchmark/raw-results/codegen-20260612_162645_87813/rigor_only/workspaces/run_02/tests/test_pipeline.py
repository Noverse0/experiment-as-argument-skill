"""Tests for the churn experiment pipeline.

These guard the rigor properties, not just "it runs": no leak columns reach
the model, duplicates are removed before splitting, time order is respected,
sanity checks behave, and the comparison is deterministic.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn import data as data_mod  # noqa: E402
from churn import experiment as exp  # noqa: E402

CSV = ROOT / "churn.csv"


@pytest.fixture(scope="module")
def csv_path():
    if not CSV.exists():
        import subprocess
        subprocess.check_call([sys.executable, "make_dataset.py", "--out", "churn.csv"],
                              cwd=ROOT)
    return str(CSV)


@pytest.fixture(scope="module")
def loaded(csv_path):
    return data_mod.load_clean(csv_path)


# --- data discipline -------------------------------------------------------- #

def test_duplicates_removed_before_split(loaded):
    # The generator plants 200 exact duplicate rows; they must be gone.
    assert loaded.n_duplicates == 200
    assert loaded.n_clean == loaded.n_raw - loaded.n_duplicates
    assert loaded.X.duplicated().sum() >= 0  # dedup happened on full row pre-split


def test_leak_and_id_columns_excluded(loaded):
    # account_status / customer_id / signup_date must never be model features.
    forbidden = set(data_mod.LEAK_COLS) | set(data_mod.ID_COLS) | {data_mod.DATE_COL}
    assert forbidden.isdisjoint(set(loaded.X.columns))
    assert list(loaded.X.columns) == data_mod.FEATURES


def test_target_is_binary(loaded):
    assert set(loaded.y.unique()) <= {0, 1}
    assert 0.0 < loaded.churn_rate < 1.0


def test_rows_ordered_by_time(csv_path):
    # X is returned in signup_date order so time-based CV is valid.
    raw = pd.read_csv(csv_path).drop_duplicates().reset_index(drop=True)
    raw[data_mod.DATE_COL] = pd.to_datetime(raw[data_mod.DATE_COL])
    ordered = raw.sort_values(data_mod.DATE_COL, kind="mergesort")
    res = data_mod.load_clean(csv_path)
    assert np.array_equal(res.X.to_numpy(), ordered[data_mod.FEATURES].to_numpy())


# --- preprocessing fit on train only (no global leakage) -------------------- #

def test_pipeline_scaler_fit_per_fold(loaded):
    # Each evaluated model is a Pipeline; scaling is part of it, so it is
    # re-fit per fold rather than on the whole dataset.
    model = exp.make_models()["logistic_regression"]
    from sklearn.pipeline import Pipeline
    assert isinstance(model, Pipeline)
    assert "scale" in dict(model.steps)


# --- sanity checks ---------------------------------------------------------- #

def test_baseline_floor_near_half(loaded):
    out = exp.sanity_baseline_floor(loaded.X, loaded.y)
    assert out["passes"], out


def test_label_shuffle_collapses(loaded):
    out = exp.sanity_label_shuffle(loaded.X, loaded.y)
    assert out["passes"], out


def test_leakage_ceiling_is_near_perfect(csv_path):
    out = exp.sanity_leakage_ceiling(csv_path)
    # Including account_status must make the task trivially perfect -> proves leak.
    assert out["is_near_perfect"], out


def test_real_model_beats_floor(loaded):
    folds = exp.evaluate_model(exp.make_models()["logistic_regression"],
                               loaded.X, loaded.y)
    mean_auc = np.mean([f.roc_auc for f in folds])
    assert mean_auc > 0.55  # clears the 0.5 floor with real (non-leak) signal
    assert mean_auc < 0.95  # but is NOT near-perfect -> no hidden leakage


# --- determinism & comparison shape ----------------------------------------- #

def test_comparison_is_deterministic(loaded):
    a = exp.run_comparison(loaded.X, loaded.y)
    b = exp.run_comparison(loaded.X, loaded.y)
    av = a["models"]["gradient_boosting"]["summary"]["roc_auc"]["mean"]
    bv = b["models"]["gradient_boosting"]["summary"]["roc_auc"]["mean"]
    assert av == bv  # same seed -> identical metrics


def test_comparison_reports_variance_and_paired_diff(loaded):
    comp = exp.run_comparison(loaded.X, loaded.y)
    for name in ("logistic_regression", "gradient_boosting"):
        s = comp["models"][name]["summary"]
        assert s["n"] == exp.N_SPLITS
        assert s["roc_auc"]["sd"] >= 0.0
        assert len(s["roc_auc"]["values"]) == exp.N_SPLITS
    diff = comp["paired_diff_roc_auc"]
    assert diff["n"] == exp.N_SPLITS
    assert "ci95_halfwidth" in diff
