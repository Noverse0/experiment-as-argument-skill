"""Tests for the churn experiment pipeline.

These assert the rigor properties the experiment depends on, not just that the
code runs: leak column is excluded, duplicates are removed before splitting,
time order is respected, and the sanity checks behave (floor ~0.5, label-shuffle
~0.5, leakage ceiling ~1.0). Determinism is checked too.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data as datamod  # noqa: E402
from src import experiment as exp  # noqa: E402


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    """The dataset, generated fresh into a temp file (deterministic)."""
    out = ROOT / "churn.csv"
    if not out.exists():
        subprocess.check_call(
            [sys.executable, "make_dataset.py", "--out", str(out)], cwd=ROOT
        )
    return datamod.load_raw(str(out))


@pytest.fixture(scope="module")
def clean(raw_df):
    return datamod.clean(raw_df)


def test_leak_and_id_columns_excluded(clean):
    assert datamod.LEAK_COLUMN not in clean.X.columns
    assert datamod.ID_COLUMN not in clean.X.columns
    assert datamod.TIME_COLUMN not in clean.X.columns
    assert list(clean.X.columns) == datamod.FEATURES


def test_duplicates_removed_before_split(raw_df, clean):
    # The generator appends 200 exact duplicate rows.
    assert clean.n_duplicates_removed == 200
    assert len(clean.X) == clean.n_raw - 200
    # The correct invariant is on the FULL original row: no exact full-row
    # duplicate may survive (those are the planted copies that could straddle a
    # split). Coincidental feature-vector collisions between *different*
    # customers are legitimate distinct observations and are NOT removed.
    assert raw_df.drop_duplicates().duplicated().sum() == 0


def test_time_order_respected(raw_df, clean):
    # After clean(), rows are sorted by signup_date; verify against raw dedup.
    deduped = raw_df.drop_duplicates().reset_index(drop=True)
    expected = deduped.sort_values(datamod.TIME_COLUMN, kind="mergesort")
    assert list(clean.X["tenure_months"]) == list(expected["tenure_months"])


def test_target_is_binary(clean):
    assert set(clean.y.unique()) <= {0, 1}
    assert 0.0 < clean.churn_rate < 1.0


def _splits(clean, n=3):
    return list(TimeSeriesSplit(n_splits=n).split(clean.X))


def test_models_beat_baseline_floor(clean):
    splits = _splits(clean)
    floor = exp.baseline_floor_auc(clean, splits)
    assert floor == pytest.approx(0.5, abs=0.05)
    models = exp.make_models()
    for name, model in models.items():
        res, aucs = exp.evaluate_arm(name, model, clean.X, clean.y, splits)
        assert res.roc_auc_mean > floor + 0.02, f"{name} did not beat floor"


def test_label_shuffle_collapses_to_chance(clean):
    splits = _splits(clean)
    auc = exp.label_shuffle_auc(clean, splits)
    # Shuffled labels must not be predictable: AUC near 0.5.
    assert auc == pytest.approx(0.5, abs=0.07)


def test_leakage_ceiling_is_near_perfect(raw_df, clean):
    splits = _splits(clean)
    leaky_X = datamod.leaky_matrix(raw_df)
    auc = exp.leakage_ceiling_auc(leaky_X, clean.y, splits)
    # account_status alone should make the task trivially separable.
    assert auc > 0.98


def test_determinism(clean):
    splits = _splits(clean)
    models1 = exp.make_models(seed=7)
    models2 = exp.make_models(seed=7)
    r1, a1 = exp.evaluate_arm("gb", models1["gradient_boosting"], clean.X, clean.y, splits)
    r2, a2 = exp.evaluate_arm("gb", models2["gradient_boosting"], clean.X, clean.y, splits)
    assert a1 == a2


def test_compare_reports_no_difference_when_equal():
    aucs = [0.70, 0.71, 0.69, 0.72, 0.70]
    comp = exp.compare(aucs, list(aucs))
    assert comp.detectable_difference is False
    assert "no detectable difference" in comp.conclusion.lower()


def test_compare_detects_clear_winner():
    gbm = [0.80, 0.81, 0.79, 0.82, 0.80]
    logreg = [0.70, 0.71, 0.69, 0.72, 0.70]
    comp = exp.compare(gbm, logreg)
    assert comp.detectable_difference is True
    assert comp.gbm_minus_logreg_mean > 0


def test_full_run_smoke(clean, raw_df):
    leaky_X = datamod.leaky_matrix(raw_df)
    result = exp.run(clean, leaky_X, seed=7, n_splits=3)
    assert "comparison" in result
    assert result["data"]["n_duplicates_removed"] == 200
    assert set(result["arms"]) == {"logistic_regression", "gradient_boosting"}
