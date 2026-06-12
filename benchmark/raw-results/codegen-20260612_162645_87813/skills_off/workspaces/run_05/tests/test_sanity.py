"""Tests for the evaluation methodology and sanity checks.

These run on the real generated dataset (small + fast) and assert the
guardrails actually hold: no leak survives, the baseline floor is chance,
shuffled labels destroy signal, and the temporal split never lets a test
fold precede its training fold.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from churn_experiment import data as data_mod
from churn_experiment import evaluate as ev

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    out = tmp_path_factory.mktemp("data") / "churn.csv"
    subprocess.run(
        [sys.executable, str(ROOT / "make_dataset.py"), "--out", str(out)],
        check=True,
    )
    return data_mod.prepare(data_mod.load_raw(str(out)))


def test_baseline_floor_is_chance(prepared):
    floor = ev.baseline_floor(prepared)
    assert floor["roc_auc_mean"] == pytest.approx(0.5, abs=1e-9)


def test_label_shuffle_collapses_to_chance(prepared):
    for name in ev.MODEL_FACTORIES:
        auc = ev.label_shuffle_auc(name, seed=7, data=prepared)
        # shuffled labels => no signal; allow modest slack for finite folds
        assert auc < 0.6, f"{name} kept signal after label shuffle: {auc}"


def test_overfit_tiny_subset(prepared):
    for name in ev.MODEL_FACTORIES:
        auc = ev.overfit_tiny_subset(name, seed=7, data=prepared, n=40)
        # must fit far above chance; a linear model won't always hit 1.0
        assert auc > 0.85, f"{name} could not fit a tiny subset: {auc}"


def test_models_beat_floor(prepared):
    for name in ev.MODEL_FACTORIES:
        arm = ev.evaluate_arm(name, seed=7, data=prepared)
        assert arm.roc_auc_mean > 0.55, f"{name} did not beat chance"
        assert arm.n_folds == ev.N_SPLITS


def test_temporal_split_has_no_lookahead(prepared):
    """Every test fold's earliest signup is >= its train fold's latest."""
    from sklearn.model_selection import TimeSeriesSplit

    times = prepared.time.reset_index(drop=True)
    for tr_idx, te_idx in TimeSeriesSplit(n_splits=ev.N_SPLITS).split(prepared.X):
        assert times.iloc[te_idx].min() >= times.iloc[tr_idx].max()


def test_determinism_same_seed(prepared):
    a = ev.evaluate_arm("gradient_boosting", seed=7, data=prepared)
    b = ev.evaluate_arm("gradient_boosting", seed=7, data=prepared)
    assert a.roc_auc_per_fold == b.roc_auc_per_fold
