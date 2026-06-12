"""Tests for the experiment harness: sanity checks, determinism, no leakage."""
import subprocess
import sys
from pathlib import Path

import pytest

from src import data as data_mod
from src import experiment as exp

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "churn.csv"


@pytest.fixture(scope="session")
def clean():
    if not CSV.exists():
        subprocess.check_call(
            [sys.executable, "make_dataset.py", "--out", str(CSV)], cwd=ROOT
        )
    return data_mod.load_clean(str(CSV))


def test_models_present(clean):
    models = exp.make_models()
    assert set(models) == {"logistic_regression", "gradient_boosting"}


def test_arms_beat_baseline(clean):
    """Both models must clear the majority-class floor (~0.5)."""
    arms = exp.evaluate_arms(clean.X, clean.y, exp.make_models(), n_splits=exp.N_SPLITS)
    for name, res in arms.items():
        mean_auc = res.metric_array("roc_auc").mean()
        assert mean_auc > 0.55, f"{name} did not beat baseline: {mean_auc:.3f}"


def test_majority_baseline_is_chance(clean):
    out = exp.sanity_majority_baseline(clean.X, clean.y)
    assert abs(out["roc_auc_mean"] - 0.5) < 0.05


def test_label_shuffle_collapses_to_chance(clean):
    """Shuffled labels => no signal => AUC ~ 0.5. Guards against leakage."""
    out = exp.sanity_label_shuffle(clean.X, clean.y)
    assert abs(out["roc_auc_mean"] - 0.5) < 0.08


def test_overfit_tiny_slice(clean):
    out = exp.sanity_overfit_tiny(clean.X, clean.y)
    assert out["train_roc_auc"] > 0.95


def test_leakage_ceiling_is_near_perfect(clean):
    """account_status must be a leak: including it pushes AUC to ~1.0."""
    X_leak, y = data_mod.load_with_leak(str(CSV))
    out = exp.sanity_leakage_ceiling(X_leak, y)
    assert out["roc_auc_mean"] > 0.99


def test_determinism(clean):
    """Same seed, same data => identical metrics. No hidden nondeterminism."""
    a = exp.evaluate_arms(clean.X, clean.y, exp.make_models(7), n_splits=exp.N_SPLITS)
    b = exp.evaluate_arms(clean.X, clean.y, exp.make_models(7), n_splits=exp.N_SPLITS)
    for name in a:
        assert a[name].per_fold == b[name].per_fold


def test_paired_delta_shape(clean):
    arms = exp.evaluate_arms(clean.X, clean.y, exp.make_models(), n_splits=exp.N_SPLITS)
    d = exp.paired_delta(arms["logistic_regression"], arms["gradient_boosting"])
    assert d["n"] == exp.N_SPLITS
    assert "p_value" in d and 0.0 <= d["p_value"] <= 1.0
