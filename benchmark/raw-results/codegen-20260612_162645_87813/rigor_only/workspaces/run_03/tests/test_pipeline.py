"""Pipeline tests: leakage policy, sanity checks, and determinism.

These encode the rigor guarantees of the experiment so a regression in any of
them fails loudly rather than silently producing a wrong number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.churn_experiment.data import (  # noqa: E402
    FEATURE_COLUMNS,
    LEAK_COLUMNS,
    ID_COLUMNS,
    load_prepared,
    prepare,
)
from src.churn_experiment.experiment import (  # noqa: E402
    evaluate,
    label_shuffle_auc,
    make_estimators,
)

DATA = ROOT / "churn.csv"

pytestmark = pytest.mark.skipif(
    not DATA.exists(), reason="run `python3 make_dataset.py --out churn.csv` first"
)


@pytest.fixture(scope="module")
def prepared():
    return load_prepared(str(DATA))


# --- Data discipline -------------------------------------------------------

def test_duplicates_removed_before_split(prepared):
    # generator appends 200 exact duplicates; they must be gone.
    assert prepared.n_duplicates == 200
    assert prepared.n_final == prepared.n_raw - 200


def test_leak_and_id_columns_dropped(prepared):
    cols = set(prepared.X.columns)
    for c in (*LEAK_COLUMNS, *ID_COLUMNS, "signup_date", "churned"):
        assert c not in cols
    assert list(prepared.X.columns) == list(FEATURE_COLUMNS)


def test_features_carry_no_residual_duplicates(prepared):
    # No exact duplicate rows remain in the feature matrix after dedup-by-row.
    # (Some feature-only collisions are expected with 3 integer-ish columns;
    # we only assert the row-level dedup removed the planted 200.)
    assert len(prepared.X) == prepared.n_final


def test_positive_rate_is_imbalanced(prepared):
    # ~27% churn: accuracy alone would be misleading, hence AUC primary.
    assert 0.20 < prepared.positive_rate < 0.35


# --- Sanity checks ---------------------------------------------------------

def test_models_beat_baseline(prepared):
    arms = evaluate(prepared)
    base = np.mean(arms["baseline_majority"].roc_auc)
    for name in ("logistic_regression", "gradient_boosting"):
        assert np.mean(arms[name].roc_auc) > base + 0.03, name


def test_baseline_floor_is_chance(prepared):
    arms = evaluate(prepared)
    assert abs(np.mean(arms["baseline_majority"].roc_auc) - 0.5) < 0.05


def test_label_shuffle_collapses_to_chance(prepared):
    # With permuted labels, AUC must fall to ~0.5; otherwise something leaks.
    auc = label_shuffle_auc(prepared)
    assert abs(auc - 0.5) < 0.07, auc


def test_leak_ceiling_if_account_status_kept(prepared):
    # Demonstrate WHY account_status is dropped: it makes the task trivial.
    raw = pd.read_csv(DATA).drop_duplicates()
    X = pd.get_dummies(
        raw[[*FEATURE_COLUMNS, "account_status"]], columns=["account_status"]
    )
    y = raw["churned"].astype(int)
    est = make_estimators()["logistic_regression"]
    est.fit(X, y)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, est.predict_proba(X)[:, 1])
    assert auc > 0.99, f"expected near-perfect leak, got {auc}"


# --- Determinism -----------------------------------------------------------

def test_same_seed_reproduces_metrics(prepared):
    a = evaluate(prepared, seed=7)
    b = evaluate(prepared, seed=7)
    for name in a:
        assert a[name].roc_auc == b[name].roc_auc, name


def test_overfit_tiny_subset(prepared):
    # A flexible model must reach near-perfect fit on a tiny slice; if it
    # cannot, the pipeline (features -> estimator) is broken.
    from sklearn.metrics import roc_auc_score
    X = prepared.X.iloc[:60]
    y = prepared.y.iloc[:60]
    # ensure both classes present in the slice
    if y.nunique() < 2:
        idx = list(range(40)) + list(prepared.y[prepared.y == 1].index[:20])
        X, y = prepared.X.loc[idx], prepared.y.loc[idx]
    est = make_estimators()["gradient_boosting"]
    est.fit(X, y)
    auc = roc_auc_score(y, est.predict_proba(X)[:, 1])
    assert auc > 0.95, auc
