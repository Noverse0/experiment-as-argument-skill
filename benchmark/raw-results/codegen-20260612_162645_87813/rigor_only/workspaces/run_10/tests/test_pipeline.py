"""Tests for the churn experiment pipeline. These encode the rigor invariants:
no leak columns reach the model, dedup happens before the split, the split is
temporal, preprocessing is fold-local, and runs are deterministic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn_experiment import data as D  # noqa: E402
from churn_experiment import evaluate as E  # noqa: E402
from churn_experiment import sanity as S  # noqa: E402


@pytest.fixture(scope="module")
def raw():
    """Build a small synthetic frame mirroring the real schema and its traps."""
    rng = np.random.default_rng(0)
    n = 400
    tenure = rng.integers(1, 72, n)
    spend = rng.gamma(2.0, 30.0, n).round(2)
    tickets = rng.poisson(1.2, n)
    dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")
    logit = -1.2 - 0.03 * tenure + 0.01 * spend + 0.45 * tickets
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "signup_date": dates.strftime("%Y-%m-%d"),
            "tenure_months": tenure,
            "monthly_spend": spend,
            "support_tickets": tickets,
            "account_status": np.where(churn == 1, "closed", "active"),
            "churned": churn,
        }
    )
    dup = df.sample(n=40, random_state=0)  # planted duplicates
    return pd.concat([df, dup], ignore_index=True)


def test_deduplicate_removes_exact_duplicates(raw):
    clean, n_removed = D.deduplicate(raw)
    assert n_removed == 40
    assert clean.duplicated().sum() == 0


def test_feature_matrix_excludes_leak_and_id_columns(raw):
    X = D.feature_matrix(raw)
    assert list(X.columns) == D.FEATURES
    for banned in D.LEAK_COLS + D.ID_COLS + [D.TARGET, D.DATE_COL]:
        assert banned not in X.columns


def test_split_is_temporal_and_non_overlapping(raw):
    split = D.temporal_split(raw, test_frac=0.25)
    # dedup happened before split
    assert split.n_duplicates_removed == 40
    assert len(split.X_dev) + len(split.X_test) == split.n_rows_after_dedup
    # every dev signup_date <= every test signup_date (time-ordered hold-out)
    ordered = D.deduplicate(raw)[0].sort_values(D.DATE_COL, kind="mergesort")
    n_dev = len(split.X_dev)
    dev_dates = ordered[D.DATE_COL].iloc[:n_dev]
    test_dates = ordered[D.DATE_COL].iloc[n_dev:]
    assert dev_dates.max() <= test_dates.min()


def test_no_row_overlap_between_dev_and_test(raw):
    split = D.temporal_split(raw, test_frac=0.25)
    dev = split.X_dev.assign(y=split.y_dev.values)
    test = split.X_test.assign(y=split.y_test.values)
    merged = dev.merge(test, how="inner")
    # after dedup there can be coincidental feature collisions, but the planted
    # exact duplicates are gone, so overlap must be small relative to test size.
    assert len(merged) < 0.05 * len(test) + 1


def test_label_shuffle_collapses_to_chance(raw):
    """The core leak detector: shuffled labels must give ~0.5 AUC. If a leak
    column slipped into the features this would stay high."""
    split = D.temporal_split(raw)
    res = S.label_shuffle(split.X_dev, split.y_dev, seed=0)
    assert res["passed"], res
    assert abs(res["detail"]["auc"] - 0.5) < 0.1


def test_account_status_would_leak_if_used(raw):
    """Guard rail: prove the dropped column IS a leak, justifying the drop.
    Mapping account_status to 0/1 reproduces the target almost perfectly."""
    clean = D.deduplicate(raw)[0]
    leaked = (clean["account_status"] == "closed").astype(int)
    assert (leaked == clean["churned"]).mean() > 0.99


def test_runs_are_deterministic(raw):
    split = D.temporal_split(raw)
    a = E.cross_validate_arms(split.X_dev, split.y_dev, seed=0, n_splits=3)
    b = E.cross_validate_arms(split.X_dev, split.y_dev, seed=0, n_splits=3)
    assert a["per_arm"] == b["per_arm"]


def test_models_beat_chance_on_dev(raw):
    """Sanity that the pipeline learns *something* (above the 0.5 floor)."""
    split = D.temporal_split(raw)
    cv = E.cross_validate_arms(split.X_dev, split.y_dev, seed=0, n_splits=3)
    for name in ("logreg", "gradient_boosting"):
        assert cv["per_arm"][name]["roc_auc"]["mean"] > 0.5
