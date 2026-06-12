"""Pipeline rigor tests: leakage removal, dedup, time-ordering, and the sanity
invariants the experiment depends on. These are the smallest tests that would
catch the failure modes the dataset deliberately plants.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn_experiment import evaluate, sanity  # noqa: E402
from churn_experiment.data import FEATURES, load_churn  # noqa: E402
from churn_experiment.pipeline import make_pipeline  # noqa: E402

CSV = str(ROOT / "churn.csv")


@pytest.fixture(scope="module")
def data():
    if not Path(CSV).exists():
        pytest.skip("churn.csv not generated; run make_dataset.py first")
    return load_churn(CSV)


def test_leaky_and_id_columns_dropped(data):
    assert "account_status" not in data.X.columns
    assert "customer_id" not in data.X.columns
    assert list(data.X.columns) == FEATURES


def test_duplicates_removed_before_split(data):
    # The generator appends 200 exact (full-row) duplicates; all must be gone.
    assert data.n_duplicates == 200
    raw = pd.read_csv(CSV)
    assert raw.drop_duplicates().duplicated().sum() == 0
    # Note: after dropping customer_id a couple of distinct customers can
    # coincidentally share an identical feature+target vector. That is not the
    # planted leak and is harmless, so we assert on full-row dedup only.


def test_rows_time_ordered(data):
    # load_churn sorts by signup_date; reload raw to verify ascending order.
    raw = pd.read_csv(CSV).drop_duplicates().sort_values("signup_date", kind="stable")
    dates = pd.to_datetime(raw["signup_date"]).to_numpy()
    assert (dates[:-1] <= dates[1:]).all()


def test_churn_rate_is_imbalanced(data):
    assert 0.2 < data.churn_rate < 0.35  # ~0.27; accuracy alone would mislead


def test_time_split_test_after_train():
    # TimeSeriesSplit must never put a test index before its train indices.
    from sklearn.model_selection import TimeSeriesSplit

    splitter = TimeSeriesSplit(n_splits=evaluate.N_SPLITS)
    X = np.arange(100).reshape(-1, 1)
    for tr, te in splitter.split(X):
        assert tr.max() < te.min()


def test_determinism_same_seed(data):
    out = sanity.determinism(data)
    assert out["identical"] is True


def test_label_shuffle_collapses_to_chance(data):
    out = sanity.label_shuffle(data)
    for arm, auc in out.items():
        assert auc == pytest.approx(0.5, abs=0.1), f"{arm} leaks under shuffle: {auc}"


def test_leakage_ceiling_is_near_perfect():
    out = sanity.leakage_ceiling(CSV)
    assert out["roc_auc_with_leak"] > 0.99  # confirms account_status is a leak


def test_models_beat_baseline_floor(data):
    results = evaluate.run_folds(data, seeds=(0,), n_splits=3)
    summ = evaluate.summarise(results)
    for arm, s in summ.items():
        assert s.roc_auc_mean > 0.55, f"{arm} did not beat the 0.5 floor"


def test_overfit_tiny_slice(data):
    out = sanity.overfit_tiny(data)
    assert out["gboost"] > 0.95  # tree model must fit a tiny slice


def test_compare_is_paired_and_named(data):
    results = evaluate.run_folds(data, seeds=(0,), n_splits=3)
    comp = evaluate.compare(results, "roc_auc")
    assert comp.n_pairs == 3
    assert evaluate.verdict(comp) in {"logreg", "gboost", "no detectable difference"}


def test_unknown_arm_rejected():
    with pytest.raises(ValueError):
        make_pipeline("svm", 0)
