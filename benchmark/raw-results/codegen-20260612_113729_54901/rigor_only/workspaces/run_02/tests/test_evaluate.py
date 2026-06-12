import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.evaluate import evaluate_model, summarize


@pytest.fixture
def separable_data():
    rng = np.random.default_rng(0)
    n = 300
    X = rng.standard_normal((n, 3))
    y = (X[:, 0] + rng.standard_normal(n) * 0.3 > 0).astype(int)
    return X, y


def test_n_records_matches_folds_times_seeds(separable_data):
    X, y = separable_data
    records = evaluate_model(
        LogisticRegression(max_iter=1000), X, y, n_splits=3, seeds=[0, 1]
    )
    assert len(records) == 6  # 3 folds × 2 seeds


def test_records_have_required_keys(separable_data):
    X, y = separable_data
    records = evaluate_model(
        LogisticRegression(max_iter=1000), X, y, n_splits=2, seeds=[0]
    )
    for r in records:
        for key in ("seed", "fold", "roc_auc", "f1", "avg_precision"):
            assert key in r


def test_auc_in_unit_interval(separable_data):
    X, y = separable_data
    records = evaluate_model(
        LogisticRegression(max_iter=1000), X, y, n_splits=2, seeds=[0]
    )
    for r in records:
        assert 0.0 <= r["roc_auc"] <= 1.0
        assert 0.0 <= r["f1"] <= 1.0
        assert 0.0 <= r["avg_precision"] <= 1.0


def test_separable_data_gets_high_auc(separable_data):
    X, y = separable_data
    records = evaluate_model(
        LogisticRegression(max_iter=1000), X, y, n_splits=3, seeds=[0]
    )
    mean_auc = np.mean([r["roc_auc"] for r in records])
    assert mean_auc > 0.85, f"Expected high AUC on separable data, got {mean_auc:.3f}"


def test_summarize_keys(separable_data):
    X, y = separable_data
    records = evaluate_model(
        LogisticRegression(max_iter=1000), X, y, n_splits=2, seeds=[0]
    )
    s = summarize(records)
    for key in (
        "roc_auc_mean", "roc_auc_std",
        "f1_mean", "f1_std",
        "avg_precision_mean", "avg_precision_std",
        "n_folds",
    ):
        assert key in s


def test_summarize_n_folds(separable_data):
    X, y = separable_data
    records = evaluate_model(
        LogisticRegression(max_iter=1000), X, y, n_splits=4, seeds=[0, 1, 2]
    )
    s = summarize(records)
    assert s["n_folds"] == 12  # 4 × 3
