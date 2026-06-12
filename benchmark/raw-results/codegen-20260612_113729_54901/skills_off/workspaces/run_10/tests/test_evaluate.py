import numpy as np
import pytest

from src.evaluate import CV_METRICS, baseline_scores, cv_scores_multi_seed, holdout_scores
from src.models import make_logistic_regression


@pytest.fixture
def split_data():
    rng = np.random.default_rng(99)
    X = rng.standard_normal((250, 3))
    y = (X[:, 0] - 0.5 * X[:, 1] + rng.standard_normal(250) * 0.3 > 0).astype(int)
    return X[:200], X[200:], y[:200], y[200:]


def test_cv_returns_all_metrics(split_data):
    X_train, _, y_train, _ = split_data
    scores = cv_scores_multi_seed(make_logistic_regression, X_train, y_train, seeds=(42,), n_splits=3)
    for m in CV_METRICS:
        assert m in scores


def test_cv_result_keys(split_data):
    X_train, _, y_train, _ = split_data
    scores = cv_scores_multi_seed(make_logistic_regression, X_train, y_train, seeds=(42,), n_splits=3)
    for m in CV_METRICS:
        assert "mean" in scores[m]
        assert "std" in scores[m]
        assert "n" in scores[m]
        assert "values" in scores[m]


def test_cv_n_equals_seeds_times_folds(split_data):
    X_train, _, y_train, _ = split_data
    scores = cv_scores_multi_seed(
        make_logistic_regression, X_train, y_train, seeds=(1, 2), n_splits=4
    )
    assert scores["roc_auc"]["n"] == 2 * 4


def test_cv_values_length_matches_n(split_data):
    X_train, _, y_train, _ = split_data
    scores = cv_scores_multi_seed(
        make_logistic_regression, X_train, y_train, seeds=(10, 20, 30), n_splits=5
    )
    assert len(scores["roc_auc"]["values"]) == scores["roc_auc"]["n"]


def test_cv_roc_auc_range(split_data):
    X_train, _, y_train, _ = split_data
    scores = cv_scores_multi_seed(make_logistic_regression, X_train, y_train, seeds=(42,), n_splits=3)
    assert 0.0 <= scores["roc_auc"]["mean"] <= 1.0


def test_holdout_scores_range(split_data):
    X_train, X_test, y_train, y_test = split_data
    model = make_logistic_regression(seed=42)
    scores = holdout_scores(model, X_train, y_train, X_test, y_test)
    for key in ["roc_auc", "f1", "precision", "recall"]:
        assert key in scores
        assert 0.0 <= scores[key] <= 1.0


def test_baseline_roc_auc_is_point_five(split_data):
    X_train, X_test, y_train, y_test = split_data
    scores = baseline_scores(X_train, y_train, X_test, y_test)
    assert scores["roc_auc"] == 0.5


def test_model_beats_baseline_on_learnable_data(split_data):
    X_train, X_test, y_train, y_test = split_data
    model = make_logistic_regression(seed=42)
    model_scores = holdout_scores(model, X_train, y_train, X_test, y_test)
    base = baseline_scores(X_train, y_train, X_test, y_test)
    assert model_scores["roc_auc"] > base["roc_auc"]
