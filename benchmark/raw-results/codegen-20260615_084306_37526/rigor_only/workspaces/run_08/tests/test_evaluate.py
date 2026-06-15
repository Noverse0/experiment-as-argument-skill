"""Tests for cross-validation evaluation logic."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.evaluate import evaluate_models, paired_ttest


def _simple_models():
    lr = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(random_state=0))])
    return {"LR": lr}


def _classification_data(n=300):
    X, y = make_classification(n_samples=n, n_features=4, random_state=42)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(4)]), y


class TestEvaluateModels:
    def test_returns_expected_keys(self):
        X, y = _classification_data()
        results = evaluate_models(X, y, _simple_models(), n_splits=3, seeds=[0])
        assert "LR" in results
        for key in ("roc_auc_mean", "roc_auc_std", "roc_auc_all",
                    "f1_mean", "f1_std", "f1_all", "n_evaluations"):
            assert key in results["LR"], f"Missing key: {key}"

    def test_metric_ranges(self):
        X, y = _classification_data()
        results = evaluate_models(X, y, _simple_models(), n_splits=3, seeds=[0])
        r = results["LR"]
        assert 0.0 <= r["roc_auc_mean"] <= 1.0
        assert 0.0 <= r["f1_mean"] <= 1.0
        assert r["roc_auc_std"] >= 0.0

    def test_n_evaluations_equals_splits_times_seeds(self):
        X, y = _classification_data()
        results = evaluate_models(X, y, _simple_models(), n_splits=4, seeds=[0, 1, 2])
        assert results["LR"]["n_evaluations"] == 12  # 4 folds * 3 seeds

    def test_all_scores_length_matches_n_evaluations(self):
        X, y = _classification_data()
        results = evaluate_models(X, y, _simple_models(), n_splits=3, seeds=[0, 1])
        r = results["LR"]
        assert len(r["roc_auc_all"]) == r["n_evaluations"]
        assert len(r["f1_all"]) == r["n_evaluations"]

    def test_mean_consistent_with_all_scores(self):
        X, y = _classification_data()
        results = evaluate_models(X, y, _simple_models(), n_splits=3, seeds=[0])
        r = results["LR"]
        assert abs(np.mean(r["roc_auc_all"]) - r["roc_auc_mean"]) < 1e-6

    def test_default_seeds_used_when_none(self):
        X, y = _classification_data()
        results = evaluate_models(X, y, _simple_models(), n_splits=3)
        # Default is seeds=[0,1,2] → 9 evaluations
        assert results["LR"]["n_evaluations"] == 9


class TestPairedTtest:
    def test_identical_scores_return_p1(self):
        a = [0.7, 0.72, 0.68, 0.71, 0.69]
        _, p = paired_ttest(a, a)
        assert p == 1.0

    def test_clearly_different_scores_significant(self):
        a = [0.60] * 25
        b = [0.80] * 25
        _, p = paired_ttest(a, b)
        assert p < 0.001

    def test_direction_of_t_statistic(self):
        a = [0.70] * 10
        b = [0.75] * 10
        t, _ = paired_ttest(a, b)
        assert t > 0  # b > a → positive t

    def test_returns_two_floats(self):
        a = [0.7, 0.72, 0.68]
        b = [0.71, 0.73, 0.70]
        result = paired_ttest(a, b)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)
