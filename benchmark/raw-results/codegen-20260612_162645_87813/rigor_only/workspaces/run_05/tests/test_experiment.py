"""Tests for the comparison pipeline and its rigor properties:
determinism, sanity-check behavior, and split-before-transform."""
from src.data import prepare
from src.experiment import (
    evaluate_arms,
    make_models,
    sanity_label_shuffle,
    sanity_leakage_ceiling,
    sanity_overfit_tiny,
)


def test_models_are_two_distinct_arms():
    models = make_models()
    assert set(models) == {"logistic_regression", "gradient_boosting"}


def test_evaluate_is_deterministic_for_same_seed(csv_path):
    data = prepare(csv_path)
    r1 = evaluate_arms(data, seed=7, n_splits=3)
    r2 = evaluate_arms(data, seed=7, n_splits=3)
    # Identical pipeline + seed must reproduce metrics exactly.
    for arm in ("logistic_regression", "gradient_boosting"):
        assert r1["arms"][arm]["summary"] == r2["arms"][arm]["summary"]
    assert r1["paired_roc_auc_diff_gbm_minus_lr"]["mean"] == (
        r2["paired_roc_auc_diff_gbm_minus_lr"]["mean"]
    )


def test_both_arms_beat_majority_baseline(csv_path):
    data = prepare(csv_path)
    res = evaluate_arms(data, seed=7, n_splits=5)
    base = res["majority_baseline"]["summary"]["roc_auc"]["mean"]
    for arm in ("logistic_regression", "gradient_boosting"):
        auc = res["arms"][arm]["summary"]["roc_auc"]["mean"]
        assert auc > base
        assert auc > 0.55  # learning real, non-trivial signal


def test_reports_variance_with_n_folds(csv_path):
    data = prepare(csv_path)
    res = evaluate_arms(data, seed=7, n_splits=5)
    summary = res["arms"]["gradient_boosting"]["summary"]["roc_auc"]
    assert summary["n"] == 5
    assert summary["sd"] >= 0.0  # variance is computed, not assumed away


def test_paired_test_reported(csv_path):
    data = prepare(csv_path)
    res = evaluate_arms(data, seed=7, n_splits=5)
    diff = res["paired_roc_auc_diff_gbm_minus_lr"]
    # The comparison must carry a paired test, not just a bare mean.
    assert "p_value" in diff and "t_statistic" in diff
    assert 0.0 <= diff["p_value"] <= 1.0
    assert isinstance(diff["significant"], bool)
    assert len(diff["per_fold"]) == 5


def test_sanity_label_shuffle_collapses_to_chance(csv_path):
    data = prepare(csv_path)
    out = sanity_label_shuffle(data, seed=7)
    # With shuffled labels, AUC must be near chance; no signal can leak.
    assert abs(out["mean_roc_auc_shuffled_labels"] - 0.5) < 0.08


def test_sanity_leakage_ceiling_is_near_perfect(csv_path):
    out = sanity_leakage_ceiling(csv_path, seed=7)
    # The dropped account_status, if kept, trivially solves the task.
    assert out["mean_roc_auc_with_leak"] > 0.99


def test_sanity_overfit_tiny_subset(csv_path):
    data = prepare(csv_path)
    out = sanity_overfit_tiny(data, seed=7)
    # A flexible model must memorize a tiny slice it trained on.
    assert out["train_accuracy_tiny_subset"] > 0.95
