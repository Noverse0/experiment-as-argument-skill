"""Tests for the evaluation and comparison pipeline."""
import numpy as np
import pytest

from src.evaluate import compute_sanity, run_comparison


@pytest.fixture
def small_split():
    """Deterministic small dataset split for testing evaluation functions."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((200, 4))
    # Weak but real signal: first feature predicts y
    logit = -0.5 + 1.5 * X_train[:, 0]
    y_train = (rng.random(200) < 1 / (1 + np.exp(-logit))).astype(int)

    X_test = rng.standard_normal((80, 4))
    logit_test = -0.5 + 1.5 * X_test[:, 0]
    y_test = (rng.random(80) < 1 / (1 + np.exp(-logit_test))).astype(int)

    return X_train, X_test, y_train, y_test


def test_run_comparison_keys(small_split):
    X_train, X_test, y_train, y_test = small_split
    results = run_comparison(X_train, X_test, y_train, y_test, seeds=[0, 1])
    assert "logistic_regression" in results
    assert "gradient_boosting" in results
    assert "comparison" in results


def test_auc_values_in_range(small_split):
    X_train, X_test, y_train, y_test = small_split
    results = run_comparison(X_train, X_test, y_train, y_test, seeds=[0, 1, 2])
    for name in ("logistic_regression", "gradient_boosting"):
        auc = results[name]["auc_mean"]
        assert 0.0 <= auc <= 1.0, f"{name} AUC out of range: {auc}"
        for seed_auc in results[name]["auc_per_seed"]:
            assert 0.0 <= seed_auc <= 1.0


def test_f1_values_in_range(small_split):
    X_train, X_test, y_train, y_test = small_split
    results = run_comparison(X_train, X_test, y_train, y_test, seeds=[0])
    for name in ("logistic_regression", "gradient_boosting"):
        f1 = results[name]["f1_mean"]
        assert 0.0 <= f1 <= 1.0, f"{name} F1 out of range: {f1}"


def test_n_seeds_matches(small_split):
    X_train, X_test, y_train, y_test = small_split
    seeds = [0, 1, 2]
    results = run_comparison(X_train, X_test, y_train, y_test, seeds=seeds)
    for name in ("logistic_regression", "gradient_boosting"):
        assert results[name]["n_seeds"] == len(seeds)
        assert len(results[name]["auc_per_seed"]) == len(seeds)


def test_conclusion_is_valid_string(small_split):
    X_train, X_test, y_train, y_test = small_split
    results = run_comparison(X_train, X_test, y_train, y_test, seeds=[0, 1])
    valid = {"gb_better", "lr_better", "no_detectable_difference"}
    assert results["comparison"]["conclusion"] in valid


def test_std_non_negative(small_split):
    X_train, X_test, y_train, y_test = small_split
    results = run_comparison(X_train, X_test, y_train, y_test, seeds=[0, 1, 2])
    for name in ("logistic_regression", "gradient_boosting"):
        assert results[name]["auc_std"] >= 0
        assert results[name]["f1_std"] >= 0


def test_compute_sanity_keys(small_split):
    X_train, X_test, y_train, y_test = small_split
    sanity = compute_sanity(X_train, X_test, y_train, y_test)
    for key in ("baseline_auc", "label_shuffle_auc", "shuffle_test_passes"):
        assert key in sanity


def test_label_shuffle_auc_near_chance(small_split):
    X_train, X_test, y_train, y_test = small_split
    sanity = compute_sanity(X_train, X_test, y_train, y_test)
    # With shuffled labels the AUC should be near 0.5. We allow 0.25 tolerance
    # because the tiny test dataset (80 test samples) has high variance.
    assert abs(sanity["label_shuffle_auc"] - 0.5) < 0.25


def test_models_beat_baseline(small_split):
    """Models should outperform the majority-class floor AUC of ~0.5."""
    X_train, X_test, y_train, y_test = small_split
    results = run_comparison(X_train, X_test, y_train, y_test, seeds=[0])
    sanity = compute_sanity(X_train, X_test, y_train, y_test)
    for name in ("logistic_regression", "gradient_boosting"):
        assert results[name]["auc_mean"] > sanity["baseline_auc"], (
            f"{name} AUC {results[name]['auc_mean']:.4f} should exceed "
            f"baseline {sanity['baseline_auc']:.4f}"
        )
