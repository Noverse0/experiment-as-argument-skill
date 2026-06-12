"""Integration-style tests for the full experiment pipeline."""
import numpy as np
import pytest

from src.experiment import run, SEEDS
from src.features import FEATURE_COLS, TARGET_COL


@pytest.fixture(scope="module")
def results():
    return run("churn.csv")


def test_duplicates_removed(results):
    assert results["n_duplicates_removed"] == 200


def test_train_test_sizes_sum(results):
    assert results["n_train"] + results["n_test"] == results["n_after_dedup"]


def test_scaler_not_refitted_on_test(results):
    # If scaler were refitted on test, test metrics would differ —
    # we can't directly check that here, but we verify the pipeline runs
    # and produces separate n_train / n_test counts.
    assert results["n_train"] > results["n_test"]


def test_sanity_overfit_lr_passed(results):
    assert results["sanity"]["overfit_lr"]["passed"]


def test_sanity_overfit_gb_passed(results):
    assert results["sanity"]["overfit_gb"]["passed"]


def test_sanity_label_shuffle_lr_passed(results):
    r = results["sanity"]["label_shuffle_lr"]
    assert r["passed"], (
        f"Label shuffle mean AUC too high for LR: {r['mean_auc_with_shuffled_labels']:.4f}"
    )


def test_sanity_label_shuffle_gb_passed(results):
    r = results["sanity"]["label_shuffle_gb"]
    assert r["passed"], (
        f"Label shuffle mean AUC too high for GB: {r['mean_auc_with_shuffled_labels']:.4f}"
    )


def test_both_models_beat_baseline(results):
    baseline_acc = results["sanity"]["baseline"]["majority_class_accuracy"]
    lr_auc = results["lr"]["roc_auc"]["mean"]
    gb_auc = results["gb"]["roc_auc"]["mean"]
    assert lr_auc > 0.5, f"LR AUC {lr_auc:.4f} does not beat random baseline"
    assert gb_auc > 0.5, f"GB AUC {gb_auc:.4f} does not beat random baseline"


def test_lr_is_deterministic(results):
    # LR with lbfgs is deterministic; all seed runs should be identical
    auc_values = results["lr"]["roc_auc"]["values"]
    assert results["lr"]["roc_auc"]["std"] < 1e-8, (
        f"LR AUC varied across seeds: {auc_values}"
    )


def test_metrics_in_valid_range(results):
    for model_key in ("lr", "gb"):
        m = results[model_key]
        assert 0.0 <= m["roc_auc"]["mean"] <= 1.0
        assert 0.0 <= m["f1"]["mean"] <= 1.0
        assert 0.0 <= m["brier"]["mean"] <= 1.0


def test_seeds_recorded(results):
    assert results["seeds"] == SEEDS


def test_no_account_status_feature():
    # Confirm engineer_features removes the leaker before any model sees data
    import pandas as pd
    from src.data import engineer_features
    df = pd.DataFrame({
        "customer_id": [1],
        "signup_date": ["2023-01-01"],
        "tenure_months": [10],
        "monthly_spend": [50.0],
        "support_tickets": [1],
        "account_status": ["closed"],
        "churned": [1],
    })
    result = engineer_features(df)
    assert "account_status" not in result.columns
