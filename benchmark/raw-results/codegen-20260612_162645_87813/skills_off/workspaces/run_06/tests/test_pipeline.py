"""Tests for the modeling pipeline: sanity checks and reproducibility."""
from churn_experiment import config
from churn_experiment.data import chronological_split, deduplicate, to_xy
from churn_experiment.evaluation import (
    baseline_floor,
    cross_validated_scores,
    label_shuffle_test,
    leakage_demonstration,
    paired_difference,
)
from churn_experiment.experiment import run


def _train_xy(raw_df):
    train_df, _ = chronological_split(deduplicate(raw_df))
    return to_xy(train_df)


def test_models_beat_baseline(raw_df):
    X, y = _train_xy(raw_df)
    base = baseline_floor(X, y)["roc_auc_mean"]
    assert abs(base - 0.5) < 0.05, "prior baseline must sit at chance"
    cv = cross_validated_scores(X, y)
    for name, s in cv.items():
        assert s.roc_auc_mean > base + 0.03, f"{name} must beat the floor"


def test_label_shuffle_destroys_signal(raw_df):
    X, y = _train_xy(raw_df)
    auc = label_shuffle_test(X, y)["roc_auc_mean"]
    assert abs(auc - 0.5) < 0.08, "shuffled labels must collapse to chance"


def test_leakage_demonstration_is_near_perfect(raw_df):
    train_df, test_df = chronological_split(deduplicate(raw_df))
    auc = leakage_demonstration(train_df, test_df)["holdout_roc_auc"]
    assert auc > 0.98, "including account_status must reveal the leak (AUC->1)"


def test_real_models_are_not_near_perfect(raw_df):
    # Leakage ceiling: an honest model on a noisy task must NOT be ~perfect.
    X, y = _train_xy(raw_df)
    cv = cross_validated_scores(X, y)
    for name, s in cv.items():
        assert s.roc_auc_mean < 0.95, f"{name} AUC suspiciously high — audit for leakage"


def test_reproducible_same_seed(raw_df):
    X, y = _train_xy(raw_df)
    a = cross_validated_scores(X, y, seed=config.SEED)
    b = cross_validated_scores(X, y, seed=config.SEED)
    for name in a:
        assert a[name].roc_auc_per_fold == b[name].roc_auc_per_fold


def test_paired_difference_verdict_is_valid(raw_df):
    X, y = _train_xy(raw_df)
    cv = cross_validated_scores(X, y)
    d = paired_difference(cv)
    assert d["verdict"] in {
        "no detectable difference",
        "gradient_boosting better",
        "logistic_regression better",
    }
    assert d["ci95_low"] <= d["mean_diff"] <= d["ci95_high"]


def test_end_to_end_run_structure(churn_csv):
    results = run(churn_csv)
    assert results["data_audit"]["n_duplicate_rows"] == 200
    assert results["split_sizes"]["train"] > results["split_sizes"]["test"]
    assert "logistic_regression" in results["final_holdout"]
    assert "gradient_boosting" in results["final_holdout"]
    for m in results["final_holdout"].values():
        assert 0.0 <= m["roc_auc"] <= 1.0
