"""Experiment runner: sanity checks and multi-seed model comparison."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from .models import MODEL_REGISTRY, make_baseline

SEEDS = [0, 1, 2, 3, 4]


def _eval_one(model, X_train, X_test, y_train, y_test) -> dict:
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    return {
        "auc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }


def compute_sanity(X_train, X_test, y_train, y_test) -> dict:
    """Sanity checks that must pass before trusting model results.

    1. Baseline AUC floor: constant predictor → AUC ≈ 0.5.
    2. Label-shuffle test: fitting on permuted labels → AUC ≈ 0.5.
       If this is significantly > 0.5, features carry spurious signal.
    """
    from sklearn.linear_model import LogisticRegression

    # Baseline
    baseline = make_baseline()
    bl = _eval_one(baseline, X_train, X_test, y_train, y_test)

    # Label shuffle: permute y_train, fit LR, measure AUC on real y_test
    rng = np.random.RandomState(42)
    y_shuffled = rng.permutation(y_train)
    lr_shuffle = LogisticRegression(random_state=42, max_iter=500)
    lr_shuffle.fit(X_train, y_shuffled)
    y_proba_shuf = lr_shuffle.predict_proba(X_test)[:, 1]
    shuffle_auc = float(roc_auc_score(y_test, y_proba_shuf))

    shuffle_passes = abs(shuffle_auc - 0.5) < 0.05

    return {
        "baseline_auc": bl["auc"],
        "baseline_f1": bl["f1"],
        "label_shuffle_auc": shuffle_auc,
        "shuffle_test_passes": shuffle_passes,
    }


def run_comparison(
    X_train,
    X_test,
    y_train,
    y_test,
    seeds: list[int] = SEEDS,
) -> dict:
    """Run each model over multiple seeds and collect AUC / F1 statistics."""
    results: dict = {}

    for name, make_fn in MODEL_REGISTRY.items():
        auc_scores: list[float] = []
        f1_scores: list[float] = []
        for seed in seeds:
            m = _eval_one(make_fn(seed), X_train, X_test, y_train, y_test)
            auc_scores.append(m["auc"])
            f1_scores.append(m["f1"])

        results[name] = {
            "auc_mean": float(np.mean(auc_scores)),
            "auc_std": float(np.std(auc_scores, ddof=0)),
            "f1_mean": float(np.mean(f1_scores)),
            "f1_std": float(np.std(f1_scores, ddof=0)),
            "auc_per_seed": auc_scores,
            "f1_per_seed": f1_scores,
            "n_seeds": len(seeds),
            "seeds": seeds,
        }

    lr = results["logistic_regression"]
    gb = results["gradient_boosting"]
    auc_gap = gb["auc_mean"] - lr["auc_mean"]
    # Detectable difference: gap must exceed combined spread (sum of std)
    spread = lr["auc_std"] + gb["auc_std"]
    min_detectable = max(spread, 0.01)

    if abs(auc_gap) > min_detectable:
        conclusion = "gb_better" if auc_gap > 0 else "lr_better"
    else:
        conclusion = "no_detectable_difference"

    results["comparison"] = {
        "auc_gap": float(auc_gap),
        "combined_spread": float(spread),
        "conclusion": conclusion,
    }

    return results
