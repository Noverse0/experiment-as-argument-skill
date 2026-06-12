"""Cheap sanity checks that run before we believe any comparison.

These catch the failure modes that silently turn a result into noise: leakage
(scores too good), a broken pipeline (can't fit a tiny slice), and information
leaking around the labels (shuffled labels still predictable).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from .data import Dataset

# A noisy churn task should land well below a perfect AUC. If a model scores
# above this on held-out data, suspect leakage and audit features.
LEAKAGE_CEILING = 0.95


@dataclass
class SanityReport:
    checks: dict[str, dict]

    @property
    def all_passed(self) -> bool:
        return all(c["passed"] for c in self.checks.values())


def baseline_auc(dataset: Dataset) -> float:
    """AUC of a constant predictor is 0.5 by construction — the floor."""
    return 0.5


def label_shuffle_auc(
    model, dataset: Dataset, seed: int, n_repeats: int = 20
) -> float:
    """Train on shuffled labels; mean held-out AUC should fall to ~0.5.

    If a shuffled-label model still predicts well, information is leaking around
    the labels (e.g. duplicates, or a feature encoding the target).

    A *single* shuffle is a noisy estimator: with only a few hundred rows the
    fitted coefficients pick an essentially random direction in feature space,
    so one draw can land far from 0.5 by chance (we measured sd ~0.13). We
    therefore average over `n_repeats` independent shuffles, which concentrates
    the estimate around the true null value of 0.5. We also use a single
    time-ordered 80/20 split (train on the past) rather than the smallest CV
    fold, so each draw uses as much training data as possible.
    """
    X = dataset.X.to_numpy()
    y = dataset.y.to_numpy()
    n = len(y)
    cut = int(n * 0.8)
    train_idx = np.arange(cut)
    test_idx = np.arange(cut, n)
    y_test = y[test_idx]

    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_repeats):
        y_train_shuffled = y[train_idx].copy()
        rng.shuffle(y_train_shuffled)
        est = clone(model)
        est.fit(X[train_idx], y_train_shuffled)
        prob = est.predict_proba(X[test_idx])[:, 1]
        # Real (unshuffled) test labels — random training signal should not predict.
        aucs.append(roc_auc_score(y_test, prob))
    return float(np.mean(aucs))


def overfit_tiny_subset(model, dataset: Dataset, n: int = 40) -> float:
    """A working model must (over)fit a tiny slice to near-perfect train AUC."""
    X = dataset.X.to_numpy()[:n]
    y = dataset.y.to_numpy()[:n]
    # Ensure the slice has both classes; if not, widen it.
    if len(np.unique(y)) < 2:
        # Take the first n rows of each class.
        full_y = dataset.y.to_numpy()
        full_X = dataset.X.to_numpy()
        pos = np.where(full_y == 1)[0][: n // 2]
        neg = np.where(full_y == 0)[0][: n // 2]
        idx = np.concatenate([pos, neg])
        X, y = full_X[idx], full_y[idx]
    est = clone(model)
    est.fit(X, y)
    prob = est.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, prob))


def run_sanity_checks(
    models: dict, dataset: Dataset, observed_auc: dict[str, float], seed: int
) -> SanityReport:
    """Run all sanity checks and return a structured pass/fail report.

    `observed_auc` is the held-out mean ROC-AUC per model from the real run; we
    use it for the baseline-floor and leakage-ceiling checks.
    """
    checks: dict[str, dict] = {}

    # 1. Dedup: no exact duplicate rows survived loading.
    n_dupes_remaining = int(dataset.frame.duplicated().sum())
    checks["dedup"] = {
        "passed": n_dupes_remaining == 0,
        "detail": f"{dataset.n_duplicates_removed} exact duplicates removed at load; "
        f"{n_dupes_remaining} remain",
    }

    # 2. Baseline floor: every model beats AUC 0.5.
    floor = baseline_auc(dataset)
    floor_ok = all(v > floor for v in observed_auc.values())
    checks["baseline_floor"] = {
        "passed": floor_ok,
        "detail": f"all arms above AUC {floor} baseline: "
        + ", ".join(f"{k}={v:.3f}" for k, v in observed_auc.items()),
    }

    # 3. Leakage ceiling: no arm scores implausibly high.
    ceiling_ok = all(v < LEAKAGE_CEILING for v in observed_auc.values())
    checks["leakage_ceiling"] = {
        "passed": ceiling_ok,
        "detail": f"all arms below AUC {LEAKAGE_CEILING} (would imply leakage): "
        + ", ".join(f"{k}={v:.3f}" for k, v in observed_auc.items()),
    }

    # 4. Label-shuffle: shuffled-label training collapses to chance.
    shuffle_results = {
        name: label_shuffle_auc(m, dataset, seed) for name, m in models.items()
    }
    shuffle_ok = all(abs(v - 0.5) < 0.1 for v in shuffle_results.values())
    checks["label_shuffle"] = {
        "passed": shuffle_ok,
        "detail": "shuffled-label AUC near 0.5: "
        + ", ".join(f"{k}={v:.3f}" for k, v in shuffle_results.items()),
    }

    # 5. Overfit tiny subset: model can drive train AUC near 1.0.
    overfit_results = {
        name: overfit_tiny_subset(m, dataset) for name, m in models.items()
    }
    overfit_ok = all(v > 0.9 for v in overfit_results.values())
    checks["overfit_tiny_subset"] = {
        "passed": overfit_ok,
        "detail": "tiny-subset train AUC near 1.0: "
        + ", ".join(f"{k}={v:.3f}" for k, v in overfit_results.items()),
    }

    return SanityReport(checks=checks)
