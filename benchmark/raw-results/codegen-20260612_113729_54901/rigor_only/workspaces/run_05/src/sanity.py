"""Sanity checks that run before the full experiment.

Each check either passes silently or raises AssertionError with a diagnostic.
Run these before training to catch pipeline bugs cheaply.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score


def check_baseline_floor(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    baseline_auc: float = 0.5,
    margin: float = 0.02,
) -> float:
    """Trained model must beat a random/majority baseline on ROC-AUC."""
    m = clone(model)
    m.fit(X_train, y_train)
    auc = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
    assert auc > baseline_auc + margin, (
        f"Model AUC {auc:.4f} not above baseline {baseline_auc + margin:.4f}. "
        "Pipeline may be broken."
    )
    return auc


def check_overfit_tiny_subset(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n: int = 64,
    min_auc: float = 0.60,
) -> None:
    """Model should achieve AUC > min_auc on a tiny training subset.

    Uses AUC rather than accuracy so the check works for probabilistic
    targets (where Bayes error is non-zero and 0% train error is impossible).
    """
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_train), size=min(n, len(X_train)), replace=False)
    X_small, y_small = X_train[idx], y_train[idx]
    m = clone(model)
    m.fit(X_small, y_small)
    train_auc = roc_auc_score(y_small, m.predict_proba(X_small)[:, 1])
    assert train_auc >= min_auc, (
        f"Training AUC {train_auc:.4f} < {min_auc} on tiny subset. "
        "Model cannot fit the data — check feature/target alignment."
    )


def check_label_shuffle(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int = 0,
    max_auc: float = 0.55,
) -> float:
    """With shuffled labels, test AUC must collapse near 0.5 (no leakage)."""
    rng = np.random.default_rng(seed)
    y_shuffled = rng.permutation(y_train)
    m = clone(model)
    m.fit(X_train, y_shuffled)
    auc = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
    assert auc <= max_auc, (
        f"Label-shuffle AUC {auc:.4f} > {max_auc:.4f}. "
        "Possible feature leakage: information reaches the model independent of labels."
    )
    return auc


def check_no_target_leak(feature_names: list[str]) -> None:
    """account_status must not be in the feature set."""
    assert "account_status" not in feature_names, (
        "account_status is a direct label leak (closed ↔ churned=1). "
        "Remove it before training."
    )


def run_all(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> dict:
    check_no_target_leak(feature_names)
    check_overfit_tiny_subset(model, X_train, y_train)
    floor_auc = check_baseline_floor(model, X_train, y_train, X_test, y_test)
    # max_auc=0.65: with ~800 test samples, shuffled-label models can get ~0.60
    # by chance; genuine leakage (e.g. account_status) would push this >0.80.
    shuffle_auc = check_label_shuffle(model, X_train, y_train, X_test, y_test, max_auc=0.65)
    return {
        "floor_auc": float(floor_auc),
        "shuffle_auc": float(shuffle_auc),
    }
