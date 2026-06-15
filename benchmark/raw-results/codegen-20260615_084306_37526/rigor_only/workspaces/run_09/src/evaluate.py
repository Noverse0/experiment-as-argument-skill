"""Evaluation utilities: cross-validation, metrics, and sanity checks."""
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def score(model, X_train, y_train, X_test, y_test) -> dict:
    """Fit model on train, return metrics on test."""
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    return {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
    }


def run_cv(model, X, y, n_splits: int = 5, seeds=(42, 123, 456)) -> list[dict]:
    """Stratified k-fold CV across multiple random seeds.

    n_splits × len(seeds) total evaluations.  Each uses a fresh clone so
    model state never leaks across folds.
    """
    results = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, val_idx in skf.split(X, y):
            m = clone(model)
            metrics = score(
                m,
                X.iloc[train_idx], y.iloc[train_idx],
                X.iloc[val_idx], y.iloc[val_idx],
            )
            results.append(metrics)
    return results


def summarise_cv(results: list[dict]) -> dict:
    aucs = [r["roc_auc"] for r in results]
    f1s = [r["f1"] for r in results]
    return {
        "roc_auc_mean": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "n_evals": len(results),
    }


def baseline_score(y_train, y_test) -> dict:
    """Majority-class dummy baseline (lower bound for any useful model)."""
    dummy = DummyClassifier(strategy="most_frequent", random_state=0)
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    y_prob = dummy.predict_proba(np.zeros((len(y_test), 1)))[:, 1]
    y_pred = dummy.predict(np.zeros((len(y_test), 1)))
    return {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
    }


def label_shuffle_roc_auc(
    model, X_train, y_train, X_test, y_test, n_shuffles: int = 20
) -> tuple[float, float]:
    """Sanity check: return (mean_shuffled_auc, std) over many permutations.

    With L2-regularised models, shuffled predictions have tiny but non-zero
    variance, causing single-run AUC estimates to swing widely.  Averaging
    over many shuffles gives a stable estimate of the null performance.

    The caller should verify that the real AUC is substantially higher than
    mean_shuffled_auc (gap > 0.05) rather than checking shuffled < fixed threshold.
    """
    aucs = []
    for seed in range(n_shuffles):
        rng = np.random.default_rng(seed)
        y_shuffled = rng.permutation(y_train.values)
        m = clone(model)
        m.fit(X_train, y_shuffled)
        y_prob = m.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, y_prob))
    return float(np.mean(aucs)), float(np.std(aucs))
