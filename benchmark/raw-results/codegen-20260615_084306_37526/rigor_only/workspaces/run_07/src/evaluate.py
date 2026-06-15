"""Evaluation metrics and sanity checks."""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def score(model, X, y) -> dict[str, float]:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
    }


def majority_baseline(X_train, y_train, X_test, y_test) -> dict[str, float]:
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_train, y_train)
    return score(clf, X_test, y_test)


def label_shuffle_auc(pipeline_factory, X_train, y_train, X_test, y_test, n_trials: int = 20) -> float:
    """Average AUC over multiple label shuffles; should be near 0.5.

    Averaging over many trials reduces sampling noise on small test sets.
    A threshold of 0.57 catches real leakage (typically 0.7+) while tolerating
    the ±0.03 variance that appears with n~800 test rows.
    """
    aucs = []
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        y_shuffled = rng.permutation(y_train)
        clf = pipeline_factory(random_state=seed)
        clf.fit(X_train, y_shuffled)
        y_prob = clf.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, y_prob))
    return float(np.mean(aucs))


def overfit_check(pipeline_factory, X_train, y_train, seed: int = 0, n: int = 50) -> float:
    """Model must reach high train accuracy on a tiny subset."""
    clf = pipeline_factory(random_state=seed)
    clf.fit(X_train[:n], y_train[:n])
    preds = clf.predict(X_train[:n])
    return float((preds == y_train[:n]).mean())
