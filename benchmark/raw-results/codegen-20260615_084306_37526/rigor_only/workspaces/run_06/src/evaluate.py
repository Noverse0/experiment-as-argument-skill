"""Evaluation utilities: cross-validation, holdout scoring, and sanity checks."""
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def _score(y_true, y_prob: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
    }


def cv_evaluate(pipeline, X, y, n_seeds: int = 3, n_folds: int = 5) -> dict:
    """Run n_seeds × n_folds stratified CV on a dataset; return aggregated metrics.

    Uses clone() per fold so the input pipeline is never mutated.

    Limitation: uses StratifiedKFold within the training period, ignoring
    time ordering within the train window. This is noted in REPORT.md.
    """
    roc_scores, ap_scores = [], []
    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for tr_idx, val_idx in skf.split(X, y):
            pipe = clone(pipeline)
            pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            probs = pipe.predict_proba(X.iloc[val_idx])[:, 1]
            s = _score(y.iloc[val_idx], probs)
            roc_scores.append(s["roc_auc"])
            ap_scores.append(s["avg_precision"])
    n = len(roc_scores)
    return {
        "roc_auc": {"mean": float(np.mean(roc_scores)), "std": float(np.std(roc_scores)), "n": n},
        "avg_precision": {"mean": float(np.mean(ap_scores)), "std": float(np.std(ap_scores)), "n": n},
    }


def holdout_evaluate(pipeline, X_train, y_train, X_test, y_test) -> dict:
    """Fit on full train, score once on held-out test. Does not mutate pipeline."""
    pipe = clone(pipeline)
    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:, 1]
    return _score(y_test, probs)


def baseline_evaluate(y_train, y_test) -> dict:
    """Stratified-random dummy classifier baseline (expected ROC-AUC ≈ 0.5)."""
    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    probs = dummy.predict_proba(np.zeros((len(y_test), 1)))[:, 1]
    return _score(y_test, probs)


def shuffle_label_test(pipeline, X_train, y_train, n_seeds: int = 3) -> dict:
    """Verify that shuffled labels drive ROC-AUC toward 0.5.

    Runs 3-fold CV with permuted labels n_seeds times using the supplied model.
    A mean AUC >= 0.6 indicates label-independent information in the features.
    """
    roc_scores = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        y_shuffled = rng.permutation(y_train.values)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        for tr_idx, val_idx in skf.split(X_train, y_shuffled):
            pipe = clone(pipeline)
            pipe.fit(X_train.iloc[tr_idx], y_shuffled[tr_idx])
            probs = pipe.predict_proba(X_train.iloc[val_idx])[:, 1]
            roc_scores.append(float(roc_auc_score(y_shuffled[val_idx], probs)))
    return {
        "mean_roc_auc": float(np.mean(roc_scores)),
        "std_roc_auc": float(np.std(roc_scores)),
        "n": len(roc_scores),
    }
