"""Evaluation utilities: CV, test-set scoring, sanity checks, and baseline."""
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.dummy import DummyClassifier


def majority_baseline(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(y_train.reshape(-1, 1), y_train)
    y_pred = clf.predict(y_test.reshape(-1, 1))
    majority_class = int(np.bincount(y_train).argmax())
    return {
        "strategy": "most_frequent",
        "majority_class": majority_class,
        "test_accuracy": float((y_pred == y_test).mean()),
        "test_auc": float(roc_auc_score(y_test, y_pred)),
    }


def cross_validate_model(
    pipe,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Stratified k-fold CV on the training set to estimate variance."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring=["roc_auc", "f1"],
        return_train_score=False,
    )
    return {
        "cv_auc_per_fold": scores["test_roc_auc"].tolist(),
        "cv_auc_mean": float(scores["test_roc_auc"].mean()),
        "cv_auc_std": float(scores["test_roc_auc"].std()),
        "cv_f1_per_fold": scores["test_f1"].tolist(),
        "cv_f1_mean": float(scores["test_f1"].mean()),
        "cv_f1_std": float(scores["test_f1"].std()),
        "n_folds": n_folds,
    }


def evaluate_on_test(pipe, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Score a fitted pipeline on the held-out test set."""
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)
    return {
        "test_auc": float(roc_auc_score(y_test, y_prob)),
        "test_f1": float(f1_score(y_test, y_pred)),
    }


def label_shuffle_check(
    pipe_factory,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
    n_repeats: int = 5,
) -> dict:
    """Shuffle labels (n_repeats seeds) and confirm mean AUC is near 0.5.

    A single-seed check has high variance on small test sets; averaging over
    n_repeats seeds gives a stable estimate.  If the mean remains above 0.65,
    information is leaking around the labels.
    """
    aucs = []
    for i in range(n_repeats):
        rng = np.random.default_rng(seed + i)
        y_shuffled = rng.permutation(y_train)
        pipe = pipe_factory()
        pipe.fit(X_train, y_shuffled)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        aucs.append(float(roc_auc_score(y_test, y_prob)))
    mean_auc = float(np.mean(aucs))
    return {
        "shuffled_label_auc_per_seed": aucs,
        "shuffled_label_auc_mean": mean_auc,
        "n_repeats": n_repeats,
        "passed": mean_auc < 0.65,
    }


def overfit_one_batch_check(pipe_factory, X: np.ndarray, y: np.ndarray, n: int = 50) -> dict:
    """Model must reach high train accuracy on a tiny subset (proves gradient flows)."""
    X_tiny, y_tiny = X[:n], y[:n]
    pipe = pipe_factory()
    pipe.fit(X_tiny, y_tiny)
    train_acc = float((pipe.predict(X_tiny) == y_tiny).mean())
    return {
        "tiny_n": n,
        "train_accuracy_on_tiny": train_acc,
        "passed": train_acc > 0.7,
    }
