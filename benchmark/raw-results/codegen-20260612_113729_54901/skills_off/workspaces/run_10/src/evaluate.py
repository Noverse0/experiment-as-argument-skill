"""Evaluation utilities: cross-validation and holdout scoring."""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

CV_METRICS = ["roc_auc", "f1", "precision", "recall"]


def cv_scores_multi_seed(
    model_factory,
    X,
    y,
    seeds: tuple = (42, 123, 456),
    n_splits: int = 5,
) -> dict:
    """Run stratified k-fold CV for multiple seeds; aggregate across all folds.

    Using 3 seeds × 5 folds = 15 estimates per model gives enough variance
    information to make honest uncertainty claims (mean ± sd, n=15).
    The scaler inside each Pipeline is re-fit per fold on that fold's train
    subset only, so there is no leakage from the held-out fold.
    """
    all_scores: dict = {m: [] for m in CV_METRICS}

    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_scores = cross_validate(
            model_factory(seed=seed), X, y, cv=cv, scoring=CV_METRICS
        )
        for m in CV_METRICS:
            all_scores[m].extend(fold_scores[f"test_{m}"].tolist())

    result = {}
    for m in CV_METRICS:
        vals = np.array(all_scores[m])
        result[m] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": int(len(vals)),
            "values": vals.tolist(),
        }
    return result


def holdout_scores(model, X_train, y_train, X_test, y_test) -> dict:
    """Fit model on full training set; score once on the held-out test set."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
    }


def baseline_scores(X_train, y_train, X_test, y_test) -> dict:
    """Majority-class baseline: a sanity-check floor every real model must beat."""
    dummy = DummyClassifier(strategy="most_frequent", random_state=0)
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)
    return {
        "roc_auc": 0.5,
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "accuracy": float((y_pred == np.asarray(y_test)).mean()),
    }
