"""Evaluation utilities: cross-validation and held-out test metrics."""
from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline


def cv_evaluate(
    model_factory: Callable[[], Pipeline],
    X,
    y,
    n_splits: int = 5,
    seeds: tuple = (42, 43, 44),
) -> dict:
    """
    Stratified k-fold CV repeated across multiple random seeds.

    Returns mean ± std of ROC-AUC over all (folds × seeds) evaluations.
    Multiple seeds give a more honest variance estimate than a single CV run.
    """
    all_roc_auc = []
    all_avg_prec = []

    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = cross_validate(
            model_factory(),
            X,
            y,
            cv=cv,
            scoring={"roc_auc": "roc_auc", "average_precision": "average_precision"},
            return_train_score=False,
        )
        all_roc_auc.extend(scores["test_roc_auc"].tolist())
        all_avg_prec.extend(scores["test_average_precision"].tolist())

    return {
        "roc_auc_mean": float(np.mean(all_roc_auc)),
        "roc_auc_std": float(np.std(all_roc_auc)),
        "avg_precision_mean": float(np.mean(all_avg_prec)),
        "avg_precision_std": float(np.std(all_avg_prec)),
        "n_evals": len(all_roc_auc),
    }


def evaluate_held_out(
    model_factory: Callable[[], Pipeline],
    X_train,
    y_train,
    X_test,
    y_test,
) -> dict:
    """Fit on the full training set; evaluate once on the held-out test set."""
    model = model_factory()
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    return {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "avg_precision": float(average_precision_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred)),
    }


def majority_class_auc(y_test) -> float:
    """ROC-AUC of a constant predictor (always 0.5, the baseline floor)."""
    constant_scores = np.full(len(y_test), 0.5)
    return float(roc_auc_score(y_test, constant_scores))
