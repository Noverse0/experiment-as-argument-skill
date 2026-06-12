"""Evaluation: cross-validation, sanity checks, final test scoring."""
import numpy as np
from typing import Dict, Any, List
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline


SEEDS = [42, 123, 777]
CV_FOLDS = 5
SCORING = ["roc_auc", "f1", "precision", "recall"]


def cv_score(pipeline_factory, X_train: np.ndarray, y_train: np.ndarray, seeds: List[int] = SEEDS) -> Dict[str, Any]:
    """Run CV over multiple seeds and return mean ± std for each metric."""
    all_scores: Dict[str, List[float]] = {m: [] for m in SCORING}

    for seed in seeds:
        model = pipeline_factory(seed=seed)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
        result = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING)
        for m in SCORING:
            all_scores[m].extend(result[f"test_{m}"].tolist())

    summary = {}
    for m in SCORING:
        vals = np.array(all_scores[m])
        summary[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    return summary


def final_test_score(pipeline_factory, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray, seed: int = 42) -> Dict[str, float]:
    """Train on full train set, evaluate once on held-out test set."""
    model = pipeline_factory(seed=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
    }


def baseline_score(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Majority-class baseline."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)
    # majority classifier has no probability spread, use constant prob for AUC
    y_prob = dummy.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    }


def label_shuffle_auc(pipeline_factory, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray, seed: int = 42) -> float:
    """Train on shuffled labels; AUC should collapse to ~0.5 if no leakage."""
    rng = np.random.default_rng(seed)
    y_shuffled = rng.permutation(y_train)
    model = pipeline_factory(seed=seed)
    model.fit(X_train, y_shuffled)
    y_prob = model.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, y_prob))
