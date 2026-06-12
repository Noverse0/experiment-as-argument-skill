"""Cross-validation and final evaluation utilities."""
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)


def cv_metrics(pipeline, X_train, y_train, seeds: list, cv: int = 5) -> dict:
    """Run CV over multiple seeds, return aggregated stats."""
    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }
    all_scores: dict = {k: [] for k in scoring}

    for seed in seeds:
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        results = cross_validate(
            pipeline, X_train, y_train, cv=kf, scoring=scoring, n_jobs=1
        )
        for metric in scoring:
            all_scores[metric].extend(results[f"test_{metric}"].tolist())

    stats = {}
    for metric, values in all_scores.items():
        arr = np.array(values)
        stats[metric] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "n": int(len(arr)),
            "values": [float(v) for v in arr],
        }
    return stats


def final_eval(pipeline, X_train, X_test, y_train, y_test) -> dict:
    """Fit on full train, evaluate once on held-out test. Touch test set exactly once."""
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = pipeline.predict(X_test)
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
    }
