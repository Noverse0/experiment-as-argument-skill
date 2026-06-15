import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate

METRICS = ["roc_auc", "f1", "accuracy"]


def run_cv(pipeline, X, y, seeds: list, n_splits: int = 5) -> dict:
    """Repeated stratified k-fold CV over multiple seeds.

    StandardScaler (and any other fit-like step) lives inside the pipeline and
    is therefore fitted only on each training fold — no leakage.
    """
    all_scores = {m: [] for m in METRICS}
    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=METRICS, n_jobs=1)
        for m in METRICS:
            all_scores[m].extend(scores[f"test_{m}"].tolist())

    return {
        m: {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": len(vals),
            "values": [float(v) for v in vals],
        }
        for m, vals in all_scores.items()
    }
