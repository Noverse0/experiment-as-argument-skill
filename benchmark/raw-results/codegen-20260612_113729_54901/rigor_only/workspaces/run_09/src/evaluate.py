import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Compute ROC-AUC, F1, precision, and recall from predicted probabilities."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def run_seeds(pipeline_fn, X_train, y_train, X_test, y_test, seeds) -> dict:
    """Train pipeline with each seed and aggregate metrics across seeds.

    The train/test split is held fixed; only the model's random_state varies.
    For deterministic models (e.g. LR with lbfgs) std will be ~0, which is
    honest — it means the model has no initialization variance.
    """
    per_seed = []
    for seed in seeds:
        pipe = pipeline_fn(seed)
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        per_seed.append(compute_metrics(y_test, y_prob))

    summary: dict = {}
    for metric in per_seed[0]:
        vals = [r[metric] for r in per_seed]
        summary[f"{metric}_mean"] = float(np.mean(vals))
        summary[f"{metric}_std"] = float(np.std(vals))
    summary["n_seeds"] = len(seeds)
    summary["per_seed"] = per_seed
    return summary
