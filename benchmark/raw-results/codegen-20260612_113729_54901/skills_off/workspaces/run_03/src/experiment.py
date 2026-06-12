import warnings
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_models():
    """Return a dict of named sklearn pipelines.

    LR needs StandardScaler; GBM is scale-invariant but including the scaler
    step in both pipelines keeps the CV loop uniform.
    """
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "gradient_boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
        ]),
        "majority_baseline": Pipeline([
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
    }


def compute_metrics(y_true, y_pred, y_prob=None):
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        m["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return m


def run_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5):
    """5-fold TimeSeriesSplit CV — training always precedes test in time.

    X must already be sorted by signup_date (ascending) before calling this,
    so that earlier indices correspond to earlier dates.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    models = make_models()
    fold_results = {name: [] for name in models}

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        for name, model in models.items():
            m = clone(model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X_tr, y_tr)
            y_pred = m.predict(X_te)
            y_prob = m.predict_proba(X_te)[:, 1] if hasattr(m, "predict_proba") else None
            metrics = compute_metrics(y_te, y_pred, y_prob)
            metrics["fold"] = fold_idx
            fold_results[name].append(metrics)

    return fold_results


def summarize(fold_results):
    """Compute mean ± std across folds for each model × metric."""
    summary = {}
    for name, folds in fold_results.items():
        metric_keys = [k for k in folds[0].keys() if k != "fold"]
        summary[name] = {}
        for key in metric_keys:
            values = [f[key] for f in folds]
            summary[name][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": [float(v) for v in values],
            }
    return summary
