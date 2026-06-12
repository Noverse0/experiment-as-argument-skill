"""Pre-experiment sanity checks. Fast and cheap — catch silent bugs before full runs."""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score


def check_baseline(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Majority-class baseline. Any real model must exceed this accuracy.
    Provides a floor reference; near-perfect real performance suggests leakage.
    """
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    acc = float((dummy.predict(X_test) == y_test).mean())
    return {
        "baseline_accuracy": acc,
        "test_target_rate": float(y_test.mean()),
    }


def check_overfit_tiny(pipeline, X_train: np.ndarray, y_train: np.ndarray,
                       n: int = 50) -> dict:
    """
    Overfit a small subset. Train acc < ~1.0 on 50 samples means the pipeline
    is broken (wrong labels, bad shapes, etc.).
    """
    X_tiny, y_tiny = X_train[:n], y_train[:n]
    pipeline.fit(X_tiny, y_tiny)
    train_acc = float((pipeline.predict(X_tiny) == y_tiny).mean())
    return {"overfit_tiny_train_acc": train_acc, "overfit_n": n}


def check_label_shuffle(pipeline_fn, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        n_shuffles: int = 3, random_state: int = 0) -> dict:
    """
    With shuffled labels, test AUC should fall to ~0.5.
    AUC >> 0.5 with shuffled labels means information leaks around the labels.
    """
    rng = np.random.default_rng(random_state)
    aucs = []
    for _ in range(n_shuffles):
        y_shuffled = rng.permutation(y_train)
        pipe = pipeline_fn(random_state=0)
        pipe.fit(X_train, y_shuffled)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        try:
            aucs.append(float(roc_auc_score(y_test, y_prob)))
        except ValueError:
            aucs.append(0.5)
    return {"shuffle_mean_auc": float(np.mean(aucs)), "shuffle_n": n_shuffles}
