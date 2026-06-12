import copy
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score


def baseline_floor(X_train, y_train, X_test, y_test) -> dict:
    """Majority-class baseline: any real model must beat this."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    preds = dummy.predict(X_test)
    return {
        "majority_class_accuracy": float((preds == y_test).mean()),
        "target_rate_train": float(y_train.mean()),
        "target_rate_test": float(y_test.mean()),
    }


def overfit_tiny_subset(model, X_train, y_train, n: int = 64, threshold: float = 0.75) -> dict:
    """
    Model should beat `threshold` accuracy on a tiny slice.
    Checks the pipeline runs end-to-end and the model can learn something.
    Threshold is 0.75 by default — appropriate for linear models; tree-based
    models will typically exceed 0.90 on 64 samples.
    """
    X_tiny = X_train[:n]
    y_tiny = y_train[:n]
    m = copy.deepcopy(model)
    m.fit(X_tiny, y_tiny)
    acc = float((m.predict(X_tiny) == y_tiny).mean())
    return {"accuracy_on_tiny": acc, "passed": acc > threshold}


def label_shuffle_test(
    model, X_train, y_train, X_test, y_test, n_shuffles: int = 7, base_seed: int = 0
) -> dict:
    """
    Average AUC over n_shuffles label permutations should be near 0.5.
    Averaging across multiple shuffles avoids single-seed flukes
    (a single seed can produce AUC > 0.65 by chance on a temporal split).
    Passes if mean_auc < 0.65.
    """
    aucs = []
    for i in range(n_shuffles):
        rng = np.random.default_rng(base_seed + i)
        y_shuffled = rng.permutation(y_train)
        m = copy.deepcopy(model)
        m.fit(X_train, y_shuffled)
        proba = m.predict_proba(X_test)[:, 1]
        aucs.append(float(roc_auc_score(y_test, proba)))
    mean_auc = float(np.mean(aucs))
    return {
        "mean_auc_with_shuffled_labels": mean_auc,
        "individual_aucs": aucs,
        "passed": mean_auc < 0.65,
    }
