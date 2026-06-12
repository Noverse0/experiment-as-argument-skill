import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def _split_scale(X, y, split_frac=0.5):
    split = int(len(X) * split_frac)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:split])
    X_test_s = scaler.transform(X[split:])
    return X_train_s, X_test_s, y[:split], y[split:]


def check_baseline_floor(model, X, y, threshold=0.52):
    """Model AUC must exceed a floor above random (0.5)."""
    X_train_s, X_test_s, y_train, y_test = _split_scale(X, y)
    m = clone(model)
    m.fit(X_train_s, y_train)
    auc = roc_auc_score(y_test, m.predict_proba(X_test_s)[:, 1])
    if auc <= threshold:
        raise AssertionError(f"Baseline floor failed: AUC {auc:.3f} <= {threshold}")
    return auc


def check_label_shuffle(model, X, y, rng_seed=42, max_auc=0.6):
    """With randomly shuffled labels, AUC must collapse near 0.5.

    Trains on shuffled labels; evaluates on shuffled test labels.
    If AUC stays high, something (feature or data structure) is leaking
    information that bypasses the label-learning step.
    """
    rng = np.random.default_rng(rng_seed)
    y_shuffled = rng.permutation(y)
    split = len(X) // 2

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:split])
    X_test_s = scaler.transform(X[split:])

    m = clone(model)
    m.fit(X_train_s, y_shuffled[:split])
    auc = roc_auc_score(y_shuffled[split:], m.predict_proba(X_test_s)[:, 1])
    if auc > max_auc:
        raise AssertionError(
            f"Label-shuffle AUC {auc:.3f} > {max_auc}: possible leakage"
        )
    return auc


def check_overfit_tiny(X, y, n=80):
    """Pipeline check: a decision tree must perfectly memorize n samples.

    Uses an unlimited DecisionTreeClassifier (not the experiment model) to
    isolate pipeline plumbing from model capacity. LR with default regularisation
    cannot overfit small imbalanced data — that is a model property, not a bug.
    Failure here means wrong shapes, bad dtypes, or a broken data route.
    """
    from sklearn.tree import DecisionTreeClassifier
    X_tiny, y_tiny = X[:n], y[:n]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_tiny)
    m = DecisionTreeClassifier(random_state=0)
    m.fit(X_s, y_tiny)
    train_acc = (m.predict(X_s) == y_tiny).mean()
    if train_acc < 0.99:
        raise AssertionError(
            f"Overfit-tiny train acc {train_acc:.2f} < 0.99: pipeline may be broken"
        )
    return train_acc
