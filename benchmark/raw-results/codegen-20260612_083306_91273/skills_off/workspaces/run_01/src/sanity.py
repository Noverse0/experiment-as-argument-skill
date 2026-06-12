"""Sanity checks that must pass before believing any training results."""
import numpy as np
from sklearn.metrics import roc_auc_score


def baseline_floor(y_train) -> float:
    """Majority-class baseline AUC (~0.5 for a balanced majority classifier)."""
    majority = int(np.array(y_train).mean() >= 0.5)
    preds = np.full(len(y_train), majority, dtype=float)
    # Majority class classifier has no discrimination power — AUC should be ~0.5
    return roc_auc_score(y_train, preds)


def leakage_ceiling_check(auc: float, threshold: float = 0.98) -> bool:
    """Return True (warning) if AUC looks suspiciously perfect."""
    if auc > threshold:
        print(
            f"[WARNING] AUC={auc:.4f} > {threshold} — possible leakage. Audit features."
        )
        return True
    return False


def label_shuffle_test(pipeline, X_train, y_train, seed: int = 0) -> float:
    """Fit with shuffled labels; AUC must fall near baseline (≤0.55)."""
    rng = np.random.default_rng(seed)
    y_shuffled = rng.permutation(np.array(y_train))
    pipeline.fit(X_train, y_shuffled)
    proba = pipeline.predict_proba(X_train)[:, 1]
    auc = roc_auc_score(y_shuffled, proba)
    print(f"[sanity] label-shuffle AUC on train: {auc:.4f} (expected near 0.5)")
    return auc
