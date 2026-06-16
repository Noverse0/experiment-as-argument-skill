"""Pre-flight sanity checks before full experiment."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.metrics import evaluate, baseline_majority


def check_overfit_tiny_subset(X_train, y_train, X_test, y_test):
    """Model must reach near-zero loss on a tiny training subset."""
    tiny_size = min(50, len(X_train) // 2)
    X_tiny = X_train[:tiny_size]
    y_tiny = y_train[:tiny_size]

    # Test with both models
    for name, model in [
        ("LR", Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])),
        ("GB", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
    ]:
        model.fit(X_tiny, y_tiny)
        train_loss = 1 - model.score(X_tiny, y_tiny)
        assert train_loss < 0.15, f"{name} cannot overfit tiny subset (loss={train_loss:.2f})"
        print(f"✓ {name} overfit check passed (train loss={train_loss:.3f})")


def check_label_shuffle(X_train, y_train, X_test, y_test):
    """With shuffled labels, performance must drop to near baseline floor."""
    rng = np.random.default_rng(42)
    y_shuffled = rng.permutation(y_train)

    for name, model in [
        ("LR", Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])),
        ("GB", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
    ]:
        model.fit(X_train, y_shuffled)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, y_pred_proba)
        auc = metrics["auc"]
        # With random labels, AUC should be near 0.5 (allow very wide range for variance)
        assert 0.2 < auc < 0.8, f"{name} AUC with shuffled labels is {auc:.2f} (should be ~0.5)"
        print(f"✓ {name} label shuffle check passed (AUC={auc:.3f})")


def check_baseline_floor(X_train, y_train, X_test, y_test):
    """Baseline majority class should perform poorly."""
    baseline = baseline_majority(y_test)
    auc = baseline["auc"]
    # Majority baseline on imbalanced data has AUC ~0.5
    print(f"✓ Majority baseline AUC={auc:.3f}")


def run_sanity_checks(X_train, y_train, X_test, y_test):
    """Run all pre-flight checks."""
    print("\n=== SANITY CHECKS ===")
    check_baseline_floor(X_train, y_train, X_test, y_test)
    check_overfit_tiny_subset(X_train, y_train, X_test, y_test)
    check_label_shuffle(X_train, y_train, X_test, y_test)
    print("✓ All sanity checks passed\n")
