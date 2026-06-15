"""Core experiment logic: train models, run sanity checks, compare."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from src.dataset import split_and_prepare


class ExperimentResult:
    """Container for a single run's results."""

    def __init__(self, seed: int):
        self.seed = seed
        self.metrics = {}
        self.sanity_checks = {}

    def add_metric(self, name: str, value: float):
        self.metrics[name] = value

    def add_sanity(self, name: str, value):
        self.sanity_checks[name] = value


def baseline_floor(y_test: np.ndarray) -> float:
    """Always-predict-majority baseline AUC (sanity check lower bound)."""
    majority = np.bincount(y_test).argmax()
    y_pred = np.full_like(y_test, majority, dtype=float)
    auc = roc_auc_score(y_test, y_pred)
    return auc


def label_shuffle_test(model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, seed: int) -> float:
    """
    Fit model with shuffled labels; AUC should drop to baseline.
    If it does not, information is leaking.
    """
    rng = np.random.RandomState(seed)
    y_train_shuffled = np.array(y_train).copy()
    rng.shuffle(y_train_shuffled)

    model.fit(X_train, y_train_shuffled)
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    return auc


def train_and_evaluate(model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Train model and compute metrics."""
    model.fit(X_train, y_train)

    # Train metrics (for overfit check)
    y_pred_train = model.predict_proba(X_train)[:, 1]
    auc_train = roc_auc_score(y_train, y_pred_train)

    # Test metrics
    y_pred = model.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, y_pred)

    # Binary predictions for precision/recall
    y_pred_binary = (y_pred >= 0.5).astype(int)

    precision = precision_score(y_test, y_pred_binary, zero_division=0)
    recall = recall_score(y_test, y_pred_binary, zero_division=0)
    f1 = f1_score(y_test, y_pred_binary, zero_division=0)

    return {
        "auc_train": auc_train,
        "auc_test": auc_test,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_experiment(csv_path: str, seed: int, test_size: float = 0.2) -> ExperimentResult:
    """
    Run one complete experiment: load data, run sanity checks, train both models.

    Claim: Gradient boosting achieves better test AUC than logistic regression
           on the churn dataset using only legitimate features.
    """
    result = ExperimentResult(seed=seed)

    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_cols, _ = split_and_prepare(
        csv_path, test_size=test_size, random_state=seed
    )

    # Sanity check 1: Baseline floor
    auc_baseline = baseline_floor(y_test)
    result.add_sanity("baseline_auc", auc_baseline)
    print(f"  Baseline (always-majority) AUC: {auc_baseline:.4f}")

    # Sanity check 2: Overfit check (can model fit training data?)
    lr_overfit = LogisticRegression(max_iter=500, random_state=seed)
    lr_overfit.fit(X_train, y_train)
    y_pred_train = lr_overfit.predict_proba(X_train)[:, 1]
    auc_train_overfit = roc_auc_score(y_train, y_pred_train)
    result.add_sanity("overfit_check_auc_train", auc_train_overfit)
    print(f"  Overfit check (LR on train): {auc_train_overfit:.4f}")

    # Sanity check 3: Label-shuffle test
    gb_shuffle = GradientBoostingClassifier(n_estimators=50, random_state=seed, max_depth=4)
    auc_shuffle = label_shuffle_test(gb_shuffle, X_train, y_train, X_test, y_test, seed)
    result.add_sanity("label_shuffle_auc", auc_shuffle)
    print(f"  Label-shuffle test AUC: {auc_shuffle:.4f} (should be near {auc_baseline:.4f})")

    # Train final models
    print(f"  Training LogisticRegression...")
    lr = LogisticRegression(max_iter=500, random_state=seed)
    lr_metrics = train_and_evaluate(lr, X_train, y_train, X_test, y_test)
    for name, val in lr_metrics.items():
        result.add_metric(f"lr_{name}", val)
    print(f"    LR test AUC: {lr_metrics['auc_test']:.4f}")

    print(f"  Training GradientBoostingClassifier...")
    gb = GradientBoostingClassifier(n_estimators=50, random_state=seed, max_depth=4, learning_rate=0.1)
    gb_metrics = train_and_evaluate(gb, X_train, y_train, X_test, y_test)
    for name, val in gb_metrics.items():
        result.add_metric(f"gb_{name}", val)
    print(f"    GB test AUC: {gb_metrics['auc_test']:.4f}")

    result.add_metric("n_train", len(X_train))
    result.add_metric("n_test", len(X_test))

    return result


def summarize_results(results: list[ExperimentResult]) -> dict:
    """Aggregate results across seeds."""
    summary = {}

    # Extract metrics for each model
    for key in ["lr_auc_test", "gb_auc_test", "lr_auc_train", "gb_auc_train"]:
        values = [r.metrics.get(key) for r in results if key in r.metrics]
        if values:
            summary[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "values": values,
            }

    # Sanity checks (should be consistent)
    for key in ["baseline_auc", "overfit_check_auc_train"]:
        values = [r.sanity_checks.get(key) for r in results if key in r.sanity_checks]
        if values:
            summary[f"sanity_{key}"] = {
                "mean": np.mean(values),
                "std": np.std(values),
            }

    return summary
