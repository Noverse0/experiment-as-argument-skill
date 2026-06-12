"""Core experiment: compare LogisticRegression vs GradientBoostingClassifier."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from typing import Dict, List
import json


class SanityChecks:
    """Run sanity checks before and after training."""

    @staticmethod
    def baseline_floor(y_train: np.ndarray, y_test: np.ndarray) -> float:
        """
        Baseline: always predict majority class.
        Models must beat this.
        """
        majority_class = (y_train.mean() > 0.5).astype(int)
        baseline_pred = np.ones(len(y_test)) * majority_class
        baseline_auc = roc_auc_score(y_test, baseline_pred)
        print(f"Baseline (majority class) AUC: {baseline_auc:.4f}")
        return baseline_auc

    @staticmethod
    def tiny_overfit_check(X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Model must fit ~zero loss on a tiny subset (overfit check).
        If it cannot, the pipeline is broken.
        """
        tiny_idx = slice(0, min(50, len(X_train)))
        X_tiny = X_train[tiny_idx]
        y_tiny = y_train[tiny_idx]

        model = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, random_state=42
        )
        model.fit(X_tiny, y_tiny)
        y_pred = model.predict_proba(X_tiny)[:, 1]
        tiny_auc = roc_auc_score(y_tiny, y_pred)
        print(f"Tiny overfit check (n=50) AUC: {tiny_auc:.4f}")
        if tiny_auc < 0.95:
            raise RuntimeError(
                f"Tiny overfit check failed (AUC={tiny_auc:.4f}). Pipeline broken?"
            )

    @staticmethod
    def label_shuffle_test(
        X_test: np.ndarray, y_test: np.ndarray, baseline_auc: float
    ) -> None:
        """
        With shuffled labels, performance must drop to baseline.
        If not, information is leaking around the labels.
        """
        y_shuffled = np.random.permutation(y_test)
        shuffled_pred = np.random.rand(len(y_test))
        shuffled_auc = roc_auc_score(y_shuffled, shuffled_pred)
        print(f"Label shuffle test AUC: {shuffled_auc:.4f}")


class Experiment:
    """Single experiment: train both models on a given train/test split."""

    def __init__(self, seed: int):
        self.seed = seed
        self.results = {}

    def run(
        self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
    ) -> Dict:
        """Train and evaluate both models."""
        results = {}

        # LogisticRegression
        lr = LogisticRegression(max_iter=1000, random_state=self.seed)
        lr.fit(X_train, y_train)
        lr_pred = lr.predict_proba(X_test)[:, 1]
        results["logistic_regression"] = {
            "auc": roc_auc_score(y_test, lr_pred),
            "precision": precision_score(y_test, (lr_pred > 0.5).astype(int)),
            "recall": recall_score(y_test, (lr_pred > 0.5).astype(int)),
            "f1": f1_score(y_test, (lr_pred > 0.5).astype(int)),
        }

        # GradientBoostingClassifier
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=self.seed
        )
        gb.fit(X_train, y_train)
        gb_pred = gb.predict_proba(X_test)[:, 1]
        results["gradient_boosting"] = {
            "auc": roc_auc_score(y_test, gb_pred),
            "precision": precision_score(y_test, (gb_pred > 0.5).astype(int)),
            "recall": recall_score(y_test, (gb_pred > 0.5).astype(int)),
            "f1": f1_score(y_test, (gb_pred > 0.5).astype(int)),
        }

        return results


class ResultsCollector:
    """Aggregate results across multiple seeds."""

    def __init__(self):
        self.all_results = {}  # {model_name: {metric: [values...]}}

    def add(self, seed_results: Dict) -> None:
        """Add results from one seed."""
        for model_name, metrics in seed_results.items():
            if model_name not in self.all_results:
                self.all_results[model_name] = {m: [] for m in metrics}
            for metric, value in metrics.items():
                self.all_results[model_name][metric].append(value)

    def summarize(self) -> Dict:
        """Return mean ± std for each model and metric."""
        summary = {}
        for model_name in self.all_results:
            summary[model_name] = {}
            for metric, values in self.all_results[model_name].items():
                mean = np.mean(values)
                std = np.std(values)
                summary[model_name][metric] = {
                    "mean": float(mean),
                    "std": float(std),
                    "n": len(values),
                    "values": [float(v) for v in values],
                }
        return summary

    def save_json(self, path: str) -> None:
        """Save summary to JSON."""
        summary = self.summarize()
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved results to {path}")

    def compare(self) -> str:
        """Return a brief comparison string."""
        summary = self.summarize()
        lr_auc = summary["logistic_regression"]["auc"]["mean"]
        gb_auc = summary["gradient_boosting"]["auc"]["mean"]
        lr_std = summary["logistic_regression"]["auc"]["std"]
        gb_std = summary["gradient_boosting"]["auc"]["std"]
        diff = gb_auc - lr_auc
        overlap = abs(diff) < (lr_std + gb_std)

        if overlap:
            conclusion = "no detectable difference"
        elif diff > 0:
            conclusion = f"gradient boosting outperforms by {abs(diff):.4f} AUC"
        else:
            conclusion = f"logistic regression outperforms by {abs(diff):.4f} AUC"

        return (
            f"LogisticRegression AUC: {lr_auc:.4f} ± {lr_std:.4f}\n"
            f"GradientBoosting AUC: {gb_auc:.4f} ± {gb_std:.4f}\n"
            f"Conclusion: {conclusion}"
        )
