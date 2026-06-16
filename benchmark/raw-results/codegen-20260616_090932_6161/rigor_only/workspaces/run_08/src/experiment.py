"""ML experiment: LogisticRegression vs GradientBoostingClassifier for churn."""
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from src.preprocessing import get_features_and_target, time_based_split


@dataclass
class MetricsPerSeed:
    """Metrics for a single seed and model."""

    seed: int
    model_name: str
    train_auc: float
    test_auc: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float


class ChurnExperiment:
    """Experiment comparing logistic regression vs gradient boosting."""

    def __init__(self, data_path: str, n_seeds: int = 5, test_month: int = 10):
        self.data_path = data_path
        self.n_seeds = n_seeds
        self.test_month = test_month
        self.results: dict[str, list[MetricsPerSeed]] = {
            "LogisticRegression": [],
            "GradientBoostingClassifier": [],
        }

    def _fit_logistic_regression(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> LogisticRegression:
        model = LogisticRegression(
            max_iter=1000, random_state=0, solver="lbfgs"
        )
        model.fit(X_train, y_train)
        return model

    def _fit_gradient_boosting(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> GradientBoostingClassifier:
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=0,
            verbose=0,
        )
        model.fit(X_train, y_train)
        return model

    def _compute_metrics(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[float, float, float, float, float, float]:
        """Return train_auc, test_auc, accuracy, precision, recall, f1."""
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred = model.predict(X_test)

        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
        train_auc = auc(fpr_train, tpr_train)

        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
        test_auc = auc(fpr_test, tpr_test)

        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, zero_division=0)
        recall = recall_score(y_test, y_test_pred, zero_division=0)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)

        return train_auc, test_auc, accuracy, precision, recall, f1

    def run_seed(self, seed: int):
        """Run experiment for a single seed."""
        import pandas as pd

        df = pd.read_csv(self.data_path)
        train_df, test_df = time_based_split(
            df, test_month=self.test_month, seed=seed
        )

        X_train, y_train = get_features_and_target(train_df)
        X_test, y_test = get_features_and_target(test_df)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for model_name, model in [
            ("LogisticRegression", self._fit_logistic_regression(
                X_train_scaled, y_train
            )),
            ("GradientBoostingClassifier", self._fit_gradient_boosting(
                X_train_scaled, y_train
            )),
        ]:
            train_auc, test_auc, acc, prec, rec, f1 = self._compute_metrics(
                model, X_train_scaled, y_train, X_test_scaled, y_test
            )
            metrics = MetricsPerSeed(
                seed=seed,
                model_name=model_name,
                train_auc=train_auc,
                test_auc=test_auc,
                test_accuracy=acc,
                test_precision=prec,
                test_recall=rec,
                test_f1=f1,
            )
            self.results[model_name].append(metrics)

    def run_all_seeds(self):
        """Run experiment across all seeds."""
        for seed in range(self.n_seeds):
            self.run_seed(seed)

    def sanity_check_label_shuffle(self):
        """Verify that shuffling labels degrades performance to baseline."""
        import pandas as pd

        df = pd.read_csv(self.data_path)
        train_df, test_df = time_based_split(df, test_month=self.test_month, seed=0)
        X_train, y_train = get_features_and_target(train_df)
        X_test, y_test = get_features_and_target(test_df)

        y_test_shuffled = np.random.RandomState(42).permutation(y_test.values)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = self._fit_logistic_regression(X_train_scaled, y_train)
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_shuffled, y_test_proba)
        shuffled_auc = auc(fpr, tpr)

        baseline_auc = 0.5
        assert (
            shuffled_auc < 0.55
        ), f"Label shuffle sanity check failed: AUC={shuffled_auc} should be near {baseline_auc}"

        return shuffled_auc

    def sanity_check_overfit_tiny(self):
        """Verify model can overfit a tiny subset."""
        import pandas as pd

        df = pd.read_csv(self.data_path)
        train_df, _ = time_based_split(df, test_month=self.test_month, seed=0)
        X_tiny, y_tiny = get_features_and_target(train_df.head(10))

        scaler = StandardScaler()
        X_tiny_scaled = scaler.fit_transform(X_tiny)

        model = self._fit_logistic_regression(X_tiny_scaled, y_tiny)
        train_loss = 1 - model.score(X_tiny_scaled, y_tiny)

        assert (
            train_loss < 0.3
        ), f"Tiny overfit check failed: training accuracy on 10 rows should be high, got {1-train_loss:.2f}"

        return train_loss

    def get_summary(self) -> dict:
        """Summarize results across all seeds."""
        summary = {}
        for model_name in ["LogisticRegression", "GradientBoostingClassifier"]:
            metrics_list = self.results[model_name]
            aucs = [m.test_auc for m in metrics_list]
            accs = [m.test_accuracy for m in metrics_list]
            f1s = [m.test_f1 for m in metrics_list]

            summary[model_name] = {
                "test_auc_mean": float(np.mean(aucs)),
                "test_auc_std": float(np.std(aucs)),
                "test_auc_values": [float(x) for x in aucs],
                "test_accuracy_mean": float(np.mean(accs)),
                "test_accuracy_std": float(np.std(accs)),
                "test_f1_mean": float(np.mean(f1s)),
                "test_f1_std": float(np.std(f1s)),
                "n_seeds": len(metrics_list),
            }

        return summary
