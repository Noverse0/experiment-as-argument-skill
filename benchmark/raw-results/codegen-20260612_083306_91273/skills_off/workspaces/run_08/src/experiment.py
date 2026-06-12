"""Core experiment logic for comparing churn prediction models."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


@dataclass
class RunMetrics:
    """Metrics for a single train/val/test run."""

    seed: int
    model_name: str
    train_auc: float
    val_auc: float
    test_auc: float
    test_precision: float
    test_recall: float
    test_f1: float


class DataLoader:
    """Load and clean the churn dataset, detecting and handling data leakage."""

    @staticmethod
    def load_and_clean(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
        """Load CSV and return cleaned features and target.

        Handling:
        - Remove account_status (perfectly derived from target: leaked feature)
        - Remove duplicate rows before split to prevent cross-boundary leakage
        - Keep other features: tenure_months, monthly_spend, support_tickets, signup_date
        """
        df = pd.read_csv(csv_path)
        initial_rows = len(df)

        # Detect and report account_status leak
        if "account_status" in df.columns:
            # Verify it's a perfect leak
            leak_check = (df["account_status"] == "closed") == (df["churned"] == 1)
            is_perfect_leak = leak_check.all()
            if is_perfect_leak:
                print(f"⚠️  Detected perfect leak: account_status derived from churned. Removing.")
            df = df.drop(columns=["account_status"])

        # Remove duplicate rows before any split to prevent leakage
        df_dedup = df.drop_duplicates()
        removed_dups = initial_rows - len(df_dedup)
        if removed_dups > 0:
            print(f"⚠️  Removed {removed_dups} duplicate rows before split.")

        target = df_dedup["churned"].copy()
        features = df_dedup.drop(columns=["churned", "customer_id"]).copy()

        # Parse signup_date for temporal understanding (but keep as feature for simplicity)
        features["signup_date"] = pd.to_datetime(features["signup_date"])
        features["days_since_signup"] = (
            pd.Timestamp("2024-12-31") - features["signup_date"]
        ).dt.days
        features = features.drop(columns=["signup_date"])

        print(f"Loaded {len(features)} rows. Target balance: {target.mean():.1%} churn rate")
        return features, target


class Preprocessor:
    """Fit preprocessing on train set, apply to val/test to prevent leakage."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None

    def fit(self, X_train: pd.DataFrame) -> None:
        """Fit scaler on training data only."""
        self.scaler.fit(X_train)
        self.feature_names = X_train.columns.tolist()

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply fitted preprocessing."""
        return self.scaler.transform(X[self.feature_names])


class ExperimentRunner:
    """Train and evaluate both models with proper methodology."""

    def __init__(self, features: pd.DataFrame, target: pd.Series):
        self.features = features
        self.target = target
        self.baseline_pred = None

    def run_seed(self, seed: int) -> list[RunMetrics]:
        """Run experiment with a single seed, returning metrics for both models."""
        np.random.seed(seed)

        # Stratified split: 60% train, 20% val, 20% test
        sss1 = StratifiedShuffleSplit(
            n_splits=1, test_size=0.4, random_state=seed
        )
        train_idx, temp_idx = next(
            sss1.split(self.features, self.target)
        )

        X_train = self.features.iloc[train_idx]
        y_train = self.target.iloc[train_idx]
        X_temp = self.features.iloc[temp_idx]
        y_temp = self.target.iloc[temp_idx]

        # Split temp into val and test (50-50)
        sss2 = StratifiedShuffleSplit(
            n_splits=1, test_size=0.5, random_state=seed + 1000
        )
        val_idx, test_idx = next(
            sss2.split(X_temp, y_temp)
        )

        X_val = X_temp.iloc[val_idx]
        y_val = y_temp.iloc[val_idx]
        X_test = X_temp.iloc[test_idx]
        y_test = y_temp.iloc[test_idx]

        # Preprocess: fit on train only
        preprocessor = Preprocessor()
        preprocessor.fit(X_train)
        X_train_scaled = preprocessor.transform(X_train)
        X_val_scaled = preprocessor.transform(X_val)
        X_test_scaled = preprocessor.transform(X_test)

        results = []

        # Train and evaluate both models
        for model_name, model_class in [
            ("LogisticRegression", LogisticRegression),
            ("GradientBoosting", GradientBoostingClassifier),
        ]:
            if model_name == "LogisticRegression":
                model = model_class(
                    max_iter=1000, random_state=seed, n_jobs=1
                )
            else:
                model = model_class(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=seed,
                    n_iter_no_change=10,
                    validation_fraction=0.1,
                )

            # Train
            model.fit(X_train_scaled, y_train)

            # Predict probabilities
            y_train_pred = model.predict_proba(X_train_scaled)[:, 1]
            y_val_pred = model.predict_proba(X_val_scaled)[:, 1]
            y_test_pred = model.predict_proba(X_test_scaled)[:, 1]

            # Compute metrics
            train_auc = roc_auc_score(y_train, y_train_pred)
            val_auc = roc_auc_score(y_val, y_val_pred)
            test_auc = roc_auc_score(y_test, y_test_pred)

            # For test set: use optimal threshold based on val set
            fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)
            optimal_threshold_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_threshold_idx]

            y_test_binary = (y_test_pred >= optimal_threshold).astype(int)
            test_precision = (
                np.sum((y_test_binary == 1) & (y_test == 1))
                / np.sum(y_test_binary == 1)
                if np.sum(y_test_binary == 1) > 0
                else 0.0
            )
            test_recall = (
                np.sum((y_test_binary == 1) & (y_test == 1))
                / np.sum(y_test == 1)
                if np.sum(y_test == 1) > 0
                else 0.0
            )
            test_f1 = (
                2
                * test_precision
                * test_recall
                / (test_precision + test_recall)
                if (test_precision + test_recall) > 0
                else 0.0
            )

            results.append(
                RunMetrics(
                    seed=seed,
                    model_name=model_name,
                    train_auc=train_auc,
                    val_auc=val_auc,
                    test_auc=test_auc,
                    test_precision=test_precision,
                    test_recall=test_recall,
                    test_f1=test_f1,
                )
            )

        return results

    def run_multiple_seeds(self, seeds: list[int]) -> list[RunMetrics]:
        """Run experiment across multiple seeds."""
        all_results = []
        for seed in seeds:
            results = self.run_seed(seed)
            all_results.extend(results)
        return all_results


def summarize_results(all_metrics: list[RunMetrics]) -> dict[str, Any]:
    """Summarize results across seeds for both models."""
    df = pd.DataFrame([asdict(m) for m in all_metrics])

    summary = {}
    for model_name in df["model_name"].unique():
        model_data = df[df["model_name"] == model_name]
        summary[model_name] = {
            "test_auc_mean": float(model_data["test_auc"].mean()),
            "test_auc_std": float(model_data["test_auc"].std()),
            "test_auc_runs": int(len(model_data)),
            "test_precision_mean": float(model_data["test_precision"].mean()),
            "test_recall_mean": float(model_data["test_recall"].mean()),
            "test_f1_mean": float(model_data["test_f1"].mean()),
        }

    return summary
