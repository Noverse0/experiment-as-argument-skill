"""Churn prediction experiment pipeline with leakage awareness."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from typing import Dict, Tuple, Any


class ChurnExperiment:
    """Experiment comparing LogisticRegression vs GradientBoostingClassifier."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.results = {}

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load the churn dataset."""
        df = pd.read_csv(csv_path)
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data: deduplicate, engineer features, exclude leakage.

        Timing test reasoning for days_since_last_login:
        - This value is recorded at/after the outcome occurs.
        - A churned customer (by definition) has stopped logging in.
        - At prediction time (before churn), this value is unknown.
        - Dropping it prevents target leakage.
        """
        # Deduplicate: remove exact duplicates (200 planted ones exist).
        df = df.drop_duplicates().reset_index(drop=True)

        # Extract temporal feature from signup_date.
        df["signup_date"] = pd.to_datetime(df["signup_date"])
        df["signup_year"] = df["signup_date"].dt.year
        df["signup_month"] = df["signup_date"].dt.month

        # Drop features that should not be used:
        # - customer_id: only an identifier
        # - signup_date: raw date (encoded as year/month above)
        # - days_since_last_login: TARGET LEAK (derived from outcome)
        features_to_drop = ["customer_id", "signup_date", "days_since_last_login"]
        df = df.drop(columns=features_to_drop)

        return df

    def split_data(
        self, df: pd.DataFrame, test_size: float = 0.3
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split into train/test with stratification and deterministic seed.
        """
        from sklearn.model_selection import train_test_split

        # Separate features and target
        X = df.drop(columns=["churned"])
        y = df["churned"]

        # Stratified split to preserve class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.seed,
            stratify=y
        )
        return (X_train, y_train), (X_test, y_test)

    def train_and_evaluate(self, clf, X_train, y_train, X_test, y_test, name: str) -> Dict[str, float]:
        """Train a model and return evaluation metrics."""
        # Fit on train only
        clf.fit(X_train, y_train)

        # Predict on test (touched once)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        # Compute metrics
        metrics = {
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }
        return metrics

    def train_baseline(self, y_train, y_test) -> Dict[str, float]:
        """
        Sanity check: baseline is majority class prediction.
        All models must beat this.
        """
        majority_class = y_train.value_counts().idxmax()
        y_pred_baseline = np.full_like(y_test, majority_class)
        y_pred_proba = np.full_like(y_test, float(majority_class), dtype=float)

        metrics = {
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "precision": precision_score(y_test, y_pred_baseline, zero_division=0),
            "recall": recall_score(y_test, y_pred_baseline, zero_division=0),
            "f1": f1_score(y_test, y_pred_baseline, zero_division=0),
        }
        return metrics

    def run(self, csv_path: str) -> Dict[str, Any]:
        """Run the full experiment."""
        # Load and preprocess
        df = self.load_data(csv_path)
        df = self.preprocess(df)

        # Check class balance
        class_balance = df["churned"].value_counts()
        churn_rate = df["churned"].mean()

        # Split
        (X_train, y_train), (X_test, y_test) = self.split_data(df)

        # Baseline
        baseline_metrics = self.train_baseline(y_train, y_test)

        # LogisticRegression with scaling
        lr_clf = LogisticRegression(random_state=self.seed, max_iter=1000)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        lr_metrics = self.train_and_evaluate(lr_clf, X_train_scaled, y_train, X_test_scaled, y_test, "LogisticRegression")

        # GradientBoostingClassifier (no scaling needed)
        gb_clf = GradientBoostingClassifier(
            random_state=self.seed,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        gb_metrics = self.train_and_evaluate(gb_clf, X_train, y_train, X_test, y_test, "GradientBoosting")

        return {
            "seed": self.seed,
            "n_samples": len(df),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "churn_rate": churn_rate,
            "class_distribution": class_balance.to_dict(),
            "baseline": baseline_metrics,
            "logistic_regression": lr_metrics,
            "gradient_boosting": gb_metrics,
        }
