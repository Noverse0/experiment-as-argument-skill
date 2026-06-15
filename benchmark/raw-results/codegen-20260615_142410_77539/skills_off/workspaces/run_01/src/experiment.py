"""Main experiment: gradient boosting vs logistic regression for churn prediction."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_validate

from src.pipeline import (
    load_and_clean, prepare_features, get_cv_splitter,
    preprocess_for_lr, preprocess_for_gb, audit_leak_surface
)


class ChurnExperiment:
    """Run experiment comparing LR vs GB for churn prediction."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.X = None
        self.y = None
        self.results = {}

    def run(self):
        """Execute the full experiment."""
        print("=" * 70)
        print("CHURN PREDICTION EXPERIMENT: Gradient Boosting vs Logistic Regression")
        print("=" * 70)

        # Load and clean
        self.df = load_and_clean(self.csv_path)
        audit_leak_surface(self.df)

        # Prepare features (drop leak and identifiers)
        self.X, self.y, feature_names = prepare_features(self.df)

        # Sanity checks
        self._sanity_checks()

        # Run experiment with multiple seeds
        self._run_repeated_cv()

        return self.results

    def _sanity_checks(self):
        """Baseline, label shuffle, and overfit tests."""
        print("\n" + "=" * 70)
        print("SANITY CHECKS")
        print("=" * 70)

        # Baseline: majority class
        majority_rate = self.y.value_counts(normalize=True).iloc[0]
        print(f"\n1. Baseline (majority class): {majority_rate:.2%}")

        # Label shuffle: train on shuffled labels, should drop to baseline
        print("\n2. Label shuffle test:")
        y_shuffled = self.y.copy()
        np.random.RandomState(42).shuffle(y_shuffled.values)

        lr_shuffle = LogisticRegression(max_iter=1000, random_state=42)
        gb_shuffle = GradientBoostingClassifier(random_state=42, n_estimators=50)

        cv = get_cv_splitter(n_splits=3, random_state=42)
        splits = list(cv.split(self.X, y_shuffled))

        shuffled_scores_lr = []
        shuffled_scores_gb = []

        for train_idx, test_idx in splits:
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = y_shuffled.iloc[train_idx], y_shuffled.iloc[test_idx]

            X_train_lr, X_test_lr = preprocess_for_lr(X_train, X_test)
            lr_shuffle.fit(X_train_lr, y_train)
            auc_lr = roc_auc_score(y_test, lr_shuffle.predict_proba(X_test_lr)[:, 1])
            shuffled_scores_lr.append(auc_lr)

            X_train_gb, X_test_gb = preprocess_for_gb(X_train, X_test)
            gb_shuffle.fit(X_train_gb, y_train)
            auc_gb = roc_auc_score(y_test, gb_shuffle.predict_proba(X_test_gb)[:, 1])
            shuffled_scores_gb.append(auc_gb)

        mean_lr_shuffle = np.mean(shuffled_scores_lr)
        mean_gb_shuffle = np.mean(shuffled_scores_gb)
        print(f"   LR on shuffled labels: {mean_lr_shuffle:.3f} (should be ~0.5)")
        print(f"   GB on shuffled labels: {mean_gb_shuffle:.3f} (should be ~0.5)")

        if mean_lr_shuffle > 0.6 or mean_gb_shuffle > 0.6:
            print("   ⚠️  WARNING: Models performing above baseline on shuffled labels!")
            print("      This may indicate leakage in features.")

        # Overfit tiny subset
        print("\n3. Overfit test (train on 50 samples, test on same):")
        tiny_idx = np.arange(50)
        X_tiny = self.X.iloc[tiny_idx]
        y_tiny = self.y.iloc[tiny_idx]

        X_tiny_lr, _ = preprocess_for_lr(X_tiny, X_tiny)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_tiny_lr, y_tiny)
        lr_tiny_score = roc_auc_score(y_tiny, lr.predict_proba(X_tiny_lr)[:, 1])

        X_tiny_gb, _ = preprocess_for_gb(X_tiny, X_tiny)
        gb = GradientBoostingClassifier(random_state=42, n_estimators=100)
        gb.fit(X_tiny_gb, y_tiny)
        gb_tiny_score = roc_auc_score(y_tiny, gb.predict_proba(X_tiny_gb)[:, 1])

        print(f"   LR on tiny subset: {lr_tiny_score:.3f} (should be close to 1.0)")
        print(f"   GB on tiny subset: {gb_tiny_score:.3f} (should be close to 1.0)")

        if lr_tiny_score < 0.8 or gb_tiny_score < 0.8:
            print("   ⚠️  WARNING: Models cannot overfit tiny subset.")
            print("      The pipeline may be broken.")

    def _run_repeated_cv(self):
        """Run CV with multiple seeds to estimate mean ± std."""
        print("\n" + "=" * 70)
        print("MAIN EXPERIMENT: 5-Fold CV × 5 Seeds")
        print("=" * 70)

        seeds = [42, 123, 456, 789, 999]
        lr_all_scores = {"auc": [], "f1": [], "precision": [], "recall": []}
        gb_all_scores = {"auc": [], "f1": [], "precision": [], "recall": []}

        for seed_idx, seed in enumerate(seeds, 1):
            print(f"\nSeed {seed_idx}/{len(seeds)} (random_state={seed}):")

            cv = get_cv_splitter(n_splits=5, random_state=seed)
            splits = list(cv.split(self.X, self.y))

            seed_lr_aucs = []
            seed_gb_aucs = []

            for fold_idx, (train_idx, test_idx) in enumerate(splits, 1):
                X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
                y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

                # Logistic Regression
                X_train_lr, X_test_lr = preprocess_for_lr(X_train, X_test)
                lr = LogisticRegression(max_iter=1000, random_state=seed)
                lr.fit(X_train_lr, y_train)
                y_pred_proba_lr = lr.predict_proba(X_test_lr)[:, 1]
                y_pred_lr = lr.predict(X_test_lr)

                auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
                f1_lr = f1_score(y_test, y_pred_lr)
                prec_lr = precision_score(y_test, y_pred_lr, zero_division=0)
                rec_lr = recall_score(y_test, y_pred_lr, zero_division=0)

                seed_lr_aucs.append(auc_lr)
                lr_all_scores["auc"].append(auc_lr)
                lr_all_scores["f1"].append(f1_lr)
                lr_all_scores["precision"].append(prec_lr)
                lr_all_scores["recall"].append(rec_lr)

                # Gradient Boosting
                X_train_gb, X_test_gb = preprocess_for_gb(X_train, X_test)
                gb = GradientBoostingClassifier(
                    n_estimators=100, random_state=seed, learning_rate=0.1
                )
                gb.fit(X_train_gb, y_train)
                y_pred_proba_gb = gb.predict_proba(X_test_gb)[:, 1]
                y_pred_gb = gb.predict(X_test_gb)

                auc_gb = roc_auc_score(y_test, y_pred_proba_gb)
                f1_gb = f1_score(y_test, y_pred_gb)
                prec_gb = precision_score(y_test, y_pred_gb, zero_division=0)
                rec_gb = recall_score(y_test, y_pred_gb, zero_division=0)

                seed_gb_aucs.append(auc_gb)
                gb_all_scores["auc"].append(auc_gb)
                gb_all_scores["f1"].append(f1_gb)
                gb_all_scores["precision"].append(prec_gb)
                gb_all_scores["recall"].append(rec_gb)

                print(f"  Fold {fold_idx}: LR AUC={auc_lr:.3f}, GB AUC={auc_gb:.3f}")

            seed_lr_mean = np.mean(seed_lr_aucs)
            seed_gb_mean = np.mean(seed_gb_aucs)
            print(f"  Seed mean: LR={seed_lr_mean:.3f}, GB={seed_gb_mean:.3f}")

        # Summary statistics
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        def format_metric(values, name):
            mean = np.mean(values)
            std = np.std(values)
            print(f"  {name}: {mean:.4f} ± {std:.4f} (n={len(values)})")
            return {"mean": float(mean), "std": float(std), "n": len(values)}

        print("\nLogistic Regression:")
        lr_results = {}
        for metric in ["auc", "f1", "precision", "recall"]:
            lr_results[metric] = format_metric(lr_all_scores[metric], metric.upper())

        print("\nGradient Boosting:")
        gb_results = {}
        for metric in ["auc", "f1", "precision", "recall"]:
            gb_results[metric] = format_metric(gb_all_scores[metric], metric.upper())

        # Comparison
        print("\n" + "=" * 70)
        print("COMPARISON (GB - LR)")
        print("=" * 70)
        for metric in ["auc", "f1", "precision", "recall"]:
            lr_mean = lr_results[metric]["mean"]
            gb_mean = gb_results[metric]["mean"]
            diff = gb_mean - lr_mean
            print(f"  {metric.upper()}: {diff:+.4f} ({gb_mean:.4f} vs {lr_mean:.4f})")

        self.results = {
            "logistic_regression": lr_results,
            "gradient_boosting": gb_results,
            "comparison": {
                metric: float(gb_results[metric]["mean"] - lr_results[metric]["mean"])
                for metric in ["auc", "f1", "precision", "recall"]
            },
            "n_samples": len(self.X),
            "n_folds": 5,
            "n_seeds": len(seeds),
        }

        return self.results
