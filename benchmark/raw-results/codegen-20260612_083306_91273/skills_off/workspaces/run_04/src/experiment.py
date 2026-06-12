"""Core experiment: compare LogisticRegression vs GradientBoostingClassifier."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from pathlib import Path


class ChurnExperiment:
    """Run churn prediction experiment with rigor checks."""

    def __init__(self, X: pd.DataFrame, y: pd.Series, seeds: list[int] = None):
        self.X = X.copy()
        self.y = y.copy()
        self.seeds = seeds or [42, 123, 456]
        self.results = {}

    def run_sanity_checks(self) -> dict:
        """Run baseline and overfit checks to catch pipeline bugs early."""
        checks = {}

        # 1. Baseline: majority class predictor
        baseline_pred = np.ones(len(self.y)) * self.y.value_counts().idxmax()
        baseline_auc = roc_auc_score(self.y, baseline_pred) if len(np.unique(baseline_pred)) > 1 else 0.5
        checks['baseline_auc'] = float(baseline_auc)

        # 2. Label shuffle: shuffle labels and check performance drops
        y_shuffled = self.y.copy().values
        np.random.seed(self.seeds[0])
        np.random.shuffle(y_shuffled)

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.seeds[0], stratify=self.y
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lr = LogisticRegression(random_state=self.seeds[0], max_iter=1000)
        lr.fit(X_train_scaled, y_train)
        y_test_pred = lr.predict_proba(X_test_scaled)[:, 1]
        normal_auc = roc_auc_score(y_test, y_test_pred)

        # Fit on shuffled labels in train set
        y_train_shuffled = y_train.copy().values
        np.random.shuffle(y_train_shuffled)
        lr.fit(X_train_scaled, y_train_shuffled)
        y_test_pred_shuffled = lr.predict_proba(X_test_scaled)[:, 1]
        shuffled_auc = roc_auc_score(y_test, y_test_pred_shuffled)

        checks['normal_auc'] = float(normal_auc)
        checks['shuffled_auc'] = float(shuffled_auc)
        checks['label_shuffle_valid'] = shuffled_auc < (normal_auc - 0.1)

        return checks

    def run_comparison(self) -> dict:
        """Run the full comparison across multiple seeds."""
        results_per_seed = {
            'LogisticRegression': [],
            'GradientBoosting': []
        }

        for seed in self.seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=seed, stratify=self.y
            )

            # Fit scaler on train, apply to all
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train LR
            lr = LogisticRegression(random_state=seed, max_iter=1000)
            lr.fit(X_train_scaled, y_train)
            y_test_pred_lr = lr.predict_proba(X_test_scaled)[:, 1]

            # Train GB on scaled features (for consistency)
            gb = GradientBoostingClassifier(random_state=seed, n_estimators=100, max_depth=5)
            gb.fit(X_train_scaled, y_train)
            y_test_pred_gb = gb.predict_proba(X_test_scaled)[:, 1]

            # Evaluate both
            lr_metrics = {
                'auc': float(roc_auc_score(y_test, y_test_pred_lr)),
                'accuracy': float(accuracy_score(y_test, (y_test_pred_lr >= 0.5).astype(int))),
                'precision': float(precision_score(y_test, (y_test_pred_lr >= 0.5).astype(int), zero_division=0)),
                'recall': float(recall_score(y_test, (y_test_pred_lr >= 0.5).astype(int), zero_division=0)),
                'f1': float(f1_score(y_test, (y_test_pred_lr >= 0.5).astype(int), zero_division=0)),
            }
            gb_metrics = {
                'auc': float(roc_auc_score(y_test, y_test_pred_gb)),
                'accuracy': float(accuracy_score(y_test, (y_test_pred_gb >= 0.5).astype(int))),
                'precision': float(precision_score(y_test, (y_test_pred_gb >= 0.5).astype(int), zero_division=0)),
                'recall': float(recall_score(y_test, (y_test_pred_gb >= 0.5).astype(int), zero_division=0)),
                'f1': float(f1_score(y_test, (y_test_pred_gb >= 0.5).astype(int), zero_division=0)),
            }

            results_per_seed['LogisticRegression'].append(lr_metrics)
            results_per_seed['GradientBoosting'].append(gb_metrics)

        return results_per_seed

    def summarize_results(self, results: dict) -> dict:
        """Compute mean ± std across seeds."""
        summary = {}
        for model_name, seed_results in results.items():
            df = pd.DataFrame(seed_results)
            summary[model_name] = {
                metric: {
                    'mean': float(df[metric].mean()),
                    'std': float(df[metric].std()),
                    'values': df[metric].tolist()
                }
                for metric in df.columns
            }
        return summary

    def write_results(self, output_dir: str, sanity_checks: dict, summary: dict):
        """Write machine-readable results to JSON."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        results_obj = {
            'claim': 'Gradient boosting outperforms logistic regression for churn prediction',
            'design': {
                'variable': 'Algorithm (GradientBoostingClassifier vs LogisticRegression)',
                'split_policy': 'Stratified train/test split (80/20)',
                'seeds': self.seeds,
                'n_seeds': len(self.seeds),
            },
            'sanity_checks': sanity_checks,
            'results': summary,
        }

        output_path = Path(output_dir) / 'metrics.json'
        with open(output_path, 'w') as f:
            json.dump(results_obj, f, indent=2)
        print(f"Wrote results to {output_path}")
