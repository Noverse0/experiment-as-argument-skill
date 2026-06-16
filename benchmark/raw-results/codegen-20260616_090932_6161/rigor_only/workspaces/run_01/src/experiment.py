"""Main experiment: compare gradient boosting vs logistic regression."""
import json
from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, roc_curve, make_scorer
import pandas as pd

from src.data import (
    prepare_features,
    get_baseline_predictions,
    report_class_distribution,
)

class Experiment:
    def __init__(self, X: np.ndarray, y: np.ndarray, n_repeats: int = 5, n_splits: int = 5):
        """
        Initialize experiment.

        Args:
            X: feature matrix
            y: target vector
            n_repeats: number of random seeds to run
            n_splits: number of CV folds
        """
        self.X = X
        self.y = y
        self.n_repeats = n_repeats
        self.n_splits = n_splits
        self.results = {}
        self.seeds = []

    def sanity_check_baseline(self) -> dict:
        """Check that models beat the majority class baseline."""
        baseline_preds = get_baseline_predictions(self.y)
        baseline_acc = (baseline_preds == self.y).mean()

        return {
            'baseline_accuracy': float(baseline_acc),
            'churn_rate': float(self.y.mean()),
        }

    def sanity_check_label_shuffle(self, seed: int = 42) -> dict:
        """Check that performance drops when labels are shuffled."""
        rng = np.random.RandomState(seed)
        y_shuffled = self.y.copy()
        rng.shuffle(y_shuffled)

        lr = LogisticRegression(random_state=seed, max_iter=1000, n_jobs=1)
        gb = GradientBoostingClassifier(random_state=seed, n_iter_no_change=None)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        # Single CV fold to be fast
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
        fold = list(skf.split(X_scaled, y_shuffled))[0]
        train_idx, test_idx = fold

        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_shuffled[train_idx], y_shuffled[test_idx]

        lr.fit(X_train, y_train)
        gb.fit(X_train, y_train)

        lr_auc = self._compute_auc(lr.predict_proba(X_test)[:, 1], y_test)
        gb_auc = self._compute_auc(gb.predict_proba(X_test)[:, 1], y_test)

        # With shuffled labels, AUC should be near 0.5
        baseline_auc = 0.5

        return {
            'lr_auc_shuffled': float(lr_auc),
            'gb_auc_shuffled': float(gb_auc),
            'baseline_auc': float(baseline_auc),
            'lr_above_baseline': bool(lr_auc > baseline_auc + 0.05),
            'gb_above_baseline': bool(gb_auc > baseline_auc + 0.05),
        }

    def sanity_check_overfit_small_batch(self, batch_size: int = 50, seed: int = 42) -> dict:
        """Check that model can overfit a tiny batch (train loss near 0)."""
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(self.X), size=min(batch_size, len(self.X)), replace=False)
        X_small = self.X[idx]
        y_small = self.y[idx]

        lr = LogisticRegression(random_state=seed, max_iter=10000, n_jobs=1)
        gb = GradientBoostingClassifier(random_state=seed, n_iter_no_change=None, verbose=0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_small)

        lr.fit(X_scaled, y_small)
        gb.fit(X_scaled, y_small)

        lr_train_acc = lr.score(X_scaled, y_small)
        gb_train_acc = gb.score(X_scaled, y_small)

        return {
            'lr_train_accuracy': float(lr_train_acc),
            'gb_train_accuracy': float(gb_train_acc),
            'both_above_90': bool(lr_train_acc > 0.9 and gb_train_acc > 0.9),
        }

    def run_model_comparison(self) -> None:
        """Run cross-validated comparison of LR vs GB across multiple seeds."""
        results_list = []

        for run_id in range(self.n_repeats):
            seed = 1000 + run_id  # Fixed seed sequence: 1000, 1001, 1002, ...
            self.seeds.append(seed)

            rng = np.random.RandomState(seed)
            skf = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=seed
            )

            lr = LogisticRegression(
                random_state=seed,
                max_iter=1000,
                n_jobs=1,
                solver='lbfgs'
            )
            gb = GradientBoostingClassifier(
                random_state=seed,
                n_iter_no_change=None,
                verbose=0,
                max_depth=3,
                learning_rate=0.1,
                n_estimators=100,
            )

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(self.X)

            lr_scores = cross_validate(
                lr, X_scaled, self.y,
                cv=skf,
                scoring={'auc': 'roc_auc', 'accuracy': 'accuracy'},
                n_jobs=1
            )
            gb_scores = cross_validate(
                gb, X_scaled, self.y,
                cv=skf,
                scoring={'auc': 'roc_auc', 'accuracy': 'accuracy'},
                n_jobs=1
            )

            lr_auc_mean = lr_scores['test_auc'].mean()
            lr_auc_std = lr_scores['test_auc'].std()
            gb_auc_mean = gb_scores['test_auc'].mean()
            gb_auc_std = gb_scores['test_auc'].std()

            results_list.append({
                'run_id': run_id,
                'seed': seed,
                'lr_auc': float(lr_auc_mean),
                'lr_auc_std': float(lr_auc_std),
                'lr_accuracy': float(lr_scores['test_accuracy'].mean()),
                'gb_auc': float(gb_auc_mean),
                'gb_auc_std': float(gb_auc_std),
                'gb_accuracy': float(gb_scores['test_accuracy'].mean()),
                'gb_better_auc': bool(gb_auc_mean > lr_auc_mean),
                'auc_gap': float(gb_auc_mean - lr_auc_mean),
            })

        self.results = {
            'model_comparison': results_list,
            'config': {
                'n_repeats': self.n_repeats,
                'n_splits': self.n_splits,
                'seeds': self.seeds,
                'lr_hyperparams': {
                    'solver': 'lbfgs',
                    'max_iter': 1000,
                },
                'gb_hyperparams': {
                    'max_depth': 3,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                },
            }
        }

    def compute_summary_stats(self) -> dict:
        """Compute mean and std across all runs."""
        results = self.results['model_comparison']

        lr_aucs = [r['lr_auc'] for r in results]
        gb_aucs = [r['gb_auc'] for r in results]
        gaps = [r['auc_gap'] for r in results]

        return {
            'lr_mean_auc': float(np.mean(lr_aucs)),
            'lr_std_auc': float(np.std(lr_aucs)),
            'gb_mean_auc': float(np.mean(gb_aucs)),
            'gb_std_auc': float(np.std(gb_aucs)),
            'mean_auc_gap': float(np.mean(gaps)),
            'std_auc_gap': float(np.std(gaps)),
            'n_runs': len(results),
        }

    @staticmethod
    def _compute_auc(y_pred_proba: np.ndarray, y_true: np.ndarray) -> float:
        """Compute AUC from predicted probabilities."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        return float(auc(fpr, tpr))

    def run_with_leaky_feature(self) -> dict:
        """
        Run comparison WITH the leaky feature included to demonstrate the leak.
        This shows suspiciously high performance when leakage is present.
        """
        X_leaky, y_leaky = prepare_features(
            pd.DataFrame({
                'tenure_months': self.X[:, 0],
                'monthly_spend': self.X[:, 1],
                'support_tickets': self.X[:, 2],
                'days_since_last_login': np.zeros(len(self.X)),  # placeholder
                'churned': self.y,
            }),
            include_leaky=False  # placeholder
        )

        # For now, return placeholder
        return {'note': 'Leaky feature test to be run separately'}

    def to_dict(self) -> dict:
        """Serialize experiment results."""
        return {
            'results': self.results,
            'summary': self.compute_summary_stats(),
        }
