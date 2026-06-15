"""Experiment comparing gradient boosting vs logistic regression for churn prediction.

CLAIM: For predicting customer churn, does gradient boosting outperform
       logistic regression?

DESIGN:
  - Variable: Algorithm (GradientBoostingClassifier vs LogisticRegression)
  - Split: Time-based 80/20 by signup_date (respects temporal order)
  - Seeds: 5 runs per model to estimate variance
  - Metrics: ROC-AUC (primary), PR-AUC, and F1 (for imbalanced churn task)
  - Sanity checks: baseline floor, shuffle test, overfit check

DATA DISCIPLINE:
  - Leak audit: days_since_last_login dropped (target leak)
  - Split before transform: scaler fit on train, applied to test
  - Dedup check before split
  - Report train/test churn rate imbalance
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import pandas as pd


class ExperimentRunner:
    def __init__(self, X_train, y_train, X_test, y_test, seeds=[42, 123, 456, 789, 999]):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.seeds = seeds
        self.results = []

    def sanity_check_baseline(self, baseline_pred, name="majority_class"):
        """Baseline floor: model must beat majority class prediction."""
        baseline_auc = roc_auc_score(self.y_test, baseline_pred)
        print(f"Sanity: baseline {name} AUC = {baseline_auc:.4f}")
        return baseline_auc

    def sanity_check_overfit(self, model_class, model_kwargs, seed):
        """Overfit check: model should reach near-zero loss on 100 train samples."""
        kwargs = {**model_kwargs, 'random_state': seed}
        model = model_class(**kwargs)
        small_sample = np.random.default_rng(seed).choice(len(self.X_train), 100, replace=False)
        X_small = self.X_train.iloc[small_sample]
        y_small = self.y_train[small_sample]
        model.fit(X_small, y_small)
        y_pred_proba = model.predict_proba(X_small)[:, 1]
        auc_small = roc_auc_score(y_small, y_pred_proba)
        print(f"Sanity: overfit check (100 samples) AUC = {auc_small:.4f} (should be high)")
        return auc_small

    def sanity_check_label_shuffle(self, model_class, model_kwargs, seed):
        """Label shuffle test: shuffled labels should drop to baseline floor.

        If performance stays high with shuffled labels, information is leaking
        around the labels.
        """
        kwargs = {**model_kwargs, 'random_state': seed}
        model = model_class(**kwargs)
        y_shuffled = self.y_train.copy()
        np.random.default_rng(seed).shuffle(y_shuffled)
        model.fit(self.X_train, y_shuffled)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        auc_shuffled = roc_auc_score(self.y_test, y_pred_proba)
        print(f"Sanity: label shuffle AUC = {auc_shuffled:.4f} (should be ~0.5)")
        return auc_shuffled

    def run_model(self, model_class, model_kwargs, seed):
        """Train model and evaluate on test set."""
        model = model_class(random_state=seed, **model_kwargs)
        model.fit(self.X_train, self.y_train)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        y_pred = model.predict(self.X_test)

        # Metrics
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        pr_auc = average_precision_score(self.y_test, y_pred_proba)
        f1 = f1_score(self.y_test, y_pred)

        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'f1': f1,
            'model': model,
        }

    def compare_models(self, verbose=True):
        """Compare logistic regression vs gradient boosting over multiple seeds."""
        lr_results = []
        gb_results = []

        print("\n=== SANITY CHECKS ===")
        # Sanity checks with first seed
        seed = self.seeds[0]
        from src.preprocessing import get_baseline_prediction
        baseline_pred = get_baseline_prediction(self.y_test)
        baseline_auc = self.sanity_check_baseline(baseline_pred)

        lr_kwargs = {'max_iter': 1000}
        gb_kwargs = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': seed}

        print(f"\nOverfit check with LogisticRegression:")
        self.sanity_check_overfit(LogisticRegression, lr_kwargs, seed)
        print(f"\nOverfit check with GradientBoosting:")
        self.sanity_check_overfit(GradientBoostingClassifier, gb_kwargs, seed)

        print(f"\nLabel shuffle test with LogisticRegression:")
        self.sanity_check_label_shuffle(LogisticRegression, lr_kwargs, seed)
        print(f"\nLabel shuffle test with GradientBoosting:")
        gb_kwargs_no_seed = {k: v for k, v in gb_kwargs.items() if k != 'random_state'}
        self.sanity_check_label_shuffle(GradientBoostingClassifier, gb_kwargs_no_seed, seed)

        print("\n=== MAIN EXPERIMENT (5 SEEDS) ===")
        for seed in self.seeds:
            lr_result = self.run_model(LogisticRegression, lr_kwargs, seed)
            gb_kwargs = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}
            gb_result = self.run_model(GradientBoostingClassifier, gb_kwargs, seed)

            lr_results.append(lr_result)
            gb_results.append(gb_result)

            if verbose:
                print(f"Seed {seed}:")
                print(f"  LR: ROC-AUC={lr_result['roc_auc']:.4f}, PR-AUC={lr_result['pr_auc']:.4f}, F1={lr_result['f1']:.4f}")
                print(f"  GB: ROC-AUC={gb_result['roc_auc']:.4f}, PR-AUC={gb_result['pr_auc']:.4f}, F1={gb_result['f1']:.4f}")

        # Summary statistics
        lr_aucs = [r['roc_auc'] for r in lr_results]
        gb_aucs = [r['roc_auc'] for r in gb_results]
        lr_pr_aucs = [r['pr_auc'] for r in lr_results]
        gb_pr_aucs = [r['pr_auc'] for r in gb_results]

        summary = {
            'baseline_auc': baseline_auc,
            'lr_roc_auc_mean': np.mean(lr_aucs),
            'lr_roc_auc_std': np.std(lr_aucs),
            'gb_roc_auc_mean': np.mean(gb_aucs),
            'gb_roc_auc_std': np.std(gb_aucs),
            'lr_pr_auc_mean': np.mean(lr_pr_aucs),
            'lr_pr_auc_std': np.std(lr_pr_aucs),
            'gb_pr_auc_mean': np.mean(gb_pr_aucs),
            'gb_pr_auc_std': np.std(gb_pr_aucs),
            'lr_roc_auc_values': lr_aucs,
            'gb_roc_auc_values': gb_aucs,
        }

        print("\n=== SUMMARY ===")
        print(f"LogisticRegression ROC-AUC: {summary['lr_roc_auc_mean']:.4f} ± {summary['lr_roc_auc_std']:.4f}")
        print(f"GradientBoosting ROC-AUC:   {summary['gb_roc_auc_mean']:.4f} ± {summary['gb_roc_auc_std']:.4f}")
        print(f"LogisticRegression PR-AUC:  {summary['lr_pr_auc_mean']:.4f} ± {summary['lr_pr_auc_std']:.4f}")
        print(f"GradientBoosting PR-AUC:    {summary['gb_pr_auc_mean']:.4f} ± {summary['gb_pr_auc_std']:.4f}")

        return summary
