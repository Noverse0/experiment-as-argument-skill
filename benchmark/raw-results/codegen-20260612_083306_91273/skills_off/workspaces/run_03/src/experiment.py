"""Main experiment: logistic regression vs gradient boosting for churn."""

import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from .preprocessing import preprocess_and_split, get_baseline_prediction
from .metrics import compute_metrics


class ChurnExperiment:
    """Experiment comparing logistic regression vs gradient boosting."""

    def __init__(self, data_path: str, seeds: list):
        """
        Args:
            data_path: Path to churn.csv
            seeds: List of random seeds to use for each run
        """
        self.data_path = data_path
        self.seeds = seeds
        self.results = {
            'config': {
                'data_path': data_path,
                'seeds': seeds,
                'n_seeds': len(seeds),
                'test_size': 0.2,
                'lr_config': {
                    'max_iter': 1000,
                    'random_state': 'varies by seed',
                    'solver': 'lbfgs',
                },
                'gb_config': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'random_state': 'varies by seed',
                },
            },
            'runs': [],
        }

    def run_seed(self, seed: int) -> dict:
        """Run one seed of the experiment."""
        # Load and split data
        df = pd.read_csv(self.data_path)

        X_train, X_test, y_train, y_test, scaler = preprocess_and_split(
            df, test_size=0.2, random_state=seed, use_leaky_features=False
        )

        baseline_info = get_baseline_prediction(y_train, y_test)

        run_result = {
            'seed': seed,
            'data_split': {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'target_rate_train': baseline_info['target_rate_train'],
                'target_rate_test': baseline_info['target_rate_test'],
            },
            'baseline': baseline_info,
            'models': {},
        }

        # Train LogisticRegression
        lr = LogisticRegression(max_iter=1000, random_state=seed, solver='lbfgs')
        lr.fit(X_train, y_train)
        y_pred_proba_lr = lr.predict_proba(X_test)[:, 1]
        y_pred_lr = lr.predict(X_test)
        metrics_lr = compute_metrics(y_test, y_pred_proba_lr, y_pred_lr)

        run_result['models']['logistic_regression'] = metrics_lr

        # Train GradientBoostingClassifier
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=seed,
        )
        gb.fit(X_train, y_train)
        y_pred_proba_gb = gb.predict_proba(X_test)[:, 1]
        y_pred_gb = gb.predict(X_test)
        metrics_gb = compute_metrics(y_test, y_pred_proba_gb, y_pred_gb)

        run_result['models']['gradient_boosting'] = metrics_gb

        return run_result

    def run(self) -> dict:
        """Run full experiment across all seeds."""
        for seed in self.seeds:
            result = self.run_seed(seed)
            self.results['runs'].append(result)

        # Aggregate results
        self.results['summary'] = self._summarize()
        return self.results

    def _summarize(self) -> dict:
        """Summarize results across all seeds."""
        lr_metrics = []
        gb_metrics = []

        for run in self.results['runs']:
            lr_metrics.append(run['models']['logistic_regression'])
            gb_metrics.append(run['models']['gradient_boosting'])

        def aggregate(metric_list, metric_name):
            values = [m[metric_name] for m in metric_list]
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values,
            }

        summary = {
            'logistic_regression': {},
            'gradient_boosting': {},
        }

        for metric_name in ['roc_auc', 'f1', 'precision', 'recall', 'accuracy']:
            summary['logistic_regression'][metric_name] = aggregate(lr_metrics, metric_name)
            summary['gradient_boosting'][metric_name] = aggregate(gb_metrics, metric_name)

        # Compute the comparison
        lr_auc_mean = summary['logistic_regression']['roc_auc']['mean']
        gb_auc_mean = summary['gradient_boosting']['roc_auc']['mean']
        auc_diff = gb_auc_mean - lr_auc_mean

        summary['comparison'] = {
            'primary_metric': 'roc_auc',
            'gb_auc_mean': gb_auc_mean,
            'lr_auc_mean': lr_auc_mean,
            'difference': auc_diff,
            'gb_wins': auc_diff > 0,
        }

        return summary

    def to_json(self, output_path: str):
        """Write results to JSON file."""
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_types(v) for v in obj]
            elif isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            else:
                return obj

        serializable_results = convert_types(self.results)
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
