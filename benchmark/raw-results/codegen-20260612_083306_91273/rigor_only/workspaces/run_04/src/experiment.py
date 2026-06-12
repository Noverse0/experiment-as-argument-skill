"""Churn prediction experiment comparing LogisticRegression vs GradientBoostingClassifier."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.pipeline import (
    load_and_deduplicate,
    drop_leaky_features,
    prepare_features_and_target,
    fit_scaler,
    apply_scaling,
)


class ChurnExperiment:
    """Rigorous churn prediction experiment with sanity checks."""

    def __init__(self, csv_path: str, seed: int):
        self.csv_path = csv_path
        self.seed = seed
        self.results = {}
        self.sanity_checks = {}

    def run(self) -> dict:
        """Execute the full experiment pipeline."""
        # 1. Load and deduplicate
        df, n_duplicates = load_and_deduplicate(self.csv_path)
        self.results['duplicates_removed'] = n_duplicates
        self.results['n_samples'] = len(df)

        # 2. Drop leaky features
        df = drop_leaky_features(df)

        # 3. Extract features and target
        X, y = prepare_features_and_target(df)
        self.results['churn_rate'] = y.mean()
        self.results['n_features'] = X.shape[1]

        # 4. Sanity checks (before main experiment)
        self._run_sanity_checks(X, y)

        # 5. Run main comparison with multiple seeds/folds
        self._run_main_comparison(X, y)

        return self.results

    def _run_sanity_checks(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Run baseline, leakage, and pipeline sanity checks."""
        # Baseline: majority class predictor
        majority_class = y.value_counts().idxmax()
        baseline_pred = np.full_like(y, majority_class)
        baseline_f1 = f1_score(y, baseline_pred)
        self.sanity_checks['baseline_f1'] = float(baseline_f1)

        # Overfit test: tiny subset
        X_tiny = X.iloc[:50]
        y_tiny = y.iloc[:50]
        scaler = StandardScaler()
        X_tiny_scaled = scaler.fit_transform(X_tiny)

        clf = LogisticRegression(max_iter=1000, random_state=self.seed)
        clf.fit(X_tiny_scaled, y_tiny)
        tiny_loss = 1 - clf.score(X_tiny_scaled, y_tiny)
        self.sanity_checks['overfit_test_loss'] = float(tiny_loss)

        # Label shuffle: performance should drop to baseline
        y_shuffled = y.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_shuffled, test_size=0.2, random_state=self.seed, stratify=y_shuffled
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, random_state=self.seed)
        clf.fit(X_train_scaled, y_train)
        shuffle_f1 = f1_score(y_test, clf.predict(X_test_scaled))
        self.sanity_checks['label_shuffle_f1'] = float(shuffle_f1)

        # All sanity checks should pass
        assert self.sanity_checks['overfit_test_loss'] < 0.3, "Overfit test failed"
        assert self.sanity_checks['label_shuffle_f1'] <= baseline_f1 * 1.15, "Label shuffle leaked"

    def _run_main_comparison(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Run main experiment with 3 seeds and report variance."""
        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=None),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=None, max_depth=4),
        }

        results_by_model = {name: [] for name in models.keys()}

        # Run 3 times with different seeds
        for run in range(3):
            seed = self.seed + run
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed, stratify=y
            )

            # Fit scaler on train only, apply to test
            scaler = fit_scaler(X_train)
            X_train_scaled = apply_scaling(X_train, scaler)
            X_test_scaled = apply_scaling(X_test, scaler)

            for name, model in models.items():
                # Create a fresh clone with the same hyperparams but run-specific seed
                if name == 'LogisticRegression':
                    clf = LogisticRegression(max_iter=1000, random_state=seed)
                else:
                    clf = GradientBoostingClassifier(n_estimators=100, random_state=seed, max_depth=4)

                clf.fit(X_train_scaled, y_train)
                y_pred = clf.predict(X_test_scaled)
                y_proba = clf.predict_proba(X_test_scaled)[:, 1]

                metrics = {
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0),
                    'roc_auc': roc_auc_score(y_test, y_proba),
                }
                results_by_model[name].append(metrics)

        # Aggregate results: mean ± std
        self.results['model_comparison'] = {}
        for name in models.keys():
            metric_runs = results_by_model[name]
            aggregated = {}
            for metric_name in ['precision', 'recall', 'f1', 'roc_auc']:
                values = [m[metric_name] for m in metric_runs]
                aggregated[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'runs': values,
                }
            self.results['model_comparison'][name] = aggregated

    def sanity_summary(self) -> str:
        """Return a summary of sanity checks."""
        return (
            f"Baseline F1: {self.sanity_checks['baseline_f1']:.4f}\n"
            f"Overfit test loss (tiny subset): {self.sanity_checks['overfit_test_loss']:.4f}\n"
            f"Label shuffle F1 (should ≈ baseline): {self.sanity_checks['label_shuffle_f1']:.4f}"
        )

    def comparison_summary(self) -> str:
        """Return a human-readable comparison."""
        comp = self.results['model_comparison']
        lines = []
        for metric in ['f1', 'roc_auc', 'precision', 'recall']:
            lines.append(f"\n{metric.upper()}:")
            for model_name in ['LogisticRegression', 'GradientBoosting']:
                m = comp[model_name][metric]
                lines.append(
                    f"  {model_name}: {m['mean']:.4f} ± {m['std']:.4f} (n={len(m['runs'])})"
                )
        return "\n".join(lines)
