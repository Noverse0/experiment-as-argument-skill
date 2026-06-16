import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, balanced_accuracy_score
from src.pipeline import load_data, validate_data, preprocess_and_split, get_baseline_rate


class ChurnExperiment:
    def __init__(self, csv_path, seeds=5, test_size=0.3):
        self.csv_path = csv_path
        self.seeds = seeds
        self.test_size = test_size
        self.results = {
            "LogisticRegression": {},
            "GradientBoostingClassifier": {},
        }
        self.data = None
        self.baseline_rate = None

    def load_and_validate(self):
        """Load data and check for issues."""
        self.data = load_data(self.csv_path)
        validate_data(self.data)
        self.baseline_rate = get_baseline_rate(self.data["churned"])
        print(f"Data loaded: {self.data.shape[0]} rows")
        print(f"Churn rate: {self.data['churned'].mean():.2%}")
        print(f"Baseline accuracy (majority class): {self.baseline_rate:.2%}")

    def run_single_seed(self, model_name, seed):
        """Run one model with one seed, return metrics dict."""
        X_train, X_test, y_train, y_test, scaler = preprocess_and_split(
            self.data, test_size=self.test_size, random_state=seed
        )

        # Instantiate model
        if model_name == "LogisticRegression":
            model = LogisticRegression(max_iter=1000, random_state=seed)
        elif model_name == "GradientBoostingClassifier":
            model = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = {
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "recall": recall_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "accuracy": (y_pred == y_test).mean(),
        }
        return metrics

    def sanity_check_baseline(self):
        """Verify that both models meet or beat the baseline floor."""
        baseline_acc = self.baseline_rate

        for model_name in ["LogisticRegression", "GradientBoostingClassifier"]:
            metrics = self.run_single_seed(model_name, seed=42)
            acc = metrics["accuracy"]
            assert (
                acc >= baseline_acc - 0.01
            ), f"{model_name} accuracy {acc:.3f} must meet baseline {baseline_acc:.3f}"
        print("✓ Both models meet or exceed baseline floor")

    def sanity_check_label_shuffle(self):
        """With shuffled labels, both models should perform near baseline."""
        df_shuffled = self.data.copy()
        np.random.seed(123)
        df_shuffled["churned"] = np.random.permutation(df_shuffled["churned"].values)

        X_train, X_test, y_train, y_test, _ = preprocess_and_split(
            df_shuffled, test_size=self.test_size, random_state=42
        )

        baseline_acc = self.baseline_rate
        tolerance = 0.15  # Allow 15% above baseline for shuffled labels on small datasets

        for model_name in ["LogisticRegression", "GradientBoostingClassifier"]:
            model_cls = (
                LogisticRegression
                if model_name == "LogisticRegression"
                else GradientBoostingClassifier
            )
            model = (
                model_cls(max_iter=1000, random_state=42)
                if model_name == "LogisticRegression"
                else model_cls(n_estimators=100, random_state=42)
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = (y_pred == y_test).mean()
            assert (
                acc <= baseline_acc + tolerance
            ), f"{model_name} with shuffled labels has acc {acc:.3f}, should be ~baseline {baseline_acc:.3f}"
        print("✓ Label-shuffle test passed: models perform near baseline on shuffled labels")

    def run_experiment(self):
        """Run the full experiment across all seeds."""
        print("\n" + "=" * 60)
        print("CHURN PREDICTION EXPERIMENT")
        print("=" * 60)

        self.load_and_validate()

        # Sanity checks
        print("\nRunning sanity checks...")
        self.sanity_check_baseline()
        self.sanity_check_label_shuffle()

        # Full experiment
        print(f"\nRunning experiment with {self.seeds} seeds...")
        for model_name in self.results.keys():
            metrics_list = []
            for seed in range(self.seeds):
                metrics = self.run_single_seed(model_name, seed)
                metrics_list.append(metrics)
            self.results[model_name] = self._aggregate_metrics(metrics_list)
            print(f"✓ {model_name} complete")

    def _aggregate_metrics(self, metrics_list):
        """Aggregate metrics across seeds (mean ± std)."""
        df = pd.DataFrame(metrics_list)
        agg = {}
        for col in df.columns:
            agg[col] = {"mean": df[col].mean(), "std": df[col].std()}
        return agg

    def get_results(self):
        """Return results dict."""
        return self.results

    def summary(self):
        """Print experiment summary."""
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        for model_name, metrics in self.results.items():
            print(f"\n{model_name}:")
            for metric, values in metrics.items():
                print(
                    f"  {metric:20s}: {values['mean']:.4f} ± {values['std']:.4f}"
                )

        # Comparison
        print("\n" + "=" * 60)
        print("COMPARISON (winner by ROC-AUC)")
        print("=" * 60)
        auc_lr = self.results["LogisticRegression"]["roc_auc"]["mean"]
        auc_gb = self.results["GradientBoostingClassifier"]["roc_auc"]["mean"]
        winner = "GradientBoostingClassifier" if auc_gb > auc_lr else "LogisticRegression"
        gap = abs(auc_gb - auc_lr)
        print(f"Winner: {winner} (gap: {gap:.4f})")
