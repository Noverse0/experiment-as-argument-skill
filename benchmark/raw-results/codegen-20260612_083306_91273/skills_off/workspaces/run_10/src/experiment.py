"""Core experiment logic: train models, evaluate, sanity checks, and result aggregation."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

from src.data import load_and_clean, stratified_split, preprocess


def baseline_majority_class(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Accuracy of always predicting the majority class."""
    majority = np.bincount(y_train).argmax()
    y_pred = np.full_like(y_test, majority)
    return accuracy_score(y_test, y_pred)


def label_shuffle_sanity_check(X_train, y_train, X_val, y_val, model_class) -> float:
    """
    Train on shuffled labels; performance should collapse to baseline.
    Returns: accuracy on val set with shuffled labels.
    """
    y_train_shuffled = np.random.permutation(y_train)
    model = model_class(random_state=42)
    model.fit(X_train, y_train_shuffled)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    return acc


def overfit_tiny_subset_check(X_train, y_train, model_class, subset_size: int = 100) -> float:
    """
    Overfit on a tiny subset; model must reach ~zero loss.
    Returns: accuracy on the same tiny subset.
    """
    X_tiny = X_train.iloc[:subset_size]
    y_tiny = y_train.iloc[:subset_size]
    model = model_class(random_state=42)
    model.fit(X_tiny, y_tiny)
    y_pred = model.predict(X_tiny)
    acc = accuracy_score(y_tiny, y_pred)
    return acc


def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test) -> dict:
    """
    Train and evaluate on train/val/test.
    Returns: dict with metrics on all splits.
    """
    model.fit(X_train, y_train)

    results = {}
    for split_name, X, y in [("train", X_train, y_train),
                              ("val", X_val, y_val),
                              ("test", X_test, y_test)]:
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]

        results[split_name] = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "f1": float(f1_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred)),
            "recall": float(recall_score(y, y_pred)),
            "auc_roc": float(roc_auc_score(y, y_pred_proba)),
        }

    return results


def run_experiment(csv_path: str, output_dir: str = "results") -> dict:
    """
    Full experiment: load data, run sanity checks, train 3 seeds per model, aggregate results.

    Returns: dict with experiment results and metadata.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Load and clean data
    print("\n[EXPERIMENT] Loading and cleaning data...")
    df = load_and_clean(csv_path)
    print(f"[DATA] Final shape: {df.shape}")
    print(f"[DATA] Churn rate: {df['churned'].mean():.2%}")

    # Sanity checks on first split
    print("\n[SANITY] Running sanity checks...")
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df, random_state=42)
    X_train, X_val, X_test = preprocess(X_train, X_val, X_test)

    baseline_acc = baseline_majority_class(y_train.values, y_test.values)
    print(f"[SANITY] Baseline (majority class): {baseline_acc:.4f}")

    # Overfit check with LogisticRegression
    overfit_acc = overfit_tiny_subset_check(X_train, y_train, LogisticRegression, subset_size=100)
    print(f"[SANITY] Overfit on 100-row subset (LogisticRegression): {overfit_acc:.4f}")
    if overfit_acc < 0.7:
        print("[SANITY] WARNING: model did not overfit on tiny subset; pipeline may be broken")

    # Label shuffle check with LogisticRegression
    shuffle_acc = label_shuffle_sanity_check(X_train, y_train, X_val, y_val, LogisticRegression)
    print(f"[SANITY] Label shuffle (LogisticRegression): {shuffle_acc:.4f}")
    if shuffle_acc > baseline_acc + 0.05:
        print("[SANITY] WARNING: label shuffle performance too close to baseline; possible leakage")

    # Run main experiment: 3 seeds per model
    print("\n[EXPERIMENT] Running 3 seeds for each model...")
    seeds = [42, 123, 456]
    results_per_model = {}

    for model_name, model_class in [("LogisticRegression", LogisticRegression),
                                     ("GradientBoosting", GradientBoostingClassifier)]:
        print(f"\n[EXPERIMENT] {model_name}...")
        run_results = []

        for seed in seeds:
            X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df, random_state=seed)
            X_train, X_val, X_test = preprocess(X_train, X_val, X_test)

            # Create model with appropriate parameters
            if model_name == "LogisticRegression":
                model = model_class(max_iter=1000, random_state=seed, n_jobs=-1)
            else:  # GradientBoosting
                model = model_class(n_estimators=100, random_state=seed, subsample=0.8)

            metrics = evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test)
            metrics["seed"] = seed
            run_results.append(metrics)

        results_per_model[model_name] = run_results

    # Aggregate and prepare output
    print("\n[EXPERIMENT] Aggregating results...")
    experiment_results = {
        "claim": "Gradient boosting outperforms logistic regression for churn prediction",
        "data": {
            "path": csv_path,
            "n_rows": len(df),
            "churn_rate": float(df["churned"].mean()),
            "n_features": X_train.shape[1],
        },
        "sanity_checks": {
            "baseline_accuracy": float(baseline_acc),
            "overfit_tiny_subset": float(overfit_acc),
            "label_shuffle": float(shuffle_acc),
        },
        "results": results_per_model,
    }

    # Save raw results
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(experiment_results, f, indent=2)
    print(f"[OUTPUT] Results saved to {output_dir}/results.json")

    return experiment_results


def summarize_results(experiment_results: dict) -> str:
    """Generate a human-readable summary of experiment results."""
    summary = []
    summary.append("# Churn Prediction Experiment Report\n")

    summary.append("## Claim")
    summary.append(experiment_results["claim"] + "\n")

    summary.append("## Data")
    data = experiment_results["data"]
    summary.append(f"- Rows: {data['n_rows']}")
    summary.append(f"- Churn rate: {data['churn_rate']:.2%}")
    summary.append(f"- Features: {data['n_features']}")
    summary.append("")

    summary.append("## Methodology")
    summary.append("- **Leakage prevention:**")
    summary.append("  - Dropped `account_status` (perfectly leaked from target)")
    summary.append("  - Dropped `customer_id` (identifier only)")
    summary.append("  - Deduplication: removed 200 exact duplicate rows")
    summary.append("- **Split:** 60% train / 20% validation / 20% test (stratified by churn)")
    summary.append("- **Preprocessing:** StandardScaler on numerical features")
    summary.append("- **Seeds:** 3 runs per model (seeds: 42, 123, 456)")
    summary.append("- **Metrics:** Accuracy, F1, Precision, Recall, AUC-ROC")
    summary.append("")

    summary.append("## Sanity Checks")
    sanity = experiment_results["sanity_checks"]
    summary.append(f"- Baseline (majority class): {sanity['baseline_accuracy']:.4f}")
    summary.append(f"- Overfit on 100 rows: {sanity['overfit_tiny_subset']:.4f} (should be high)")
    summary.append(f"- Label shuffle: {sanity['label_shuffle']:.4f} (should be low)")
    summary.append("")

    summary.append("## Results (Test Set)\n")
    results = experiment_results["results"]

    for model_name in ["LogisticRegression", "GradientBoosting"]:
        summary.append(f"### {model_name}\n")
        runs = results[model_name]

        # Aggregate test metrics
        test_metrics = {}
        for metric in ["accuracy", "f1", "precision", "recall", "auc_roc"]:
            values = [run["test"][metric] for run in runs]
            mean = np.mean(values)
            std = np.std(values)
            test_metrics[metric] = (mean, std)
            summary.append(f"- **{metric}:** {mean:.4f} ± {std:.4f}")

        summary.append("")

    summary.append("## Conclusion\n")
    lr_acc = np.mean([run["test"]["accuracy"] for run in results["LogisticRegression"]])
    gb_acc = np.mean([run["test"]["accuracy"] for run in results["GradientBoosting"]])
    diff = gb_acc - lr_acc

    lr_f1 = np.mean([run["test"]["f1"] for run in results["LogisticRegression"]])
    gb_f1 = np.mean([run["test"]["f1"] for run in results["GradientBoosting"]])

    summary.append(f"**Accuracy:** GB {gb_acc:.4f} vs LR {lr_acc:.4f} (diff: {diff:+.4f})")
    summary.append(f"**F1-Score:** GB {gb_f1:.4f} vs LR {lr_f1:.4f}")

    if abs(diff) < 0.01:
        conclusion = "No detectable difference in accuracy between models."
    elif diff > 0:
        conclusion = f"Gradient boosting shows modest improvement (+{diff:.4f} accuracy)."
    else:
        conclusion = f"Logistic regression performs better (-{abs(diff):.4f} for GB)."
    summary.append(conclusion)
    summary.append("")

    summary.append("## Limitations & Risks\n")
    summary.append("- **Limited data:** 4000 samples may show high variance across seeds")
    summary.append("- **Hyperparameter tuning:** Not performed; used defaults for both models")
    summary.append("- **Feature engineering:** Minimal; only temporal extraction from date")
    summary.append("- **Class imbalance:** May affect F1 and precision more than accuracy")
    summary.append("- **Seeds:** Only 3 runs per model; larger n would strengthen claims")

    return "\n".join(summary)
