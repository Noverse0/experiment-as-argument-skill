"""End-to-end test: the experiment runs, writes artifacts, and emits an honest
conclusion of a recognized shape."""
from __future__ import annotations

import json
from pathlib import Path

from src.experiment import render_report, run


def test_run_writes_artifacts_and_conclusion(churn_csv, tmp_path):
    results_dir = tmp_path / "results"
    metrics = run(churn_csv, str(results_dir), n_splits=3)

    # machine-readable artifacts exist
    assert (results_dir / "metrics.json").exists()
    assert (results_dir / "summary.csv").exists()

    saved = json.loads((results_dir / "metrics.json").read_text())
    assert set(saved["arms"]) == {"logistic_regression", "gradient_boosting"}

    # conclusion is one of the two honest verdicts
    assert metrics["conclusion"]["verdict"] in {"winner", "no_detectable_difference"}

    # both arms must beat the chance floor (sanity)
    for arm in metrics["arms"].values():
        assert arm["roc_auc_mean"] > 0.5

    # seeds and config recorded
    assert "seeds" in saved and "config" in saved
    assert saved["data"]["n_duplicates_removed"] > 0


def test_render_report_contains_key_sections(churn_csv, tmp_path):
    metrics = run(churn_csv, str(tmp_path / "results"), n_splits=3)
    report = tmp_path / "REPORT.md"
    render_report(metrics, str(report))
    text = report.read_text()
    for needle in ["## Conclusion", "## Methodology", "## Limitations", "account_status"]:
        assert needle in text
