"""Entrypoint: run the full churn experiment and write artifacts.

Usage:
    python3 make_dataset.py --out churn.csv   # generate data first
    python3 run_experiment.py                 # runs comparison, writes results/

Writes:
    results/metrics.json  -- machine-readable config, seeds, sanity, scores
    REPORT.md             -- human-readable conclusion, methodology, limitations
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.data import prepare
from src.experiment import run_full_experiment
from src.report import render_report


def _code_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="churn.csv")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--report", default="REPORT.md")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(
            f"error: {data_path} not found. Run: python3 make_dataset.py --out {data_path}",
            file=sys.stderr,
        )
        return 1

    data = prepare(str(data_path))
    result = run_full_experiment(data, str(data_path))
    result["code_version"] = _code_version()
    result["data_command"] = f"python3 make_dataset.py --out {data_path}"

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2))

    Path(args.report).write_text(render_report(result))

    print(f"conclusion: {result['conclusion']}")
    print(f"wrote {metrics_path} and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
