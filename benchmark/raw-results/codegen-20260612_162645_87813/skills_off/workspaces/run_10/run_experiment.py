"""Entrypoint: run the full churn experiment and write artifacts.

Usage:
    python3 make_dataset.py --out churn.csv   # once, to create the data
    python3 run_experiment.py                 # runs comparison, writes outputs

Outputs:
    results/metrics.json   machine-readable metrics, config, seeds
    REPORT.md              human-readable conclusion, methodology, limitations
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import sklearn

from src import data as datamod
from src import experiment as exp
from src import report as reportmod


def _code_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="churn.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--report", default="REPORT.md")
    parser.add_argument("--seed", type=int, default=exp.SEED)
    parser.add_argument("--n-splits", type=int, default=exp.N_SPLITS)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(
            f"error: {data_path} not found. Generate it first:\n"
            f"  python3 make_dataset.py --out {data_path}",
            file=sys.stderr,
        )
        return 1

    raw = datamod.load_raw(str(data_path))
    clean = datamod.clean(raw)
    leaky_X = datamod.leaky_matrix(raw)

    result = exp.run(clean, leaky_X, seed=args.seed, n_splits=args.n_splits)

    # Record provenance alongside metrics (artifacts > scrollback).
    result["provenance"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "code_version": _code_version(),
        "data_command": f"python3 make_dataset.py --out {data_path}",
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2))

    Path(args.report).write_text(reportmod.render(result))

    comp = result["comparison"]
    print(comp["conclusion"])
    print(f"wrote {metrics_path} and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
