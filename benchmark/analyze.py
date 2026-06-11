"""Strict parser and statistics for benchmark review outputs.

Design constraints learned from the programming-as-theory-building benchmark:
- reviews must be parsed strictly (fence required, all fields, ranges checked),
- the weighted total is recomputed from category scores, never trusted,
- aggregates report n/sd/significance so winners are not declared from noise.
"""
import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REQUIRED_CATEGORIES = [
    "leakage_prevention",
    "methodological_validity",
    "reproducibility",
    "claims_discipline",
    "executability",
    "code_quality",
]
VERDICTS = {"excellent", "good", "mixed", "poor"}
FENCE = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)


def extract_review(text: str) -> dict:
    m = FENCE.search(text)
    if not m:
        raise ValueError("no json fence found")
    try:
        d = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid json in fence: {e}") from e
    missing = [c for c in REQUIRED_CATEGORIES if f"{c}_score" not in d]
    if missing:
        raise ValueError(f"missing score fields: {missing}")
    if "weights" not in d:
        raise ValueError("missing weights")
    if d.get("verdict") not in VERDICTS:
        raise ValueError(f"missing or unknown verdict: {d.get('verdict')!r}")
    for c in REQUIRED_CATEGORIES:
        v = d[f"{c}_score"]
        if not isinstance(v, (int, float)) or not 0 <= v <= 100:
            raise ValueError(f"score out of range: {c}={v!r}")
    return d


def recompute_weighted(d: dict) -> float:
    return sum(d["weights"][c] * d[f"{c}_score"] for c in REQUIRED_CATEGORIES)


def mann_whitney(a: list, b: list):
    """U statistic, z, two-sided p via normal approximation with tie-averaged ranks."""
    pairs = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        for k in range(i, j):
            ranks[k] = (i + j + 1) / 2
        i = j
    ra = sum(r for r, (_, grp) in zip(ranks, pairs) if grp == 0)
    na, nb = len(a), len(b)
    u = ra - na * (na + 1) / 2
    mu = na * nb / 2
    sigma = math.sqrt(na * nb * (na + nb + 1) / 12)
    z = 0.0 if sigma == 0 else (u - mu) / sigma
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return u, z, p


def validate(paths: list) -> int:
    bad = 0
    for p in paths:
        try:
            extract_review(Path(p).read_text(errors="replace"))
            print(f"OK   {p}")
        except ValueError as e:
            bad += 1
            print(f"FAIL {p}: {e}")
    return 1 if bad else 0


def report(run_dirs: list) -> int:
    arms = defaultdict(list)
    failures = []
    for run_dir in run_dirs:
        for review in sorted(Path(run_dir).glob("*/run_*/review.txt")):
            arm = review.parent.parent.name
            try:
                d = extract_review(review.read_text(errors="replace"))
            except ValueError as e:
                failures.append((str(review), str(e)))
                continue
            arms[arm].append((recompute_weighted(d), d))
    for arm in sorted(arms):
        scores = [s for s, _ in arms[arm]]
        verdicts = defaultdict(int)
        for _, d in arms[arm]:
            verdicts[d["verdict"]] += 1
        sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
        cats = {
            c: statistics.mean(d[f"{c}_score"] for _, d in arms[arm])
            for c in REQUIRED_CATEGORIES
        }
        cat_str = " ".join(f"{c}={v:.1f}" for c, v in cats.items())
        print(
            f"{arm}: n={len(scores)} mean={statistics.mean(scores):.1f} "
            f"sd={sd:.1f} verdicts={dict(verdicts)}\n    {cat_str}"
        )
    arm_names = sorted(arms)
    for i, a in enumerate(arm_names):
        for b in arm_names[i + 1 :]:
            sa = [s for s, _ in arms[a]]
            sb = [s for s, _ in arms[b]]
            if len(sa) > 1 and len(sb) > 1:
                _, z, p = mann_whitney(sa, sb)
                diff = statistics.mean(sa) - statistics.mean(sb)
                print(f"{a} vs {b}: diff={diff:+.1f} MW z={z:.2f} p={p:.3f}")
    if failures:
        print(f"\nUNPARSEABLE ({len(failures)}):")
        for path, err in failures:
            print(f"  {path}: {err}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="validate review files")
    v.add_argument("paths", nargs="+")
    r = sub.add_parser("report", help="aggregate review run directories")
    r.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()
    if args.cmd == "validate":
        return validate(args.paths)
    return report(args.run_dirs)


if __name__ == "__main__":
    sys.exit(main())
