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


DEFAULT_LEAK_COLS = ["account_status", "days_since_last_login"]
_DROP_RE = re.compile(
    r"\bdrop\b|exclude|remove|\bdel\b|not in|!=|\.difference|\.pop\(|\bdropna\b"
)


_FEATURE_USE_RE = re.compile(r"df\[\[|features?\s*=|feature_cols|\bX\s*=|\bX\[")
_HANDLE_ASSIGN_RE = re.compile(r"(leak|exclude|drop)\w*\s*=", re.I)


def classify_leak(code: str, leaky_cols: list) -> str:
    """Objectively classify whether generated code leaks a target-derived column.

    Reviewer-independent cross-check: reads the code, not the review. Per column:
    - leaked:  the column sits inside an explicit feature-selection construct
               (df[[...]], X = ..., features = ...) with no drop on that line
    - handled: it appears with a drop/exclude keyword, or in a leak/drop/exclude
               list variable (e.g. LEAK_COLS = ["..."]) — including drops routed
               through a variable, which a line-local check would miss
    - review:  it appears but neither clearly used nor dropped (needs a glance)
    - absent:  the column name never appears
    Worst status wins (leaked > handled > review > absent). This is a heuristic,
    not full dataflow analysis: it is a fast cross-check on the reviewer, not a
    replacement. A `leaked` verdict is high-confidence; `review` means look.
    """
    statuses = []
    lines = code.splitlines()
    for col in leaky_cols:
        if col not in code:
            statuses.append("absent")
            continue
        used = any(
            col in ln and _FEATURE_USE_RE.search(ln) and not _DROP_RE.search(ln)
            for ln in lines
        )
        handled = any(
            col in ln and (_DROP_RE.search(ln) or _HANDLE_ASSIGN_RE.search(ln))
            for ln in lines
        )
        if used:
            statuses.append("leaked")
        elif handled:
            statuses.append("handled")
        else:
            statuses.append("review")
    for level in ("leaked", "handled", "review", "absent"):
        if level in statuses:
            return level
    return "absent"


def leak_check(run_dirs: list, leaky_cols: list) -> int:
    from collections import defaultdict as _dd
    tallies = _dd(lambda: _dd(int))
    for run_dir in run_dirs:
        for ws in sorted(Path(run_dir).glob("*/workspaces/run_*")):
            arm = ws.parent.parent.name
            code = "\n".join(
                p.read_text(errors="replace")
                for p in ws.rglob("*.py")
                if ".claude" not in p.parts and ".venv" not in p.parts
                and p.name != "make_dataset.py"  # the seeded fixture, not agent code
            )
            tallies[arm][classify_leak(code, leaky_cols)] += 1
    leaked_any = False
    for arm in sorted(tallies):
        t = tallies[arm]
        print(
            f"{arm}: leaked={t['leaked']} handled={t['handled']} "
            f"review={t['review']} absent={t['absent']}"
        )
        if t["leaked"]:
            leaked_any = True
    return 1 if leaked_any else 0


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


def aggregate(run_dirs: list):
    """Collect per-arm stats and pairwise significance from review run dirs."""
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
    arm_stats = []
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
        arm_stats.append({
            "arm": arm, "n": len(scores), "mean": statistics.mean(scores),
            "sd": sd, "cats": cats, "verdicts": dict(verdicts), "scores": scores,
        })
    pairwise = []
    by = {s["arm"]: s for s in arm_stats}
    names = [s["arm"] for s in arm_stats]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa, sb = by[a]["scores"], by[b]["scores"]
            if len(sa) > 1 and len(sb) > 1:
                _, z, p = mann_whitney(sa, sb)
                pairwise.append({
                    "a": a, "b": b, "diff": by[a]["mean"] - by[b]["mean"], "z": z, "p": p,
                })
    return arm_stats, pairwise, failures


def render_text(arm_stats, pairwise, failures) -> str:
    out = []
    for s in arm_stats:
        cat_str = " ".join(f"{c}={v:.1f}" for c, v in s["cats"].items())
        out.append(
            f"{s['arm']}: n={s['n']} mean={s['mean']:.1f} sd={s['sd']:.1f} "
            f"verdicts={s['verdicts']}\n    {cat_str}"
        )
    for pw in pairwise:
        out.append(
            f"{pw['a']} vs {pw['b']}: diff={pw['diff']:+.1f} "
            f"MW z={pw['z']:.2f} p={pw['p']:.3f}"
        )
    if failures:
        out.append(f"\nUNPARSEABLE ({len(failures)}):")
        out.extend(f"  {path}: {err}" for path, err in failures)
    return "\n".join(out)


def render_markdown(arm_stats, pairwise, failures) -> str:
    cats = REQUIRED_CATEGORIES
    header = (
        "| Arm | n | Weighted mean | sd | "
        + " | ".join(c.replace("_", " ") for c in cats)
        + " | Verdicts |"
    )
    sep = "| " + " | ".join(["---"] * (4 + len(cats) + 1)) + " |"
    rows = [header, sep]
    for s in arm_stats:
        verdicts = ", ".join(f"{k} {v}" for k, v in s["verdicts"].items())
        cat_vals = " | ".join(f"{s['cats'][c]:.1f}" for c in cats)
        rows.append(
            f"| `{s['arm']}` | {s['n']} | {s['mean']:.1f} | {s['sd']:.1f} | "
            f"{cat_vals} | {verdicts} |"
        )
    out = "\n".join(rows)
    if pairwise:
        out += "\n\n" + "\n".join(
            f"- `{pw['a']}` − `{pw['b']}`: diff {pw['diff']:+.1f}, "
            f"Mann-Whitney z={pw['z']:.2f}, p={pw['p']:.2f}"
            for pw in pairwise
        )
    if failures:
        out += f"\n\n**Unparseable: {len(failures)}** — rerun before trusting the table."
    return out


def render_csv(arm_stats) -> str:
    cats = REQUIRED_CATEGORIES
    lines = [",".join(["arm", "n", "weighted_mean", "sd"] + cats + ["verdicts"])]
    for s in arm_stats:
        verdicts = ";".join(f"{k}={v}" for k, v in s["verdicts"].items())
        row = (
            [s["arm"], str(s["n"]), f"{s['mean']:.1f}", f"{s['sd']:.1f}"]
            + [f"{s['cats'][c]:.1f}" for c in cats]
            + [verdicts]
        )
        lines.append(",".join(row))
    return "\n".join(lines)


RENDERERS = {"text": render_text, "markdown": render_markdown, "csv": render_csv}


def report(run_dirs: list, fmt: str = "text") -> int:
    arm_stats, pairwise, failures = aggregate(run_dirs)
    if fmt == "csv":
        print(render_csv(arm_stats))
    else:
        print(RENDERERS[fmt](arm_stats, pairwise, failures))
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="validate review files")
    v.add_argument("paths", nargs="+")
    r = sub.add_parser("report", help="aggregate review run directories")
    r.add_argument("run_dirs", nargs="+")
    r.add_argument(
        "--format", choices=["text", "markdown", "csv"], default="text",
        help="text (default), markdown table, or csv for import into a tracker",
    )
    lc = sub.add_parser(
        "leak-check",
        help="objectively check codegen workspaces for leaked target-derived columns",
    )
    lc.add_argument("run_dirs", nargs="+", help="codegen run dir(s)")
    lc.add_argument(
        "--cols", nargs="+", default=DEFAULT_LEAK_COLS,
        help=f"leaky column names (default: {' '.join(DEFAULT_LEAK_COLS)})",
    )
    args = parser.parse_args()
    if args.cmd == "validate":
        return validate(args.paths)
    if args.cmd == "leak-check":
        return leak_check(args.run_dirs, args.cols)
    return report(args.run_dirs, args.format)


if __name__ == "__main__":
    sys.exit(main())
