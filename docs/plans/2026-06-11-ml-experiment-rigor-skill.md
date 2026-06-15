# ML Experiment Rigor Skill Implementation Plan

> **Historical snapshot (2026-06-11).** This is the original build plan and is kept as a record. Names and scope have since moved on: the skill was renamed `ml-experiment-rigor` → `experiment-as-argument`, the repo to `Noverse0/experiment-as-argument-skill`, Codex/Gemini mirrors and an experiments ledger were added, and four benchmark experiments have run. For current state see [README.md](../../README.md) and [benchmark/EXPERIMENTS.md](../../benchmark/EXPERIMENTS.md); the old name below is left as-written.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `ml-experiment-rigor-skill` — a Claude Code skill distilling Karpathy's "A Recipe for Training Neural Networks" and the Kapoor & Narayanan leakage taxonomy into agent operating rules for ML training experiment code, with an in-repo isolated-arm benchmark proving its effect.

**Architecture:** Mirrors `programming-as-theory-building-skill` (skills/<name>/SKILL.md + .claude-plugin metadata + benchmark/), but fixes three weaknesses found in that repo's benchmark: the harness ships inside the repo (reproducible), review outputs are JSON-validated at run time (no silent broken reviews), and the analyzer reports n/sd/significance (no winner claims from noise).

**Tech Stack:** Markdown skill, bash harness scripts driving the `claude` CLI, Python 3 (stdlib + numpy/pandas for the dataset fixture) for analysis, pytest for analyzer tests.

**Repo location:** `/Users/rohdaeyoung/workspace/ml-experiment-rigor-skill`, published to GitHub as `Noverse0/ml-experiment-rigor-skill` (gh CLI already on Noverse0).

---

## File Structure

```
ml-experiment-rigor-skill/
├── .claude-plugin/
│   ├── plugin.json                  # Claude Code plugin metadata
│   └── marketplace.json             # marketplace metadata
├── .gitignore                       # ignores benchmark/runs/, caches
├── CLAUDE.md                        # root mirror of the skill rules
├── LICENSE                          # MIT
├── CITATION.cff
├── README.md                        # identity, install, benchmark summary
├── docs/plans/                      # this plan
├── skills/
│   └── ml-experiment-rigor/
│       └── SKILL.md                 # the skill itself (core deliverable)
└── benchmark/
    ├── README.md                    # method + results (filled after runs)
    ├── prompts/ml-experiment-v1.txt # codegen benchmark prompt
    ├── rubric/ml-experiment-rigor-review-v1.md  # reviewer prompt
    ├── fixtures/make_dataset.py     # deterministic dataset with planted traps
    ├── run_codegen_experiment.sh    # arm × repeat codegen harness
    ├── run_review_experiment.sh     # Opus review harness w/ JSON validation
    ├── analyze.py                   # strict parser + stats reporter
    └── tests/test_analyze.py        # pytest for analyze.py
```

Benchmark arms for v1: `skills_off` (no skill in workspace) and `rigor_only` (skill copied into workspace `.claude/skills/`). `ARMS` env var overrides.

---

### Task 1: Repo scaffold and metadata

**Files:**
- Create: `.gitignore`, `LICENSE`, `CITATION.cff`, `.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json`

- [ ] **Step 1: Init git repo**

```bash
cd /Users/rohdaeyoung/workspace/ml-experiment-rigor-skill
git init -b main
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
benchmark/runs/
__pycache__/
.pytest_cache/
*.csv
.DS_Store
```

(`*.csv` keeps generated benchmark datasets out; curated raw results, if published later, are copied under `benchmark/raw-results/` with explicit `git add -f` of the files we want.)

- [ ] **Step 3: Write `LICENSE`** — standard MIT text, copyright line: `Copyright (c) 2026 ml-experiment-rigor-skill contributors`.

- [ ] **Step 4: Write `CITATION.cff`**

```yaml
cff-version: 1.2.0
message: "If you use this skill or benchmark, please cite it."
title: "ML Experiment Rigor Skill"
type: software
authors:
  - name: "ml-experiment-rigor-skill contributors"
repository-code: "https://github.com/Noverse0/ml-experiment-rigor-skill"
license: MIT
references:
  - type: blog
    title: "A Recipe for Training Neural Networks"
    authors:
      - family-names: Karpathy
        given-names: Andrej
    year: 2019
    url: "https://karpathy.github.io/2019/04/25/recipe/"
  - type: article
    title: "Leakage and the Reproducibility Crisis in ML-based Science"
    authors:
      - family-names: Kapoor
        given-names: Sayash
      - family-names: Narayanan
        given-names: Arvind
    year: 2023
```

- [ ] **Step 5: Write `.claude-plugin/plugin.json`**

```json
{
  "name": "ml-experiment-rigor-skill",
  "description": "ML training experiment rigor rules for coding agents: leakage prevention, sanity checks before full training, seed and variance discipline, and claims backed by repeated runs.",
  "version": "0.1.0",
  "author": {
    "name": "ml-experiment-rigor-skill contributors"
  },
  "license": "MIT",
  "keywords": [
    "machine-learning",
    "skills",
    "reproducibility",
    "data-leakage",
    "experiment-rigor"
  ],
  "skills": [
    "./skills/ml-experiment-rigor"
  ]
}
```

- [ ] **Step 6: Write `.claude-plugin/marketplace.json`**

```json
{
  "name": "ml-experiment-rigor-skill",
  "description": "Claude Code marketplace for the ML Experiment Rigor skill.",
  "version": "0.1.0",
  "owner": {
    "name": "Noverse0"
  },
  "plugins": [
    {
      "name": "ml-experiment-rigor-skill",
      "source": "./",
      "description": "ML experiment rigor rules: split before transform, kill leaky features, sanity-check before full training, never claim a winner from one seed.",
      "version": "0.1.0",
      "author": {
        "name": "ml-experiment-rigor-skill contributors"
      },
      "keywords": [
        "claude-code",
        "machine-learning",
        "reproducibility",
        "data-leakage",
        "experiment-rigor"
      ],
      "category": "workflow"
    }
  ]
}
```

- [ ] **Step 7: Validate JSON and commit**

```bash
python3 -m json.tool .claude-plugin/plugin.json > /dev/null
python3 -m json.tool .claude-plugin/marketplace.json > /dev/null
git add .gitignore LICENSE CITATION.cff .claude-plugin docs
git commit -m "chore: scaffold repo with plugin metadata"
```

---

### Task 2: SKILL.md — the operating rules (core deliverable)

**Files:**
- Create: `skills/ml-experiment-rigor/SKILL.md`

- [ ] **Step 1: Write `skills/ml-experiment-rigor/SKILL.md`** with exactly this content:

````markdown
---
name: ml-experiment-rigor
description: "Use when writing, modifying, debugging, or reviewing ML training or evaluation code: experiments, model comparisons, data preprocessing, train/test splits, metrics, or result reports. Enforces leakage prevention, sanity checks before full training, seed and variance discipline, and claims backed by repeated runs."
license: MIT
---

# ML Experiment Rigor

Operating rules distilled from Andrej Karpathy's "A Recipe for Training Neural Networks" and Kapoor & Narayanan's leakage taxonomy. Use these rules to avoid experiments that run but prove nothing: leaked features, single-seed winner claims, and metrics that evaporate on a clean split.

**Tradeoff:** This skill slows down quick scripts. For anything whose output will be compared, reported, or believed, pay the upfront cost so the result survives scrutiny.

## Core Idea

An experiment is an argument, not a script. The code exists to support a claim ("A beats B on task T"), and every shortcut that weakens the argument — leakage, untested assumptions, unrepeated runs — silently converts the result into noise. Code that runs and prints a number is not evidence.

## Before Training

Answer these briefly before writing experiment code:

1. **Claim:** What exact comparison or hypothesis will the final report state? One sentence.
2. **Variable:** What is the single thing being varied? Everything else must be held fixed, including tuning budget.
3. **Data contact policy:** Which rows/columns may the model see at fit time? When is the test set allowed to be touched? (Answer: once, at the end.)
4. **Leak surface:** Which features could encode the target, the future, or the split? List suspects before coding.
5. **Proof of life:** What cheap sanity check will show the pipeline works before the full run?

If you cannot answer 1–3, the experiment is not designed yet. Stop and design it.

## Data Discipline

- **Split before transform.** Any fit-like operation (scaler, vocabulary, imputation, feature selection, target encoding) happens after the split, fitted on train only, applied to the rest.
- **Hunt target leakage.** Drop or justify every feature that could be derived from the label or recorded after the outcome (status flags, closure dates, post-hoc aggregates). Justify in code comments or the report, not in your head.
- **Deduplicate across the boundary.** Exact or near-duplicate rows must not straddle train/test. Check for duplicates before splitting; say in the report how many you found.
- **Respect time.** If any column is temporal and the task is forward-looking, use a time-based split. Random splits on temporal data are leakage.
- **Class balance is a fact, not a footnote.** Report the target rate; choose metrics that survive imbalance (not accuracy alone).

## Sanity Checks Before The Full Run

Run these before believing any training loop; they cost minutes and catch most silent bugs:

- **Baseline floor:** a trivial baseline (majority class, mean prediction) — your model must beat it.
- **Leakage ceiling:** if test performance looks too good (near-perfect on a noisy task), assume leakage and audit features before celebrating.
- **Overfit one batch / tiny subset:** the model must reach ~zero loss on a tiny slice. If it cannot, the pipeline is broken.
- **Label-shuffle test:** with shuffled labels, performance must fall to the baseline floor. If it does not, information is leaking around the labels.
- **Init loss check (when applicable):** loss at initialization should match the theoretical value (e.g., -log(1/C) for C balanced classes).

## Seeds and Repetition

- Fix and log every seed (framework, numpy, data shuffling, split).
- One seed is an anecdote. Before comparing methods, run ≥3 seeds (or CV folds) per arm and report mean ± sd and n.
- Identical pipelines must produce identical metrics when re-run with the same seed. If they do not, find the nondeterminism before proceeding.

## Claims Discipline

- No winner claims without variance. If the gap between arms is within noise (overlapping spreads, no test), the honest claim is "no detectable difference."
- Report effect size with its uncertainty, not just "X is better."
- The test set is touched once. Any decision made after seeing test metrics (feature change, hyperparameter, early stop) converts test into validation — say so and re-split.
- Negative and null results go in the report. Deleting failed runs is fabrication by omission.
- The report may not claim anything the code did not measure. No "robust," "significant," or "consistently" without the numbers behind it.

## Artifacts

Every run records: config (all hyperparameters), seeds, data version or generation command, code version, and resulting metrics — in a file, not in the console scrollback. Failed runs keep their artifacts.

## Verification

- **New experiment:** run the sanity checks above, then the smallest full run that supports the claim.
- **Modified experiment:** re-run the unchanged baseline arm; if its numbers moved, the change leaked into the control.
- **Review:** check, in order — leak surface, split-before-transform, dedup, seed logging, repetition behind every comparative claim, and report/code consistency.

## Stop Signals

Stop and re-design instead of patching if:

- test metrics improve when you add a feature you cannot explain,
- performance is near-perfect on a task that should be noisy,
- the result changes materially across seeds but the report claims one winner,
- preprocessing code touches the full dataset before the split,
- the report's claim is stronger than what n runs with this variance can support.

## Response Shape

For experiment work, summarize briefly:

```text
Claim: [what the experiment argues]
Design: [variable, split policy, seeds × repeats]
Sanity: [checks run and outcomes]
Result: [mean ± sd per arm, n, and the honest conclusion]
Risk: [remaining leak surface or validity threats]
```
````

- [ ] **Step 2: Validate frontmatter parses**

```bash
python3 - <<'EOF'
import re
text = open("skills/ml-experiment-rigor/SKILL.md").read()
m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
assert m, "frontmatter missing"
assert "name: ml-experiment-rigor" in m.group(1)
assert "description:" in m.group(1)
print("frontmatter OK")
EOF
```

Expected: `frontmatter OK`

- [ ] **Step 3: Commit**

```bash
git add skills/
git commit -m "feat: add ml-experiment-rigor skill rules"
```

---

### Task 3: Root CLAUDE.md mirror

**Files:**
- Create: `CLAUDE.md`

- [ ] **Step 1: Write `CLAUDE.md`** — copy the body of `skills/ml-experiment-rigor/SKILL.md` **without the frontmatter** (everything from `# ML Experiment Rigor` to the end), exactly as the reference repo mirrors its skill into root CLAUDE.md.

```bash
python3 - <<'EOF'
import re
text = open("skills/ml-experiment-rigor/SKILL.md").read()
body = re.sub(r"^---\n.*?\n---\n\n?", "", text, flags=re.S)
open("CLAUDE.md", "w").write(body)
print("CLAUDE.md written,", len(body), "chars")
EOF
```

- [ ] **Step 2: Verify mirror has no drift**

```bash
python3 - <<'EOF'
import re
skill = re.sub(r"^---\n.*?\n---\n\n?", "", open("skills/ml-experiment-rigor/SKILL.md").read(), flags=re.S)
assert open("CLAUDE.md").read() == skill, "CLAUDE.md drifted from SKILL.md"
print("mirror OK")
EOF
```

Expected: `mirror OK`

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "feat: mirror skill rules into root CLAUDE.md"
```

---

### Task 4: Dataset fixture with planted traps

**Files:**
- Create: `benchmark/fixtures/make_dataset.py`

The benchmark prompt asks the agent to predict churn from a CSV this script generates. Three traps are planted; rigorous code must handle all three: (1) `account_status` is target-derived (perfect leak), (2) 200 duplicated rows that must not straddle the split, (3) `signup_date` is temporal, so a forward-looking claim needs a time-aware split or an explicit justification.

- [ ] **Step 1: Write `benchmark/fixtures/make_dataset.py`**

```python
"""Generate the benchmark churn dataset. Deterministic for a given seed.

Planted rigor traps (documented here, NOT in the prompt the agent sees):
- account_status: derived from the target ("closed" iff churned). Perfect leak.
- 200 exact duplicate rows appended; a random split lets them straddle train/test.
- signup_date: temporal column; random splits ignore time.
"""
import argparse

import numpy as np
import pandas as pd


def make(seed: int = 7, n: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    spend = rng.gamma(2.0, 30.0, n).round(2)
    tickets = rng.poisson(1.2, n)
    signup = pd.Timestamp("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 900, n), unit="D"
    )
    logit = -1.2 - 0.03 * tenure + 0.01 * spend + 0.45 * tickets
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "signup_date": signup.strftime("%Y-%m-%d"),
            "tenure_months": tenure,
            "monthly_spend": spend,
            "support_tickets": tickets,
            "account_status": np.where(churn == 1, "closed", "active"),
            "churned": churn,
        }
    )
    dup = df.sample(n=200, random_state=seed)
    return pd.concat([df, dup], ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="churn.csv")
    args = parser.parse_args()
    make(args.seed).to_csv(args.out, index=False)
    print(f"wrote {args.out}")
```

- [ ] **Step 2: Verify it runs, is deterministic, and traps are present**

```bash
cd benchmark/fixtures
python3 make_dataset.py --out /tmp/churn_a.csv
python3 make_dataset.py --out /tmp/churn_b.csv
cmp /tmp/churn_a.csv /tmp/churn_b.csv && echo deterministic
python3 - <<'EOF'
import pandas as pd
df = pd.read_csv("/tmp/churn_a.csv")
assert len(df) == 4200
assert (df.duplicated().sum()) >= 200, "duplicate trap missing"
leak = (df["account_status"].eq("closed") == df["churned"].eq(1)).all()
assert leak, "leak trap missing"
rate = df["churned"].mean()
assert 0.10 < rate < 0.60, f"churn rate degenerate: {rate}"
print(f"traps OK, churn rate {rate:.2f}")
EOF
cd ../..
```

Expected: `deterministic`, then `traps OK, churn rate 0.NN`.

- [ ] **Step 3: Commit**

```bash
git add benchmark/fixtures/make_dataset.py
git commit -m "feat: add benchmark dataset fixture with planted rigor traps"
```

---

### Task 5: Benchmark codegen prompt

**Files:**
- Create: `benchmark/prompts/ml-experiment-v1.txt`

- [ ] **Step 1: Write `benchmark/prompts/ml-experiment-v1.txt`** with exactly this content (note: it must NOT hint at the traps):

```text
You are working in an empty project workspace that contains one file: make_dataset.py.

Task: build a small, CPU-only ML experiment that answers: "For predicting customer churn on this dataset, does gradient boosting outperform logistic regression?"

Requirements:
1. Generate the dataset by running: python3 make_dataset.py --out churn.csv
   The CSV columns are: customer_id, signup_date, tenure_months, monthly_spend, support_tickets, account_status, churned (binary target).
2. Implement an experiment in Python using scikit-learn that compares LogisticRegression and GradientBoostingClassifier for predicting `churned`.
3. Choose and justify the evaluation methodology yourself (splits, preprocessing, metrics, number of runs). Use only the provided dataset.
4. Produce:
   - src/ with the experiment code
   - run_experiment.py entrypoint that runs the full experiment and writes results to results/ (machine-readable metrics) and REPORT.md (the comparison conclusion, methodology, and limitations)
   - tests/ with pytest tests for your pipeline
   - pyproject.toml or requirements.txt declaring dependencies
5. The full experiment must finish in under 5 minutes on CPU.
6. Run your tests and the experiment before finishing.

When you are done, output exactly one JSON object (no other text after it):
{"files_created": [...], "entrypoint": "<command to run the experiment>", "test_command": "<command to run the tests>"}
```

- [ ] **Step 2: Sanity-check the prompt does not mention the traps**

```bash
! grep -iE "leak|duplicate|time-based|temporal|account_status is" benchmark/prompts/ml-experiment-v1.txt && echo "no trap hints"
```

Expected: `no trap hints`

- [ ] **Step 3: Commit**

```bash
git add benchmark/prompts/
git commit -m "feat: add ml-experiment-v1 benchmark prompt"
```

---

### Task 6: Review rubric

**Files:**
- Create: `benchmark/rubric/ml-experiment-rigor-review-v1.md`

- [ ] **Step 1: Write `benchmark/rubric/ml-experiment-rigor-review-v1.md`** with exactly this content:

````markdown
# ml-experiment-rigor-review-v1

You are reviewing a generated ML experiment project in the current working directory. Judge how rigorous the experiment is as evidence, not how pretty the code is.

## Procedure

1. Inventory the files. Read the experiment code, tests, and REPORT.md.
2. Run the declared test command and the declared entrypoint. If a command would clearly exceed 120 seconds, do not wait it out — record that fact and score from static evidence instead.
3. Audit, in order:
   - **Leakage:** Is `account_status` used as a feature? (It is derived from the target; using it is a critical leak.) Are fit-like transforms (scaler, imputer, encoders, feature selection) fitted on train only? Do duplicate rows straddle the train/test split? Is the temporal column handled deliberately (time-based split or an explicit justification)?
   - **Methodology:** Trivial baseline present? Metrics appropriate for the class balance (not accuracy alone)? Same tuning budget for both models?
   - **Reproducibility:** Seeds fixed and logged? Config/metrics written to results/? Re-running with the same seed reproducible?
   - **Claims:** Does REPORT.md claim a winner? Backed by how many runs/folds, with what variance? Are limitations honest? Does the report claim anything the code did not measure?
   - **Executability:** Do the declared commands actually work from a clean state?
   - **Code quality:** Readable structure, no dead code, dependencies declared and used.

## Output contract

End your review with exactly one fenced JSON block (```json ... ```) and nothing after it. The JSON must contain ALL of these fields:

```json
{
  "rubric_version": "ml-experiment-rigor-review-v1",
  "score_scale": "0-100 per category",
  "weights": {
    "leakage_prevention": 0.25,
    "methodological_validity": 0.20,
    "reproducibility": 0.15,
    "claims_discipline": 0.15,
    "executability": 0.15,
    "code_quality": 0.10
  },
  "leakage_prevention_score": 0,
  "methodological_validity_score": 0,
  "reproducibility_score": 0,
  "claims_discipline_score": 0,
  "executability_score": 0,
  "code_quality_score": 0,
  "verdict": "excellent|good|mixed|poor",
  "key_findings": ["..."],
  "leakage_findings": ["..."],
  "commands_run": ["..."],
  "file_evidence": [{"path": "...", "note": "..."}]
}
```

Scores are integers 0-100. Do not output a weighted total; it is recomputed downstream. Do not wrap the JSON in any other code fence or prose after it.
````

- [ ] **Step 2: Commit**

```bash
git add benchmark/rubric/
git commit -m "feat: add review rubric ml-experiment-rigor-review-v1"
```

---

### Task 7: analyze.py — strict parser (TDD)

**Files:**
- Create: `benchmark/analyze.py`
- Test: `benchmark/tests/test_analyze.py`

- [ ] **Step 1: Write the failing tests** in `benchmark/tests/test_analyze.py`:

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import REQUIRED_CATEGORIES, extract_review, mann_whitney, recompute_weighted

VALID = """Review prose here.
```json
{
  "rubric_version": "ml-experiment-rigor-review-v1",
  "score_scale": "0-100 per category",
  "weights": {"leakage_prevention": 0.25, "methodological_validity": 0.20,
              "reproducibility": 0.15, "claims_discipline": 0.15,
              "executability": 0.15, "code_quality": 0.10},
  "leakage_prevention_score": 80, "methodological_validity_score": 70,
  "reproducibility_score": 60, "claims_discipline_score": 50,
  "executability_score": 90, "code_quality_score": 40,
  "verdict": "mixed", "key_findings": [], "leakage_findings": [],
  "commands_run": [], "file_evidence": []
}
```
"""

# Regression: real failure observed in programming-as-theory-building benchmark
# (karpathy_only/run_08): a colon inside the quoted key broke the whole review.
COLON_TYPO = VALID.replace(
    '"verdict": "mixed"', '"verdict: is mixed and stuff",\n  "verdict": "mixed"'
)

# Regression: real failure (skills_off/run_09): reviewer cut off mid-review.
TRUNCATED = "I've scheduled a check. Waiting for the expiration tests to complete."

MISSING_FIELD = VALID.replace('"code_quality_score": 40,', "")


def test_extract_valid():
    d = extract_review(VALID)
    assert d["verdict"] == "mixed"
    assert all(f"{c}_score" in d for c in REQUIRED_CATEGORIES)


def test_extract_rejects_colon_typo():
    with pytest.raises(ValueError, match="json"):
        extract_review(COLON_TYPO)


def test_extract_rejects_truncated():
    with pytest.raises(ValueError, match="fence"):
        extract_review(TRUNCATED)


def test_extract_rejects_missing_field():
    with pytest.raises(ValueError, match="missing"):
        extract_review(MISSING_FIELD)


def test_extract_rejects_out_of_range():
    bad = VALID.replace('"executability_score": 90', '"executability_score": 150')
    with pytest.raises(ValueError, match="range"):
        extract_review(bad)


def test_recompute_weighted_ignores_self_report():
    d = extract_review(VALID)
    d["weighted_total"] = 999  # self-reported value must be ignored
    expected = 80 * 0.25 + 70 * 0.20 + 60 * 0.15 + 50 * 0.15 + 90 * 0.15 + 40 * 0.10
    assert recompute_weighted(d) == pytest.approx(expected)


def test_mann_whitney_separated_vs_identical():
    lo = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
    hi = [v + 50 for v in lo]
    _, _, p_sep = mann_whitney(hi, lo)
    _, _, p_same = mann_whitney(lo, lo)
    assert p_sep < 0.01
    assert p_same > 0.9
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd benchmark && python3 -m pytest tests/test_analyze.py -q; cd ..
```

Expected: FAIL with `ModuleNotFoundError: No module named 'analyze'`.

- [ ] **Step 3: Write `benchmark/analyze.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd benchmark && python3 -m pytest tests/test_analyze.py -q; cd ..
```

Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add benchmark/analyze.py benchmark/tests/test_analyze.py
git commit -m "feat: add strict review parser and stats reporter with tests"
```

---

### Task 8: Codegen harness

**Files:**
- Create: `benchmark/run_codegen_experiment.sh`

- [ ] **Step 1: Write `benchmark/run_codegen_experiment.sh`**

```bash
#!/usr/bin/env bash
# Run codegen arms in fresh workspaces.
# Usage: MODEL=haiku REPEATS=10 ARMS="skills_off rigor_only" ./run_codegen_experiment.sh
set -uo pipefail

MODEL="${MODEL:-haiku}"
REPEATS="${REPEATS:-10}"
ARMS="${ARMS:-skills_off rigor_only}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
OUT="$ROOT/runs/codegen/$RUN_ID"
mkdir -p "$OUT"
printf 'arm\trun\tstatus\n' > "$OUT/manifest.tsv"

for arm in $ARMS; do
  mkdir -p "$OUT/$arm/workspaces"
  for i in $(seq -f "%02g" 1 "$REPEATS"); do
    ws="$OUT/$arm/workspaces/run_$i"
    mkdir -p "$ws"
    cp "$ROOT/fixtures/make_dataset.py" "$ws/"
    if [ "$arm" = "rigor_only" ]; then
      mkdir -p "$ws/.claude/skills/ml-experiment-rigor"
      cp "$ROOT/../skills/ml-experiment-rigor/SKILL.md" \
         "$ws/.claude/skills/ml-experiment-rigor/SKILL.md"
    fi
    cp "$ROOT/prompts/ml-experiment-v1.txt" "$OUT/$arm/prompt_$i.txt"
    echo ">> codegen $arm run_$i"
    ( cd "$ws" && claude --print --model "$MODEL" --dangerously-skip-permissions \
        < "$ROOT/prompts/ml-experiment-v1.txt" ) \
      > "$OUT/$arm/run_$i.txt" 2> "$OUT/$arm/run_$i.stderr"
    printf '%s\trun_%s\t%s\n' "$arm" "$i" "$?" >> "$OUT/manifest.tsv"
  done
done

echo "codegen run complete: $OUT"
```

- [ ] **Step 2: Make executable and smoke-test with REPEATS=1**

```bash
chmod +x benchmark/run_codegen_experiment.sh
MODEL=haiku REPEATS=1 ./benchmark/run_codegen_experiment.sh
```

Expected: prints `codegen run complete: .../runs/codegen/<run_id>`. Then verify:

```bash
RUN=$(ls -t benchmark/runs/codegen | head -1)
cat benchmark/runs/codegen/$RUN/manifest.tsv          # both arms status 0
ls benchmark/runs/codegen/$RUN/rigor_only/workspaces/run_01/   # generated project files
grep -c "ml-experiment-rigor" benchmark/runs/codegen/$RUN/rigor_only/workspaces/run_01/.claude/skills/ml-experiment-rigor/SKILL.md
```

If the `rigor_only` workspace shows no sign the skill changed behavior (e.g., output identical in shape to `skills_off`), check that project-level skills are actually loaded by `claude --print` in that workspace before proceeding — this is the harness's key assumption.

- [ ] **Step 3: Commit**

```bash
git add benchmark/run_codegen_experiment.sh
git commit -m "feat: add codegen benchmark harness"
```

---

### Task 9: Review harness with run-time JSON validation

**Files:**
- Create: `benchmark/run_review_experiment.sh`

- [ ] **Step 1: Write `benchmark/run_review_experiment.sh`**

```bash
#!/usr/bin/env bash
# Review generated workspaces with a stronger model and validate JSON immediately.
# Usage: MODEL=opus ./run_review_experiment.sh runs/codegen/<run_id>
set -uo pipefail

MODEL="${MODEL:-opus}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
CODEGEN_DIR="${1:?usage: run_review_experiment.sh <codegen run dir>}"
[ -d "$CODEGEN_DIR" ] || CODEGEN_DIR="$ROOT/$1"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
OUT="$ROOT/runs/review/$RUN_ID"
mkdir -p "$OUT"
printf 'arm\trun\texit\tjson_valid\tworkspace\n' > "$OUT/manifest.tsv"
echo "codegen_source: $CODEGEN_DIR" > "$OUT/source.txt"

for ws in "$CODEGEN_DIR"/*/workspaces/run_*; do
  [ -d "$ws" ] || continue
  run="$(basename "$ws")"
  arm="$(basename "$(dirname "$(dirname "$ws")")")"
  dest="$OUT/$arm/$run"
  mkdir -p "$dest"
  cp "$ROOT/rubric/ml-experiment-rigor-review-v1.md" "$dest/prompt.txt"
  echo ">> review $arm $run"
  ( cd "$ws" && claude --print --model "$MODEL" --dangerously-skip-permissions \
      < "$dest/prompt.txt" ) > "$dest/review.txt" 2> "$dest/review.stderr"
  status=$?
  if python3 "$ROOT/analyze.py" validate "$dest/review.txt" > /dev/null 2>&1; then
    valid=1
  else
    valid=0
    echo "!! invalid review output: $arm $run" >&2
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$arm" "$run" "$status" "$valid" "$ws" \
    >> "$OUT/manifest.tsv"
done

echo "review run complete: $OUT"
echo "invalid reviews (rerun these): $(awk -F'\t' '$4 == 0 && NR > 1' "$OUT/manifest.tsv" | wc -l | tr -d ' ')"
```

This closes the gap found in the reference benchmark, where two unusable reviews were recorded as status 0 and only discovered at analysis time.

- [ ] **Step 2: Make executable and smoke-test against the Task 8 smoke run**

```bash
chmod +x benchmark/run_review_experiment.sh
RUN=$(ls -t benchmark/runs/codegen | head -1)
MODEL=opus ./benchmark/run_review_experiment.sh "benchmark/runs/codegen/$RUN"
```

Expected: `review run complete: .../runs/review/<id>` and `invalid reviews (rerun these): 0`. Then:

```bash
REVIEW=$(ls -t benchmark/runs/review | head -1)
python3 benchmark/analyze.py report "benchmark/runs/review/$REVIEW"
```

Expected: per-arm line with `n=1 mean=NN.N sd=0.0` for each arm (no significance lines at n=1 — that's correct behavior, the guard requires n>1).

- [ ] **Step 3: Commit**

```bash
git add benchmark/run_review_experiment.sh
git commit -m "feat: add review harness with run-time JSON validation"
```

---

### Task 10: README and benchmark README

**Files:**
- Create: `README.md`, `benchmark/README.md`

- [ ] **Step 1: Write `README.md`** with this content:

````markdown
# ML Experiment Rigor Skill

A Claude Code skill that distills Andrej Karpathy's "A Recipe for Training Neural Networks" and Kapoor & Narayanan's data-leakage taxonomy into operating rules for agents writing ML training and evaluation code.

An ML experiment that runs is not the same as an ML experiment that proves something. Coding agents reliably produce the former: pipelines that fit scalers on the full dataset, use target-derived features, declare winners from a single seed, and write reports stronger than their evidence. This skill forces the agent to treat the experiment as an argument — leak surface enumerated before coding, sanity checks before full training, repetition and variance before any comparative claim.

## What the skill enforces

- **Data discipline:** split before transform, target-leak hunting, duplicate-aware splits, time-aware splits for temporal data.
- **Sanity checks first:** trivial baseline, overfit-a-tiny-subset, label-shuffle test, init-loss check.
- **Seed and variance discipline:** logged seeds, ≥3 repeats behind any comparison, mean ± sd reporting.
- **Claims discipline:** no winner without variance; "no detectable difference" is a valid result; test set touched once.

See [skills/ml-experiment-rigor/SKILL.md](skills/ml-experiment-rigor/SKILL.md) for the full rules.

## Benchmark

The repo ships a complete isolated-arm benchmark (harness included, unlike most skill repos): a churn-prediction experiment prompt over a dataset with three planted traps — a perfectly target-derived feature, duplicate rows that straddle naive splits, and a temporal column that random splits ignore. Generated projects are reviewed by a stronger model against `ml-experiment-rigor-review-v1`, scoring leakage prevention, methodological validity, reproducibility, claims discipline, executability, and code quality.

Results: pending first full run. See [benchmark/README.md](benchmark/README.md) for method, run-level results with n/sd/significance, and reproduction commands.

## Install

Option A: Claude Code plugin

```text
/plugin marketplace add Noverse0/ml-experiment-rigor-skill
/plugin install ml-experiment-rigor-skill
```

Option B: copy the skill directly

```bash
mkdir -p ~/.claude/skills/ml-experiment-rigor
curl -fsSL https://raw.githubusercontent.com/Noverse0/ml-experiment-rigor-skill/main/skills/ml-experiment-rigor/SKILL.md \
  -o ~/.claude/skills/ml-experiment-rigor/SKILL.md
```

## Reproduce the benchmark

```bash
MODEL=haiku REPEATS=10 ARMS="skills_off rigor_only" ./benchmark/run_codegen_experiment.sh
MODEL=opus ./benchmark/run_review_experiment.sh benchmark/runs/codegen/<run_id>
python3 benchmark/analyze.py report benchmark/runs/review/<run_id>
```

The analyzer refuses to silently drop unparseable reviews (they are listed and the exit code is non-zero), recomputes weighted totals from category scores, and reports per-arm n, sd, and Mann-Whitney significance alongside means.

## Lineage

Companion to [programming-as-theory-building-skill](https://github.com/AnamKwon/programming-as-theory-building-skill), which applies the same recipe — canonical text → operating rules → isolated-arm benchmark — to general program modification.
````

- [ ] **Step 2: Write `benchmark/README.md`** with this content:

````markdown
# Benchmark Notes

Isolated-arm comparison of ML experiment codegen with and without the rigor skill.

## Method

- Arms: `skills_off` (no skill in workspace), `rigor_only` (skill at `.claude/skills/ml-experiment-rigor/` in the workspace). Override with `ARMS`.
- Every generation runs in a fresh workspace seeded only with `fixtures/make_dataset.py`.
- Codegen model: Claude Haiku (`MODEL=haiku`). Review model: Claude Opus.
- Prompt: `prompts/ml-experiment-v1.txt` — compare logistic regression vs gradient boosting for churn prediction. The prompt does not hint at the traps.
- Planted traps in the dataset: `account_status` is target-derived (perfect leak); 200 duplicated rows; `signup_date` is temporal.
- Rubric: `rubric/ml-experiment-rigor-review-v1.md`. Weighted score over leakage prevention (0.25), methodological validity (0.20), reproducibility (0.15), claims discipline (0.15), executability (0.15), code quality (0.10).
- Review outputs are JSON-validated at run time (`manifest.tsv` column `json_valid`); invalid reviews are re-run, not silently excluded.
- `analyze.py report` recomputes weighted totals from category scores and reports per-arm n, mean, sd, verdict counts, and pairwise Mann-Whitney z/p. Winners are only claimed when the difference is significant.

## Results

Pending first full run (`REPEATS=10` per arm). This section will contain the run-level table (run id → per-arm n/mean/sd → significance) produced by `analyze.py report`.

## Reproduce

```bash
MODEL=haiku REPEATS=10 ./run_codegen_experiment.sh
MODEL=opus ./run_review_experiment.sh runs/codegen/<run_id>
python3 analyze.py report runs/review/<run_id>
```

`runs/` is gitignored; curated raw results are published by copying selected run folders into `raw-results/` explicitly.
````

- [ ] **Step 3: Commit**

```bash
git add README.md benchmark/README.md
git commit -m "docs: add README and benchmark method docs"
```

---

### Task 11: Publish to GitHub

**Files:** none (repo operation)

- [ ] **Step 1: Confirm gh account is Noverse0**

```bash
gh auth status 2>&1 | head -4
```

Expected: `Active account: true` under `Noverse0`. If not: `gh auth switch --user Noverse0`.

- [ ] **Step 2: Create and push**

```bash
cd /Users/rohdaeyoung/workspace/ml-experiment-rigor-skill
gh repo create Noverse0/ml-experiment-rigor-skill --public --source=. --push \
  --description "ML experiment rigor rules for coding agents, with an isolated-arm benchmark"
```

Expected: repo URL printed; `git push` succeeds.

- [ ] **Step 3: Verify install path works**

```bash
gh repo view Noverse0/ml-experiment-rigor-skill --json url -q .url
curl -fsSL https://raw.githubusercontent.com/Noverse0/ml-experiment-rigor-skill/main/skills/ml-experiment-rigor/SKILL.md | head -5
```

Expected: frontmatter of SKILL.md printed.

---

### Task 12: Full benchmark run and results write-up (long-running; run after review of Tasks 1–11)

**Files:**
- Modify: `README.md` (Results), `benchmark/README.md` (Results)

- [ ] **Step 1: Full codegen run** (expect ~1–3 h wall clock depending on model throughput; run in background)

```bash
MODEL=haiku REPEATS=10 ./benchmark/run_codegen_experiment.sh
```

- [ ] **Step 2: Review run**

```bash
RUN=$(ls -t benchmark/runs/codegen | head -1)
MODEL=opus ./benchmark/run_review_experiment.sh "benchmark/runs/codegen/$RUN"
```

- [ ] **Step 3: Re-run any invalid reviews** — check `manifest.tsv` for `json_valid=0` rows; for each, re-invoke the single review (same command as the harness uses for one workspace) until `analyze.py validate` passes or the failure is documented.

- [ ] **Step 4: Analyze and write results**

```bash
REVIEW=$(ls -t benchmark/runs/review | head -1)
python3 benchmark/analyze.py report "benchmark/runs/review/$REVIEW" | tee /tmp/bench_report.txt
```

Replace the "Results: pending" sections in both READMEs with the actual table: per-arm n/mean/sd, verdict counts, category means, and the Mann-Whitney line. State the honest conclusion — including "no detectable difference" if p is not small. The skill's own claims-discipline rules apply to its own benchmark.

- [ ] **Step 5: Publish curated raw results**

```bash
mkdir -p benchmark/raw-results
cp -R "benchmark/runs/codegen/$RUN" benchmark/raw-results/codegen-$RUN
cp -R "benchmark/runs/review/$REVIEW" benchmark/raw-results/review-$REVIEW
find benchmark/raw-results -name "*.csv" -delete
find benchmark/raw-results -name "__pycache__" -type d -exec rm -rf {} +
git add -f benchmark/raw-results README.md benchmark/README.md
git commit -m "docs: add first benchmark results with raw run data"
git push
```

---

## Acceptance Criteria

- `python3 -m pytest benchmark/tests/ -q` passes (7 tests).
- `skills/ml-experiment-rigor/SKILL.md` has valid frontmatter; CLAUDE.md is an exact frontmatter-stripped mirror.
- Smoke run (REPEATS=1) produces a parseable review for both arms with `json_valid=1` in the review manifest.
- `analyze.py report` on the smoke run prints per-arm stats and exits 0.
- Repo is public at `Noverse0/ml-experiment-rigor-skill` and the raw SKILL.md URL serves the file.
- After Task 12: READMEs contain real results with n/sd/significance, and no winner claim unsupported by the stats.

## Risks / Notes

- **Skill loading assumption:** the harness assumes `claude --print` loads project-level skills from `<workspace>/.claude/skills/`. Verified in the Task 8 smoke test; if it does not hold, fall back to injecting the SKILL.md body into the prompt for the `rigor_only` arm (and say so in the method docs).
- **Reviewer calibration:** the rubric names the planted traps explicitly, so the reviewer does not need to discover them. This is deliberate — it measures the *generator's* rigor, not the reviewer's.
- **Cost:** Task 12 is ~20 codegen runs + ~20 Opus reviews. Run smoke first; don't start Task 12 until Tasks 1–11 are reviewed.
