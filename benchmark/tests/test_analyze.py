import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import (
    REQUIRED_CATEGORIES,
    extract_review,
    mann_whitney,
    recompute_weighted,
    render_csv,
    render_markdown,
    review_spread,
)

VALID = """Review prose here.
```json
{
  "rubric_version": "experiment-as-argument-review-v1",
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


def _sample_stats():
    return [
        {"arm": "skills_off", "n": 10, "mean": 81.6, "sd": 4.6,
         "cats": {c: 80.0 for c in REQUIRED_CATEGORIES}, "verdicts": {"good": 10}},
        {"arm": "rigor_only", "n": 10, "mean": 78.9, "sd": 12.2,
         "cats": {c: 75.0 for c in REQUIRED_CATEGORIES}, "verdicts": {"good": 9, "poor": 1}},
    ]


def test_render_markdown_has_table_and_significance():
    md = render_markdown(
        _sample_stats(),
        [{"a": "rigor_only", "b": "skills_off", "diff": -2.7, "z": 0.08, "p": 0.94}],
        [],
    )
    assert "| Arm |" in md
    assert "skills_off" in md and "rigor_only" in md
    assert "81.6" in md
    assert "0.94" in md
    assert "good 10" in md


def test_render_csv_header_and_rows():
    out = render_csv(_sample_stats())
    lines = out.strip().splitlines()
    assert lines[0].startswith("arm,n,weighted_mean,sd,")
    assert any(line.startswith("skills_off,10,81.6,") for line in lines)
    assert len(lines) == 3  # header + 2 arms


# NOTE: the leak-check feature (classify_leak / `analyze.py leak-check`) was
# removed after exp-005. It could not distinguish a real main-pipeline leak from
# an intentional leakage-ceiling demo (the skill induces the latter), so it
# systematically misclassified rigor_only runs. See benchmark/EXPERIMENTS.md
# exp-005 + correction. Static syntax checks don't capture the domain invariant.


def test_review_spread_tight_vs_wide():
    assert review_spread([80.0, 82.0, 81.0]) == pytest.approx(2.0)
    assert review_spread([60.0, 90.0]) == pytest.approx(30.0)


def test_review_spread_single_reviewer_is_zero():
    assert review_spread([75.0]) == 0.0


def test_review_spread_empty_is_zero():
    assert review_spread([]) == 0.0
