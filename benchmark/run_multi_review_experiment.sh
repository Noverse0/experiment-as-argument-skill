#!/usr/bin/env bash
# Review generated workspaces with MULTIPLE independent reviewer models and
# validate each review's JSON immediately. One review file per (reviewer, run).
#
# Why: a single reviewer shares the author's blind spots. Independent reviewers
# that fail differently surface disagreement, which is the signal — the most
# uncertain claims and the first place to look (see SKILL.md "Independent Review").
#
# Usage:
#   REVIEWERS="claude:opus codex:gpt-5 gemini:gemini-2.5-pro" \
#   RUBRIC=experiment-as-argument-review-v2.md \
#     ./run_multi_review_experiment.sh runs/codegen/<run_id>
#
# Each REVIEWERS entry is "<cli>:<model>". Supported clis: claude, codex, gemini.
set -uo pipefail

REVIEWERS="${REVIEWERS:-claude:opus}"
RUBRIC="${RUBRIC:-experiment-as-argument-review-v1.md}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
[ -f "$ROOT/rubric/$RUBRIC" ] || { echo "no such rubric: $RUBRIC" >&2; exit 2; }
CODEGEN_DIR="${1:?usage: run_multi_review_experiment.sh <codegen run dir>}"
[ -d "$CODEGEN_DIR" ] || CODEGEN_DIR="$ROOT/$1"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
OUT="$ROOT/runs/multi-review/$RUN_ID"
mkdir -p "$OUT"
printf 'reviewer\tarm\trun\texit\tjson_valid\tattempts\tworkspace\n' > "$OUT/manifest.tsv"
echo "codegen_source: $CODEGEN_DIR" > "$OUT/source.txt"
echo "reviewers: $REVIEWERS" >> "$OUT/source.txt"

# Invoke one reviewer CLI non-interactively from inside the workspace dir.
# Stdin is the rubric prompt; stdout must end with the rubric's JSON block.
# --setting-sources local (claude) / read-only sandbox (codex) / plan mode
# (gemini) keep the reviewer from loading the workspace's own skill or editing.
run_reviewer() {
  local cli="$1" model="$2" ws="$3" prompt="$4"
  case "$cli" in
    claude)
      ( cd "$ws" && claude --print --model "$model" \
          --setting-sources local --dangerously-skip-permissions < "$prompt" ) ;;
    codex)
      ( cd "$ws" && codex exec --model "$model" \
          --sandbox read-only --skip-git-repo-check - < "$prompt" ) ;;
    gemini)
      ( cd "$ws" && gemini --model "$model" --approval-mode plan \
          -p "$(cat "$prompt")" ) ;;
    *)
      echo "unknown reviewer cli: $cli" >&2; return 99 ;;
  esac
}

for entry in $REVIEWERS; do
  cli="${entry%%:*}"; model="${entry#*:}"
  for ws in "$CODEGEN_DIR"/*/workspaces/run_*; do
    [ -d "$ws" ] || continue
    run="$(basename "$ws")"
    arm="$(basename "$(dirname "$(dirname "$ws")")")"
    dest="$OUT/$cli/$arm/$run"
    mkdir -p "$dest"
    cp "$ROOT/rubric/$RUBRIC" "$dest/prompt.txt"
    echo ">> review [$cli:$model] $arm $run"
    attempt=0; status=1; valid=0
    while [ "$attempt" -lt 3 ]; do
      attempt=$((attempt + 1))
      run_reviewer "$cli" "$model" "$ws" "$dest/prompt.txt" \
        > "$dest/review.txt" 2> "$dest/review.stderr"
      status=$?
      if python3 "$ROOT/analyze.py" validate "$dest/review.txt" > /dev/null 2>&1; then
        valid=1; break
      fi
      echo "!! invalid review output, retrying ($cli $arm $run attempt $attempt)" >&2
    done
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$cli" "$arm" "$run" "$status" "$valid" "$attempt" "$ws" >> "$OUT/manifest.tsv"
  done
done

echo "multi-review run complete: $OUT"
echo "invalid reviews (rerun these): $(awk -F'\t' '$5 == 0 && NR > 1' "$OUT/manifest.tsv" | wc -l | tr -d ' ')"
