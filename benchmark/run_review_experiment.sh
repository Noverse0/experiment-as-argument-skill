#!/usr/bin/env bash
# Review generated workspaces with a stronger model and validate JSON immediately.
# Usage: MODEL=opus ./run_review_experiment.sh runs/codegen/<run_id>
# Match the rubric to the dataset variant used at codegen time (default v1):
#   RUBRIC=experiment-as-argument-review-v2.md ./run_review_experiment.sh <dir>
set -uo pipefail

MODEL="${MODEL:-opus}"
RUBRIC="${RUBRIC:-experiment-as-argument-review-v1.md}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
[ -f "$ROOT/rubric/$RUBRIC" ] || { echo "no such rubric: $RUBRIC" >&2; exit 2; }
CODEGEN_DIR="${1:?usage: run_review_experiment.sh <codegen run dir>}"
[ -d "$CODEGEN_DIR" ] || CODEGEN_DIR="$ROOT/$1"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
OUT="$ROOT/runs/review/$RUN_ID"
mkdir -p "$OUT"
printf 'arm\trun\texit\tjson_valid\tattempts\tworkspace\n' > "$OUT/manifest.tsv"
echo "codegen_source: $CODEGEN_DIR" > "$OUT/source.txt"

for ws in "$CODEGEN_DIR"/*/workspaces/run_*; do
  [ -d "$ws" ] || continue
  run="$(basename "$ws")"
  arm="$(basename "$(dirname "$(dirname "$ws")")")"
  dest="$OUT/$arm/$run"
  mkdir -p "$dest"
  cp "$ROOT/rubric/$RUBRIC" "$dest/prompt.txt"
  echo ">> review $arm $run"
  # Validity is judged by analyze.py (strict JSON contract), not exit code:
  # truncated or malformed reviews are retried instead of silently recorded.
  attempt=0
  status=1
  valid=0
  while [ "$attempt" -lt 3 ]; do
    attempt=$((attempt + 1))
    # --setting-sources local: the reviewer must not load the workspace's
    # project skill (present only in rigor_only arms) nor the operator's
    # user-level config — otherwise reviewer conditions differ between arms.
    ( cd "$ws" && claude --print --model "$MODEL" --dangerously-skip-permissions \
        --setting-sources local \
        < "$dest/prompt.txt" ) > "$dest/review.txt" 2> "$dest/review.stderr"
    status=$?
    if python3 "$ROOT/analyze.py" validate "$dest/review.txt" > /dev/null 2>&1; then
      valid=1
      break
    fi
    echo "!! invalid review output, retrying ($arm $run attempt $attempt)" >&2
  done
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$run" "$status" "$valid" "$attempt" "$ws" \
    >> "$OUT/manifest.tsv"
done

echo "review run complete: $OUT"
echo "invalid reviews (rerun these): $(awk -F'\t' '$4 == 0 && NR > 1' "$OUT/manifest.tsv" | wc -l | tr -d ' ')"
